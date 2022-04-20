import asyncio
import contextlib
import dataclasses
import grp
import inspect
import json
import logging
import os
import re
import shutil
import shlex
import subprocess
import sys
import textwrap
from collections import OrderedDict
from functools import cached_property
from fnmatch import fnmatch
from pathlib import Path
from random import choices
from string import ascii_lowercase, digits, hexdigits
from timeit import default_timer as timer
from typing import (
    Generator,
    Iterable,
    List,
    MutableMapping,
    Mapping,
    Optional,
    Tuple,
    Union,
)
from urllib.request import urlretrieve, urlopen
from urllib.parse import urlencode
from urllib.error import HTTPError
from zipfile import Path as ZipPath

import jinja2
import pytest
import pytest_asyncio.plugin
import yaml
from _pytest.config import Config
from _pytest.config.argparsing import Parser
from juju.client.jujudata import FileJujuData
from juju.model import Model, Controller, websockets

log = logging.getLogger(__name__)


def pytest_addoption(parser: Parser):
    parser.addoption(
        "--cloud",
        action="store",
        help="Juju cloud to use; if not provided, will "
        "use the default for the controller",
    )
    parser.addoption(
        "--controller",
        action="store",
        help="Juju controller to use; if not provided, "
        "will use the current controller",
    )
    parser.addoption(
        "--model-alias",
        action="store",
        help="Alias name used for the model created by ops_test",
        default="main",
    )
    parser.addoption(
        "--model",
        action="store",
        help="Juju model to use; if not provided, a new model "
        "will be created for each test which requires one",
    )
    parser.addoption(
        "--keep-models",
        action="store_true",
        help="Keep any automatically created models",
    )
    parser.addoption(
        "--destructive-mode",
        action="store_true",
        help="Whether to run charmcraft in destructive mode "
        "(as opposed to doing builds in lxc containers)",
    )
    parser.addoption(
        "--no-crash-dump",
        action="store_true",
        help="Disabled automatic runs of juju-crashdump after failed tests, "
        "juju-crashdump runs by default.",
    )
    parser.addoption(
        "--crash-dump-output",
        action="store",
        default=None,
        help="Store the completed crash dump in this dir. "
        "The default is current working directory.",
    )
    parser.addoption(
        "--no-deploy",
        action="store_true",
        help="This, together with the `--model` parameter, ensures that all functions "
        "marked with the` skip_if_deployed` tag are skipped.",
    )
    parser.addoption(
        "--model-config",
        action="store",
        default=None,
        help="path to a yaml file which will be applied to the model on creation. "
        "* ignored if `--model` supplied"
        "* if the specified file doesn't exist, an error will be raised.",
    )


def pytest_load_initial_conftests(parser: Parser, args: List[str]) -> None:
    known_args = parser.parse_known_args(args)
    if known_args.no_deploy and known_args.model is None:
        optparser = parser._getparser()
        optparser.error("must specify --model when using --no-deploy")


def pytest_configure(config: Config):
    config.addinivalue_line("markers", "abort_on_fail")
    config.addinivalue_line("markers", "skip_if_deployed")
    # These need to be fixed in libjuju and just clutter things up for tests using this.
    config.addinivalue_line(
        "filterwarnings", "ignore:The loop argument:DeprecationWarning"
    )
    config.addinivalue_line(
        "filterwarnings", r"ignore:'with \(yield from lock\)':DeprecationWarning"
    )


def pytest_runtest_setup(item):
    if (
        "skip_if_deployed" in item.keywords
        and item.config.getoption("--no-deploy")
        and item.config.getoption("--model") is not None
    ):
        pytest.skip("Skipping deployment because --no-deploy was specified.")


@pytest.fixture(scope="session")
def tmp_path_factory(request):
    # Override temp path factory to create temp dirs under Tox env so that
    # confined snaps (e.g., charmcraft) can access them.
    return pytest.TempPathFactory(
        given_basetemp=Path(os.environ["TOX_ENV_DIR"]) / "tmp" / "pytest",
        trace=request.config.trace.get("tmpdir"),
        _ispytest=True,
    )


def check_deps(*deps):
    missing = []
    for dep in deps:
        res = subprocess.run(["which", dep], capture_output=True)
        if res.returncode != 0:
            missing.append(dep)
    if missing:
        raise RuntimeError(
            "Missing dependenc{}: {}".format(
                "y" if len(missing) == 1 else "ies",
                ", ".join(missing),
            )
        )


@pytest.fixture(scope="module")
def event_loop():
    """Create an instance of the default event loop for each test module."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Plugin load order can't be set, replace asyncio directly
pytest_asyncio.plugin.event_loop = event_loop


def pytest_collection_modifyitems(session, config, items):
    """Automatically apply the "asyncio" marker to any async test items."""
    for item in items:
        is_async = inspect.iscoroutinefunction(getattr(item, "function", None))
        has_marker = item.get_closest_marker("asyncio")
        if is_async and not has_marker:
            item.add_marker("asyncio")


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Make test results available to fixture finalizers."""
    # execute all other hooks to obtain the report object
    outcome = yield
    rep = outcome.get_result()

    # set a report attribute for each phase of a call, which can
    # be "setup", "call", "teardown"
    setattr(item, "rep_" + rep.when, rep)

    # set attribute which indicates fail / xfail in any phase
    item.failed = getattr(item, "failed", False) or rep.failed
    item.xfailed = getattr(item, "xfailed", False) or getattr(rep, "wasxfail", False)


@pytest.fixture(autouse=True)
def abort_on_fail(request):
    if OpsTest._instance is None:
        # If we don't have an ops_test already in play, this should be a no-op.
        yield
        return
    ops_test = OpsTest._instance
    if ops_test.aborted:
        pytest.xfail("aborted")

    yield
    abort_on_fail = request.node.get_closest_marker("abort_on_fail")
    failed = getattr(request.node, "failed", False)
    if abort_on_fail and abort_on_fail.kwargs.get("abort_on_xfail", False):
        failed = failed or request.node.xfailed
    if failed and abort_on_fail:
        ops_test.aborted = True


@pytest.fixture(scope="module")
@pytest.mark.asyncio
async def ops_test(request, tmp_path_factory):
    check_deps("juju", "charmcraft")
    ops_test = OpsTest(request, tmp_path_factory)
    await ops_test._setup_model()
    OpsTest._instance = ops_test
    yield ops_test
    OpsTest._instance = None
    await ops_test._cleanup_models()


def handle_file_delete_error(function, path, execinfo):
    log.warning(f"Failed to delete '{path}' due to {execinfo[1]}")


class FileResource:
    """Represents a File based Resource."""

    """
    Some resources are arch specific but don't include amd64 in the name
    they'll be identified as being named <base><tail> where there is
    another resource with <base>-<arch><tail>
    """
    ARCH_RE = re.compile(r"^(\S+)(?:-(amd64|arm64|s390x))(\S+)?")

    def __init__(self, name, filename, arch=None):
        self.name = name
        self.filename = filename
        self.name_without_arch = self.name
        self.arch = arch
        matches = self.ARCH_RE.match(self.name)
        if matches:
            base, arch, tail = matches.groups()
            self.name_without_arch = f"{base}{tail or ''}"
            self.arch = arch

    def __repr__(self):
        return f"FileResource('{self.name}','{self.filename}','{self.arch}')"

    @property
    def download_path(self):
        return Path(self.name) / self.filename


def json_request(url, params=None):
    if params:
        url = f"{url}?{urlencode(params)}"
    with urlopen(url) as resp:
        if 200 <= resp.status < 300:
            return json.loads(resp.read())


class Charmhub:
    """
    Fetch resources from Charmhub
    API DOCS: https://api.snapcraft.io/docs/charms.html
    """

    CH_URL = "https://api.charmhub.io/v2"

    def __init__(self, charmhub_name, channel):
        self._name = charmhub_name
        self._channel = channel

    @cached_property
    def info(self):
        params = dict(channel=self._channel, fields="default-release.resources")
        url = f"{self.CH_URL}/charms/info/{self._name}"
        try:
            return json_request(url, params)
        except HTTPError as ex:
            raise RuntimeError(f"Charm {self._name} not found in charmhub.") from ex

    @property
    def exists(self):
        try:
            return bool(self.info)
        except RuntimeError:
            return False

    @cached_property
    def resource_map(self):
        return {rsc["name"]: rsc for rsc in self.info["default-release"]["resources"]}

    def download_resource(self, resource, destination: Path):
        rsc = self.resource_map[resource]
        log.info(f"Retrieving {resource} from charmhub...")
        destination.parent.mkdir(parents=True, exist_ok=True)
        target, _msg = urlretrieve(rsc["download"]["url"], destination)
        return target


class CharmStore:
    CS_URL = "https://api.jujucharms.com/charmstore/v5"

    def __init__(self, charmstore_name, channel="edge"):
        self._name = charmstore_name
        self._channel = channel

    @staticmethod
    def _charmpath(charm):
        if charm.startswith("cs:"):
            return charm[3:]
        return charm

    @cached_property
    def _charm_id(self):
        params = dict(channel=self._channel)
        url = f"{self.CS_URL}/{self._charmpath(self._name)}/meta/id"
        try:
            resp = json_request(url, params)
        except HTTPError as ex:
            raise RuntimeError(
                f"Charm {self._name} not found in charmstore at channel={self._channel}"
            ) from ex
        return resp["Id"]

    @property
    def exists(self):
        try:
            return bool(self._charm_id)
        except RuntimeError:
            return False

    def download_resource(self, resource, destination: Path):
        charm_id = self._charmpath(self._charm_id)
        url = f"{self.CS_URL}/{charm_id}/meta/resources/{resource}"
        try:
            resp = json_request(url)
        except HTTPError as ex:
            raise RuntimeError(
                f"Charm {charm_id} {resource} not found in charmstore"
            ) from ex
        rev = resp["Revision"]
        log.info(f"Retrieving {resource} from charmstore...")
        url = f"{self.CS_URL}/{charm_id}/resource/{resource}/{rev}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        local_file, header = urlretrieve(url, destination)
        return local_file


class ModelNotFoundError(Exception):
    """Raise when referencing to a model that doesn't exist."""


class ModelInUseError(Exception):
    """Raise when trying to add a model alias which already exists."""


@dataclasses.dataclass
class ModelState:
    model: Model
    keep: bool
    controller_name: str
    cloud_name: Optional[str]
    model_name: str
    config: Optional[dict] = None
    tmp_path: Optional[Path] = None

    @property
    def full_name(self) -> str:
        return f"{self.controller_name}:{self.model_name}"


class OpsTest:
    """Utility class for testing Operator Charms."""

    _instance = None  # store instance, so we can tell if it's been used yet

    def __init__(self, request, tmp_path_factory):
        self.request = request
        self._tmp_path_factory = tmp_path_factory
        self._global_tmp_path = None

        # Flag indicating whether all subsequent tests should be aborted.
        self.aborted = False

        # Flag for using destructive mode or not for charm builds.
        self.destructive_mode = request.config.option.destructive_mode

        # Config options to determine first model specs used by tests
        self._orig_model_alias = request.config.option.model_alias
        self._init_cloud_name = request.config.option.cloud
        self._init_model_name = request.config.option.model
        self._init_keep_model = request.config.option.keep_models

        # These may be modified by _setup_model
        self.controller_name = request.config.option.controller
        self._init_model_config = request.config.option.model_config

        # Flag for enabling the juju-crashdump
        self.crash_dump = not request.config.option.no_crash_dump
        self.crash_dump_output = request.config.option.crash_dump_output

        # These will be set by _setup_model
        self.jujudata = None
        self._controller: Optional[Controller] = None

        # maintains a set of all models connected by this fixture
        # use an OrderedDict so that the first model made is destroyed last.
        self._current_alias = None
        self._models: MutableMapping[str, ModelState] = OrderedDict()

    @contextlib.contextmanager
    def model_context(self, alias: str) -> Generator[Model, None, None]:
        """
        Analog to `juju switch` where the focus of the current model is moved.
        """
        prior = self.current_alias
        model = self._switch(alias)
        try:
            yield model
        finally:
            # if the there's a failure after yielding, don't fail to
            # switch back to the prior alias but still raise whatever
            # error condition occurred through the context
            self._switch(prior, raise_not_found=False)

    def _switch(self, alias: str, raise_not_found=True) -> Model:
        if alias in self._models:
            self._current_alias = alias
        elif not raise_not_found:
            self._current_alias = None
        else:
            raise ModelNotFoundError(f"{alias} not found")

        return self.model

    @property
    def current_alias(self) -> Optional[str]:
        return self._current_alias

    @property
    def models(self) -> Mapping[str, ModelState]:
        """Returns the dict of managed models by this fixture."""
        return {k: dataclasses.replace(v) for k, v in self._models.items()}

    @property
    def tmp_path(self) -> Path:
        tmp_path = self._global_tmp_path
        current_state = self.current_alias and self._models.get(self.current_alias)
        if current_state and current_state.tmp_path is None:
            tmp_path = self._tmp_path_factory.mktemp(current_state.model_name)
            current_state.tmp_path = tmp_path
        elif current_state and current_state.tmp_path:
            tmp_path = current_state.tmp_path
        elif not tmp_path:
            tmp_path = self._global_tmp_path = self._tmp_path_factory.mktemp(
                self.default_model_name
            )
        log.info(f"Using tmp_path: {tmp_path}")
        return tmp_path

    @property
    def model_config(self) -> Optional[dict]:
        """Represents the config used when adding the model."""
        current_state = self.current_alias and self._models.get(self.current_alias)
        return current_state.config if current_state else None

    @property
    def model(self) -> Optional[Model]:
        """Represents the current model."""
        current_state = self.current_alias and self._models.get(self.current_alias)
        return current_state.model if current_state else None

    @property
    def model_full_name(self) -> Optional[str]:
        """Represents the current model's full name."""
        current_state = self.current_alias and self._models.get(self.current_alias)
        return current_state.full_name if current_state else None

    @property
    def model_name(self) -> Optional[str]:
        """Represents the current model name."""
        current_state = self.current_alias and self._models.get(self.current_alias)
        return current_state.model_name if current_state else None

    @property
    def cloud_name(self) -> Optional[str]:
        """Represents the current model's cloud name."""
        current_state = self.current_alias and self._models.get(self.current_alias)
        return current_state.cloud_name if current_state else None

    @property
    def keep_model(self) -> bool:
        """Represents whether the current model should be kept after tests."""
        if self._init_keep_model:
            return True
        current_state = self.current_alias and self._models.get(self.current_alias)
        return current_state.keep if current_state else False

    def _generate_model_name(self) -> str:
        module_name = self.request.module.__name__.rpartition(".")[-1]
        suffix = "".join(choices(ascii_lowercase + digits, k=4))
        return f"{module_name.replace('_', '-')}-{suffix}"

    @cached_property
    def default_model_name(self) -> str:
        return self._generate_model_name()

    async def run(
        self,
        *cmd: str,
        cwd: Optional[os.PathLike] = None,
        check: bool = False,
        fail_msg: Optional[str] = None,
        stdin: Optional[bytes] = None,
    ) -> Tuple[Optional[int], str, str]:
        """Asynchronously run a subprocess command.

        @param                   str cmd: command to execute within a juju context
        @param Optional[os.Pathlink] cwd: current working directory
        @param                bool check: if False, returns a tuple (rc, stdout, stderr)
                                          if True,  calls `pytest.fail` with `fail_msg`
                                          and relevant command information
        @param Optional[str]    fail_msg: Message to present if check=True and rc != 0
        @param Optional[bytes]     stdin: Bytes read by stdin of the called process
        """
        env = {**os.environ}
        if self.jujudata:
            env["JUJU_DATA"] = self.jujudata.path
        if self.model_full_name:
            env["JUJU_MODEL"] = self.model_full_name

        if not isinstance(stdin, bytes) and stdin is not None:
            raise TypeError("'stdin' parameter must be a Optional[bytes] typed")

        proc = await asyncio.create_subprocess_exec(
            *(str(c) for c in cmd),
            stdin=asyncio.subprocess.PIPE if isinstance(stdin, bytes) else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd or "."),
            env=env,
        )

        _stdout, _stderr = await proc.communicate(input=stdin)
        stdout, stderr = _stdout.decode("utf8"), _stderr.decode("utf8")
        if check and proc.returncode != 0:
            if fail_msg is None:
                fail_msg = f"Command {list(cmd)} failed"
            raise AssertionError(
                f"{fail_msg} ({proc.returncode}): {(stderr or stdout).strip()}"
            )
        return proc.returncode, stdout, stderr

    _run = run  # backward compatibility alias

    async def juju(self, *args, **kwargs):
        """Runs a Juju CLI command.

        Useful for cases where python-libjuju sees things differently than the Juju CLI.
        Will set `JUJU_MODEL`, so manually passing in `-m model-name` is unnecessary.
        """

        return await self.run("juju", *args, **kwargs)

    async def _add_model(
        self, controller_name, cloud_name, model_name, keep=False, **kwargs
    ):
        """
        Creates a model used by the test framework which would normally be destroyed
        after the tests are run in the module.
        """
        controller = self._controller
        if not controller:
            controller = Controller()
            await controller.connect(controller_name)
        if not cloud_name:
            # if not provided, try the default cloud name
            cloud_name = self._init_cloud_name
        if not cloud_name:
            # if not provided, use the controller's default cloud
            cloud_name = await controller.get_cloud()
        model_full_name = f"{controller_name}:{model_name}"
        log.info(f"Adding model {model_full_name} on cloud {cloud_name}")

        model = await controller.add_model(model_name, cloud_name, **kwargs)
        # NB: This call to `juju models` is needed because libjuju's
        # `add_model` doesn't update the models.yaml cache that the Juju
        # CLI depends on with the model's UUID, which the CLI requires to
        # connect. Calling `juju models` beforehand forces the CLI to
        # update the cache from the controller.
        await self.juju("models")
        state = ModelState(model, keep, controller_name, cloud_name, model_name)
        state.config = await model.get_config()
        return state

    async def _model_exists(self, model_name: str) -> bool:
        """
        returns True when the model_name exists in the model.
        """
        all_models = await self._controller.list_models()
        return model_name in all_models

    @staticmethod
    async def _connect_to_model(controller_name, model_name, keep=True):
        """
        Makes a reference to an existing model used by the test framework
        which will not be destroyed after the tests are run in the module.
        """
        model = Model()
        state = ModelState(model, keep, controller_name, None, model_name)
        log.info(
            "Connecting to existing model %s on unspecified cloud", state.full_name
        )
        await model.connect(state.full_name)
        state.config = await model.get_config()
        return state

    @staticmethod
    def read_model_config(
        config_path_or_obj: Union[dict, str, os.PathLike, None]
    ) -> Optional[dict]:
        if isinstance(config_path_or_obj, dict):
            return config_path_or_obj
        model_config = None
        if config_path_or_obj:
            model_config_file = Path(config_path_or_obj)
            if not model_config_file.exists():
                log.error("model-config file %s doesn't exist", model_config_file)
                raise FileNotFoundError(model_config_file)
            else:
                log.info("Loading model config from %s", model_config_file)
                model_config = yaml.safe_load(model_config_file.read_text())
        return model_config

    async def _setup_model(self):
        # TODO: We won't need this if Model.debug_log is implemented in libjuju
        self.jujudata = FileJujuData()
        alias = self._orig_model_alias
        if not self.controller_name:
            self.controller_name = self.jujudata.current_controller()
        if not self._init_model_name:
            # no --model flag specified, automatically generate a model
            config = self.read_model_config(self._init_model_config)
            model_state = await self._add_model(
                self.controller_name,
                self._init_cloud_name,
                self.default_model_name,
                config=config,
            )
        else:
            # --model flag specified, reuse existing model and set keep flag
            model_state = await self._connect_to_model(
                self.controller_name, self._init_model_name
            )

        if not self._controller:
            self._controller = await model_state.model.get_controller()

        self._models[alias] = model_state
        self._current_alias = alias

    async def track_model(
        self,
        alias: str,
        model_name: Optional[str] = None,
        cloud_name: Optional[str] = None,
        use_existing: Optional[bool] = None,
        keep: Optional[bool] = None,
        **kwargs,
    ) -> Model:
        """
        Track a new or existing model in the existing controller.

        @param str           alias     : alias to the model used only by ops_test
                                         to differentiate between models.
        @param Optional[str] model_name: name of the new model to track,
                                         None will craft a unique name
        @param Optional[str] cloud_name: cloud name in which to add a new model,
                                         None will use current cloud
        @param Optional[bool] use_existing:
               None:  True if model_name exists on this controller
               False: create a new model and keep=False, unless keep=True explicitly set
               True:  connect to a model and keep=True, unless keep=False explicitly set
        @param Optional[bool] keep:
               None:  follows the value of use_existing
               False: tracked model is destroyed at tests end
               True:  tracked model remains once tests complete
               --keep-models flag will override this options

        Common Examples:
        ----------------------------------
        # make a new model with any juju name and destroy it when the tests are over
        await ops_test.track_model("alias")

        # make or reuse a model known to juju as "bob"
        # don't destroy model if it existed, destroy it if it didn't already exist
        await ops_test.track_model("alias", model_name="bob")
        ----------------------------------
        """
        if alias in self._models:
            raise ModelInUseError(f"Cannot add new model with alias '{alias}'")

        if model_name and use_existing is None:
            use_existing = await self._model_exists(model_name)

        keep = bool(use_existing) if keep is None else keep
        if use_existing:
            if not model_name:
                raise NotImplementedError(
                    "Cannot use_existing model if model_name is unspecified"
                )
            model_state = await self._connect_to_model(
                self.controller_name, model_name, keep
            )
        else:
            cloud_name = cloud_name or self.cloud_name
            model_name = model_name or self._generate_model_name()
            model_state = await self._add_model(
                self.controller_name, cloud_name, model_name, keep, **kwargs
            )
        self._models[alias] = model_state
        return model_state.model

    async def log_model(self):
        """Log a summary of the status of the model."""
        # TODO: Implement a pretty model status in libjuju
        _, stdout, _ = await self.juju("status")
        log.info(f"Model status:\n\n{stdout}")

        # TODO: Implement Model.debug_log in libjuju
        _, stdout, _ = await self.juju(
            "debug-log", "--replay", "--no-tail", "--level", "ERROR"
        )
        log.info(f"Juju error logs:\n\n{stdout}")

    async def create_crash_dump(self) -> bool:
        """Run the juju-crashdump if it's possible."""
        cmd = shlex.split(
            f"juju-crashdump -s -m {self.model_full_name} -a debug-layer -a config"
        )

        output_directory = self.crash_dump_output
        if output_directory:
            log.debug("juju-crashdump will use output dir `%s`", output_directory)
            cmd.append("-o")
            cmd.append(output_directory)

        try:
            return_code, stdout, stderr = await self.run(*cmd)
            log.info("juju-crashdump finished [%s]", return_code)
            return True
        except FileNotFoundError:
            log.info("juju-crashdump command was not found.")
            return False

    async def forget_model(
        self,
        alias: str,
        timeout: Optional[Union[float, int]] = None,
        allow_failure: bool = True,
    ):
        """
        Forget a model and wait for it to be removed from the controller.
        If the model is marked as kept, ops_tests forgets about this model immediately.
        If the model is not marked as kept, ops_test will destroy the model.
        If timeout is None don't wait on the model to be completely destroyed

        @param                   str alias: alias of the model
        @param Optional[float,int] timeout: how long to wait for it to be removed,
                                            if None, don't block waiting for success
        @param          bool allow_failure: if False, failures raise an exception
        """
        if not self._controller:
            log.error("No access to controller, skipping...")
            return

        if alias not in self.models:
            raise ModelNotFoundError(f"{alias} not found")

        with self.model_context(alias) as model:
            await self.log_model()
            model_name = model.info.name

            # NOTE (rgildein): Create juju-crashdump only if any tests failed,
            # `juju-crashdump` flag is enabled and OpsTest.keep_model == False
            if (
                self.request.session.testsfailed > 0
                and self.crash_dump
                and self.keep_model is False
            ):
                await self.create_crash_dump()

            if not self.keep_model:
                await self._reset(model, allow_failure, timeout=timeout)
                await self._controller.destroy_model(model_name, force=True)
            else:
                await model.disconnect()

        # stop managing this model now
        log.info(f"Forgetting {alias}...")
        self._models.pop(alias)
        if alias is self.current_alias:
            self._current_alias = None

    @staticmethod
    async def _reset(model: Model, allow_failure, timeout: Optional[int] = None):
        # Forcibly destroy applications/machines in case any units are in error.
        log.info(f"Resetting model {model.info.name}...")
        for app in model.applications.values():
            log.info(f"   Destroying application {app.name}")
            await app.destroy()

        for machine in model.machines.values():
            log.info(f"  Destroying machine {machine.id}")
            await machine.destroy(force=True)

        if timeout is None:
            log.info("Not waiting on reset to complete.")
            return

        try:
            await model.block_until(
                lambda: len(model.machines) == 0 and len(model.applications) == 0,
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            log.exception(f"Timeout resetting {model.info.name}")
            if not allow_failure:
                raise
        except websockets.ConnectionClosed:
            log.error(f"Disconnected while resetting {model.info.name}")
            if not allow_failure:
                raise
        else:
            log.info(f"Reset {model.info.name} completed successfully.")

    async def _cleanup_models(self):
        if not self.models:
            return

        # remove models from most recently made, to first made
        aliases = list(reversed(self._models.keys()))
        for models in aliases:
            await self.forget_model(models)

        await self._controller.disconnect()

    # maintain backwards compatibility (though this was a private method)
    _cleanup_model = _cleanup_models

    def abort(self, *args, **kwargs):
        """Fail the current test method and mark all remaining test methods as xfail.

        This can be used if a given step is required for subsequent steps to be
        successful, such as the initial deployment.

        Any args will be passed through to `pytest.fail()`.

        You can also mark a test with `@pytest.marks.abort_on_fail` to have this
        automatically applied if the marked test method fails or errors.
        """
        self.aborted = True
        pytest.fail(*args, **kwargs)

    async def build_charm(self, charm_path) -> Path:
        """Builds a single charm.

        This can handle charms using the older charms.reactive framework as
        well as charms written against the modern operator framework.

        Returns a Path for the built charm file.
        """
        charms_dst_dir = self.tmp_path / "charms"
        charms_dst_dir.mkdir(exist_ok=True)
        charm_path = Path(charm_path)
        charm_abs = Path(charm_path).absolute()
        metadata_path = charm_path / "metadata.yaml"
        layer_path = charm_path / "layer.yaml"
        charm_name = yaml.safe_load(metadata_path.read_text())["name"]
        if layer_path.exists():
            # Handle older, reactive framework charms.
            check_deps("charm")
            cmd = ["charm", "build", "--charm-file"]
        else:
            # Handle newer, operator framework charms.
            all_groups = {g.gr_name for g in grp.getgrall()}
            users_groups = {grp.getgrgid(g).gr_name for g in os.getgroups()}
            if self.destructive_mode:
                # host builder never requires lxd group
                cmd = ["charmcraft", "pack", "--destructive-mode"]
            elif "lxd" in users_groups:
                # user already has lxd group active
                cmd = ["charmcraft", "pack"]
            else:
                # building with lxd builder and user does't already have lxd group;
                # make sure it's available and if so, try using `sg` to acquire it
                assert "lxd" in all_groups, (
                    "Group 'lxd' required but not available; "
                    "ensure that lxd is available or use --destructive-mode"
                )
                cmd = ["sg", "lxd", "-c", "charmcraft pack"]

        log.info(f"Building charm {charm_name}")
        start = timer()
        returncode, stdout, stderr = await self.run(*cmd, cwd=charm_abs)
        elapsed = timer() - start
        if returncode == 0:
            log.info(f"Built charm {charm_name} in {elapsed:.2f}s")
        else:
            log.info(
                f"Charm build for {charm_name} completed with errors (return "
                f"code={returncode}) in {elapsed:.2f}s"
            )

        if not layer_path.exists():
            # Clean up build dir created by charmcraft.
            build_path = charm_path / "build"
            if build_path.exists():
                # In some rare cases, some files under the created 'build' dir have
                # odd permissions which interfer with cleanup; just log and continue.
                shutil.rmtree(build_path, onerror=handle_file_delete_error)

        if returncode != 0:
            m = re.search(
                r"Failed to build charm.*full execution logs in '([^']+)'", stderr
            )
            if m:
                try:
                    stderr = Path(m.group(1)).read_text()
                except FileNotFoundError:
                    log.error(f"Failed to read full build log from {m.group(1)}")
            raise RuntimeError(
                f"Failed to build charm {charm_path}:\n{stderr}\n{stdout}"
            )

        charm_file_src = next(charm_abs.glob(f"{charm_name}*.charm"))
        charm_file_dst = charms_dst_dir / charm_file_src.name
        charm_file_src.rename(charm_file_dst)
        return charm_file_dst

    async def build_charms(self, *charm_paths) -> Mapping[str, Path]:
        """Builds one or more charms in parallel.

        This can handle charms using the older charms.reactive framework as
        well as charms written against the modern operator framework.

        Returns a mapping of charm names to Paths for the built charm files.
        """
        charms = await asyncio.gather(
            *(self.build_charm(charm_path) for charm_path in charm_paths)
        )
        return {charm.stem.split("_")[0]: charm for charm in charms}

    @staticmethod
    def charm_file_resources(built_charm: Path):
        """
        Locate all file-typed resources to download from store.

        Flag architecture specific file resources by presence of
        arch names in the resource.  Supported arches are `amd64`, `arm64`, and
        `s390x`.  If there is a resource that shares the same base and tail of
        another arch specific resource but doesn't include an arch (e.g., `cni.tgz`
        and `cni-s390x.tgz` both exist), assume that the unspecified arch is `amd64`.

        Non-architecture specific files will have a None in the `arch` field
        """

        if not built_charm.exists():
            raise FileNotFoundError(f"Failed to locate built charm {built_charm}")

        charm_path = ZipPath(built_charm)
        metadata_path = charm_path / "metadata.yaml"
        resources = yaml.safe_load(metadata_path.read_text())["resources"]

        resources = {
            name: FileResource(name, resource.get("filename"), None)
            for name, resource in resources.items()
            if resource.get("type") == "file"
        }

        potentials = {rsc.name_without_arch for rsc in resources.values() if rsc.arch}
        for rsc_name in potentials:
            if resources.get(rsc_name):
                resources[rsc_name].arch = "amd64"
        return resources

    def arch_specific_resources(self, build_charm):
        return {
            name: rsc
            for name, rsc in self.charm_file_resources(build_charm).items()
            if rsc.arch
        }

    async def build_resources(self, build_script: Path):
        build_script = build_script.absolute()
        if not build_script.exists():
            raise FileNotFoundError(
                f"Failed to locate resource build script {build_script}"
            )

        log.info("Build Resources...")
        dst_dir = self.tmp_path / "resources"
        dst_dir.mkdir(exist_ok=True)
        start = timer()
        rc, stdout, stderr = await self.run(
            *shlex.split(f"sudo {build_script}"), cwd=dst_dir, check=False
        )
        if rc != 0:
            log.warning(f"{build_script} failed: {(stderr or stdout).strip()}")
        else:
            elapsed = timer() - start
            log.info(f"Built resources in {elapsed:.2f}s")
        return list(dst_dir.glob("*.*"))

    @staticmethod
    def _charm_name(built_charm: Path):
        if not built_charm.exists():
            raise FileNotFoundError(f"Failed to locate built charm {built_charm}")

        charm_path = ZipPath(built_charm)
        metadata_path = charm_path / "metadata.yaml"
        return yaml.safe_load(metadata_path.read_text())["name"]

    async def download_resources(
        self, built_charm: Path, owner="", channel="edge", resources=None
    ):
        """
        Download Resources associated with a local charm.

        @param Path built_charm: path to local charm
        @param str  owner:   if the charm is associated with an owner in the charmstore
                             or namespace in charmhub
        @param str  channel: channel to pull resources associated with the local charm
        @param dict[str,FileResource] resources: specific resources associated with
                                                 this local charm
        """

        charm_name = (f"{owner}-" if owner else "") + self._charm_name(built_charm)
        downloader: Union[Charmhub, CharmStore] = Charmhub(charm_name, channel)
        if not downloader.exists:
            charm_name = (f"~{owner}/" if owner else "") + self._charm_name(built_charm)
            downloader = CharmStore(charm_name, channel)
        if not downloader.exists:
            raise RuntimeError(
                f"Cannot find {charm_name} in either Charmstore or Charmhub"
            )

        dst_dir = self.tmp_path / "resources"
        dl = downloader.download_resource
        resources = resources or self.charm_file_resources(built_charm)
        return {
            resource.name: dl(resource.name, dst_dir / resource.download_path)
            for resource in resources.values()
        }

    async def build_bundle(
        self,
        bundle: Optional[str] = None,
        output_bundle: Optional[str] = None,
        serial: bool = False,
    ):
        """Builds bundle using juju-bundle build."""
        cmd = ["juju-bundle", "build"]
        if bundle is not None:
            cmd += ["--bundle", bundle]
        if output_bundle is not None:
            cmd += ["--output-bundle", output_bundle]
        if self.destructive_mode:
            cmd += ["--destructive-mode"]
        if serial:
            cmd += ["--serial"]
        await self.run(*cmd, check=True)

    async def deploy_bundle(
        self,
        bundle: Optional[str] = None,
        build: bool = True,
        serial: bool = False,
        extra_args: Iterable[str] = (),
    ):
        """Deploys bundle using juju-bundle deploy."""
        cmd = ["juju-bundle", "deploy"]
        if bundle is not None:
            cmd += ["--bundle", bundle]
        if build:
            cmd += ["--build"]
        if self.destructive_mode:
            cmd += ["--destructive-mode"]
        if serial:
            cmd += ["--serial"]

        cmd += ["--"] + list(extra_args)

        log.info(
            "Deploying (and possibly building) bundle using juju-bundle command:"
            f"'{' '.join(cmd)}'"
        )
        await self.run(*cmd, check=True)

    def render_bundle(self, bundle, context=None, **kwcontext):
        """Render a templated bundle using Jinja2.

        This can be used to populate built charm paths or config values.

        :param bundle (str or Path): Path to bundle file or YAML content.
        :param context (dict): Optional context mapping.
        :param **kwcontext: Additional optional context as keyword args.

        Returns the Path for the rendered bundle.
        """
        bundles_dst_dir = self.tmp_path / "bundles"
        bundles_dst_dir.mkdir(exist_ok=True)
        if context is None:
            context = {}
        context.update(kwcontext)
        if re.search(r".yaml(.j2)?$", str(bundle)):
            bundle_path = Path(bundle)
            bundle_text = bundle_path.read_text()
            if bundle_path.suffix == ".j2":
                bundle_name = bundle_path.stem
            else:
                bundle_name = bundle_path.name
        else:
            bundle_text = textwrap.dedent(bundle).strip()
            infix = "".join(choices(hexdigits, k=4))
            bundle_name = f"{self.model_name}-{infix}.yaml"
        log.info(f"Rendering bundle {bundle_name}")
        rendered = jinja2.Template(bundle_text).render(**context)
        dst = bundles_dst_dir / bundle_name
        dst.write_text(rendered)
        return dst

    def render_bundles(self, *bundles, context=None, **kwcontext):
        """Render one or more templated bundles using Jinja2.

        This can be used to populate built charm paths or config values.

        :param *bundles (str or Path): One or more bundle Paths or YAML contents.
        :param context (dict): Optional context mapping.
        :param **kwcontext: Additional optional context as keyword args.

        Returns a list of Paths for the rendered bundles.
        """
        # Jinja2 does support async, but rendering bundles should be relatively quick.
        return [
            self.render_bundle(bundle_path, context=context, **kwcontext)
            for bundle_path in bundles
        ]

    async def build_lib(self, lib_path):
        """Build a Python library (sdist) for use in a test.

        Returns a Path for the built library archive file.
        """
        libs_dst_dir = self.tmp_path / "libs"
        libs_dst_dir.mkdir(exist_ok=True)
        lib_path_abs = Path(lib_path).absolute()

        returncode, stdout, stderr = await self.run(
            sys.executable, "setup.py", "--fullname", cwd=lib_path_abs
        )
        if returncode != 0:
            raise RuntimeError(
                f"Failed to get library name {lib_path}:\n{stderr}\n{stdout}"
            )
        lib_name_ver = stdout.strip()
        lib_dst_path = libs_dst_dir / f"{lib_name_ver}.tar.gz"

        log.info(f"Building library {lib_path}")
        returncode, stdout, stderr = await self.run(
            sys.executable, "setup.py", "sdist", "-d", libs_dst_dir, cwd=lib_path_abs
        )
        if returncode != 0:
            raise RuntimeError(
                f"Failed to build library {lib_path}:\n{stderr}\n{stdout}"
            )

        return lib_dst_path

    def render_charm(
        self, charm_path, include=None, exclude=None, context=None, **kwcontext
    ):
        """Render a templated charm using Jinja2.

        This can be used to make certain files in a test charm templated, such
        as a path to a library file that is built locally.

        Note: Because charmcraft builds charms in a LXD container, any files
        referenced by the charm will need to be relative to the charm directory.
        To make this work as transparently as possible, any Path values in the
        context will be copied into the rendered charm directory and the values
        changed to point to that copy instead. This won't work if the file
        reference is a string, or if it's under a nested data structure.

        :param charm_path (str): Path to top-level directory of charm to render.
        :include (list[str or Path]): Optional list of glob patterns or file paths
            to pass through Jinja2, relative to base charm path. (default: all files
            are passed through Jinja2)
        :exclude (list[str or Path]): Optional list of glob patterns or file paths
            to exclude from passing through Jinja2, relative to the base charm path.
            (default: all files are passed through Jinja2)
        :param context (dict): Optional context mapping.
        :param **kwcontext: Additional optional context as keyword args.

        Returns a Path for the rendered charm source directory.
        """
        context = dict(context or {})  # make a copy, since we modify it
        context.update(kwcontext)
        charm_path = Path(charm_path)
        charm_dst_path = self.tmp_path / "charms" / charm_path.name
        log.info(f"Rendering charm {charm_path}")
        shutil.copytree(
            charm_path,
            charm_dst_path,
            ignore=shutil.ignore_patterns(".git", ".bzr", "__pycache__", "*.pyc"),
        )

        suffix = "".join(choices(ascii_lowercase + digits, k=4))
        files_path = charm_dst_path / f"_files_{suffix}"
        files_path.mkdir()
        for k, v in context.items():
            if not isinstance(v, Path):
                continue
            # account for possibility of file name collisions
            dst_dir = files_path / "".join(choices(ascii_lowercase + digits, k=4))
            dst_dir.mkdir()
            dst_path = dst_dir / v.name
            shutil.copy2(v, dst_dir)
            context[k] = Path("/root/project") / dst_path.relative_to(charm_dst_path)

        if include is None:
            include = ["*"]
        if exclude is None:
            exclude = []

        def _filter(root, node):
            # Filter nodes based on whether they match include and don't match exclude.
            rel_node = (Path(root) / node).relative_to(charm_dst_path)
            if not any(fnmatch(rel_node, pat) for pat in include):
                return False
            if any(fnmatch(rel_node, pat) for pat in exclude):
                return False
            return True

        for root, dirs, files in os.walk(charm_dst_path):
            dirs[:] = [dn for dn in dirs if _filter(root, dn)]
            files[:] = [fn for fn in files if _filter(root, fn)]
            for file_name in files:
                file_path = Path(root) / file_name
                file_text = file_path.read_text()
                rendered = jinja2.Template(file_text).render(**context)
                file_path.write_text(rendered)

        return charm_dst_path

    def render_charms(
        self, *charm_paths, include=None, exclude=None, context=None, **kwcontext
    ):
        """Render one or more templated charms using Jinja2.

        This can be used to make certain files in a test charm templated, such
        as a path to a library file that is built locally.

        :param *charm_paths (str): Path to top-level directory of charm to render.
        :include (list[str or Path]): Optional list of glob patterns or file paths
            to pass through Jinja2, relative to base charm path. (default: all files
            are passed through Jinja2)
        :exclude (list[str or Path]): Optional list of glob patterns or file paths
            to exclude from passing through Jinja2, relative to the base charm path.
            (default: all files are passed through Jinja2)
        :param context (dict): Optional context mapping.
        :param **kwcontext: Additional optional context as keyword args.

        Returns a list of Paths for the rendered charm source directories.
        """
        # Jinja2 does support async, but rendering individual files should be
        # relatively quick, meaning this will end up blocking on the IO from
        # os.walk() and Path.read/write_text() most of the time anyway.
        return [
            self.render_charm(charm_path, include, exclude, context, **kwcontext)
            for charm_path in charm_paths
        ]
