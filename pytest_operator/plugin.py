import asyncio
import base64
import contextlib
import dataclasses
import enum
import grp
import inspect
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
from collections import OrderedDict
from fnmatch import fnmatch
from functools import cached_property
from pathlib import Path
from random import choices
from string import ascii_lowercase, digits, hexdigits
from timeit import default_timer as timer
from typing import (
    Any,
    Dict,
    Generator,
    Iterable,
    List,
    Literal,
    Mapping,
    MutableMapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
    overload,
)
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen, urlretrieve
from zipfile import Path as ZipPath

import jinja2
import kubernetes.config
import pytest
import pytest_asyncio.plugin
import yaml
from _pytest.config import Config
from _pytest.config.argparsing import Parser
from juju.client import client
from juju.client.jujudata import FileJujuData
from juju.errors import JujuError
from juju.exceptions import DeadEntityException
from juju.model import Controller, Model, websockets
from kubernetes import client as k8s_client
from kubernetes.client import Configuration as K8sConfiguration

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
        help="Keep models handled by opstest, can be overriden in track_model",
    )
    parser.addoption(
        "--destroy-storage",
        action="store_true",
        help="Destroy storage created in models handled by opstest,"
        "can be overriden in track_model",
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
        help="(Deprecated - use '--crash-dump=never' instead.  Overrides anything"
        " specified in '--crash-dump')\n"
        "Disable automatic runs of juju-crashdump after failed tests, "
        "juju-crashdump runs by default.",
    )
    parser.addoption(
        "--crash-dump",
        action="store",
        default="legacy",
        help="Sets whether to output a juju-crashdump after tests.  Options are:\n"
        "* always: dumps after all tests\n"
        "* on-failure: dumps after failed tests\n"
        "* legacy: (DEFAULT) dumps after a failed test if '--keep-models' is False\n"
        "* never: never dumps",
    )
    parser.addoption(
        "--crash-dump-args",
        action="store",
        default="",
        help="If crashdump is run, run with provided extra arguments.",
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
    parser.addoption(
        "--juju-max-frame-size",
        action="store",
        default=None,
        help="Set the maximum frame size for websocket communication with Juju.",
        type=int,
    )
    parser.addoption(
        "--charmcraft-args",
        action="append",
        help="Set extra charmcraft args.",
        default=[],
    )


def pytest_load_initial_conftests(parser: Parser, args: List[str]) -> None:
    known_args = parser.parse_known_args(args)
    if known_args.no_deploy and known_args.model is None:
        optparser = parser._getparser()
        optparser.error("must specify --model when using --no-deploy")


def pytest_configure(config: Config):
    config.addinivalue_line("markers", "abort_on_fail")
    config.addinivalue_line("markers", "skip_if_deployed")

    if config.option.basetemp is None:
        tox_dir = os.environ.get("TOX_ENV_DIR")
        if tox_dir:
            config.option.basetemp = Path(tox_dir) / "tmp/pytest"


def pytest_runtest_setup(item):
    if (
        "skip_if_deployed" in item.keywords
        and item.config.getoption("--no-deploy")
        and item.config.getoption("--model") is not None
    ):
        pytest.skip("Skipping deployment because --no-deploy was specified.")


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


@pytest_asyncio.fixture(scope="module")
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


def validate_crash_dump(crash_dump: str, no_crash_dump: bool):
    """Validates the crash-dump inputs, raising if they are not accepted values."""
    if no_crash_dump:
        log.warning(
            "Got flag --no-crash-dump.  Ignoring value of flag --crash-dump and "
            "setting --crash-dump=never"
        )
        crash_dump = "never"

    accepted_crash_dump = ["always", "legacy", "on-failure", "never"]
    if crash_dump not in accepted_crash_dump:
        raise ValueError(
            f"Got invalid --crash-dump={crash_dump}, must be one of"
            f" {accepted_crash_dump}"
        )

    return crash_dump


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


BundleOpt = Union[str, Path, "OpsTest.Bundle"]
Timeout = TypeVar("Timeout", float, int)


def _connect_kwds(request) -> Dict[str, Any]:
    """Create a dict of keyword arguments for connecting to a model."""
    kwds = {}
    if val := request.config.option.juju_max_frame_size:
        if 0 < val:
            kwds["max_frame_size"] = val
        else:
            raise ValueError(f"max-frame-size must be positive integer, not {val}")
    return kwds


@dataclasses.dataclass
class ModelState:
    model: Model
    keep: bool
    destroy_storage: bool
    controller_name: str
    cloud_name: Optional[str]
    model_name: str
    config: Optional[dict] = None
    tmp_path: Optional[Path] = None
    timeout: Optional[Timeout] = None

    @property
    def full_name(self) -> str:
        return f"{self.controller_name}:{self.model_name}"


@dataclasses.dataclass
class CloudState:
    cloud_name: str
    models: List[str] = dataclasses.field(default_factory=list)
    timeout: Optional[Timeout] = None


class OpsTest:
    """Utility class for testing Operator Charms."""

    class ModelKeep(enum.Enum):
        """
        Used to select the appropriate behavior for cleaning up models
        created or used by ops_test.
        """

        NEVER = "never"
        """
        This gives pytest-operator the duty to delete this model
        at the end of the test regardless of any outcome.
        """

        ALWAYS = "always"
        """
        This gives pytest-operator the duty to keep this model
        at the end of the test regardless of any outcome.
        """

        IF_EXISTS = "if-exists"
        """
        If the model already exists before ops_test encounters it,
        follow the rules defined by `track_model.use_existing`
           * respects the --keep-models flag, otherwise
           * newly created models mapped to ModelKeep.NEVER
           * existing models mapped to ModelKeep.ALWAYS
        """

    # store instance, so we can tell if it's been used yet
    _instance: Optional["OpsTest"] = None

    # objects can be created with `ops_test.Bundle(...)`
    # since fixtures are autoloaded for pytest users,
    # this exposes the class instantiation through
    #     ops_test.Bundle(...)
    @dataclasses.dataclass
    class Bundle:
        """Represents a charmhub bundle."""

        name: str
        channel: str = "stable"
        arch: str = "all"
        series: str = "all"

        @property
        def juju_download_args(self):
            """Create cli arguments used in juju download to download bundle to disk."""
            return [
                f"--{field.name}={getattr(self, field.name)}"
                for field in dataclasses.fields(OpsTest.Bundle)
                if field.default is not dataclasses.MISSING
            ]

    def __init__(self, request, tmp_path_factory):
        self.request = request
        self._tmp_path_factory = tmp_path_factory
        self._global_tmp_path = None

        # Flag indicating whether all subsequent tests should be aborted.
        self.aborted = False

        # Flag for using destructive mode or not for charm builds.
        self.destructive_mode = request.config.option.destructive_mode

        # Config options to determine first model specs used by tests
        self._orig_model_alias: Optional[str] = request.config.option.model_alias
        self._init_cloud_name: Optional[str] = request.config.option.cloud
        self._init_model_name: Optional[str] = request.config.option.model
        self._init_keep_model: bool = request.config.option.keep_models
        self._init_destroy_storage: bool = request.config.option.destroy_storage
        self._juju_connect_kwds: Dict[str, Any] = _connect_kwds(request)
        self._charmcraft_args: List[str] = request.config.option.charmcraft_args

        # These may be modified by _setup_model
        self.controller_name = request.config.option.controller
        self._init_model_config = request.config.option.model_config

        # Flag for enabling the juju-crashdump
        self.crash_dump = validate_crash_dump(
            crash_dump=request.config.option.crash_dump,
            no_crash_dump=request.config.option.no_crash_dump,
        )
        self.crash_dump_output = request.config.option.crash_dump_output
        self.crash_dump_args = request.config.option.crash_dump_args

        # These will be set by _setup_model
        self.jujudata = None
        self._controller: Optional[Controller] = None

        # maintains a set of all models connected by this fixture
        # use an OrderedDict so that the first model made is destroyed last.
        self._current_alias = None
        self._models: MutableMapping[str, ModelState] = OrderedDict()
        self._clouds: MutableMapping[str, CloudState] = OrderedDict()

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
        current_state = self.current_alias and self._models.get(self.current_alias)
        return current_state.keep if current_state else self._init_keep_model

    @property
    def destroy_storage(self) -> bool:
        """
        Represents whether the current model storage should be destroyed after tests.
        """
        current_state = self.current_alias and self._models.get(self.current_alias)
        return (
            current_state.destroy_storage
            if current_state
            else self._init_destroy_storage
        )

    def _generate_name(self, kind: str) -> str:
        module_name = self.request.module.__name__.rpartition(".")[-1]
        suffix = "".join(choices(ascii_lowercase + digits, k=4))
        if kind != "model":
            suffix = "-".join((kind, suffix))
        return f"{module_name.replace('_', '-')}-{suffix}"

    @cached_property
    def default_model_name(self) -> str:
        return self._generate_name(kind="model")

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
        self, cloud_name, model_name, keep=False, destroy_storage=False, **kwargs
    ):
        """
        Creates a model used by the test framework which would normally be destroyed
        after the tests are run in the module.
        """
        controller = self._controller
        controller_name = controller.controller_name
        credential_name = None
        timeout = None
        if not cloud_name:
            # if not provided, try the default cloud name
            cloud_name = self._init_cloud_name
        if not cloud_name:
            # if not provided, use the controller's default cloud
            cloud_name = await controller.get_cloud()
        if ops_cloud := self._clouds.get(cloud_name):
            credential_name = cloud_name
            timeout = ops_cloud.timeout

        model_full_name = f"{controller_name}:{model_name}"
        log.info(f"Adding model {model_full_name} on cloud {cloud_name}")

        model = await controller.add_model(
            model_name, cloud_name, credential_name=credential_name, **kwargs
        )
        # NB: This call to `juju models` is needed because libjuju's
        # `add_model` doesn't update the models.yaml cache that the Juju
        # CLI depends on with the model's UUID, which the CLI requires to
        # connect. Calling `juju models` beforehand forces the CLI to
        # update the cache from the controller.
        await self.juju("models")
        state = ModelState(
            model,
            keep,
            destroy_storage,
            controller_name,
            cloud_name,
            model_name,
            timeout=timeout,
        )
        state.config = await model.get_config()
        return state

    async def _model_exists(self, model_name: str) -> bool:
        """
        returns True when the model_name exists in the model.
        """
        all_models = await self._controller.list_models()
        return model_name in all_models

    @staticmethod
    async def _connect_to_model(
        controller_name, model_name, keep=True, destroy_storage=False, **connect_kwargs
    ):
        """
        Makes a reference to an existing model used by the test framework
        which will not be destroyed after the tests are run in the module.
        """
        model = Model()
        state = ModelState(
            model, keep, destroy_storage, controller_name, None, model_name
        )
        log.info(
            "Connecting to existing model %s on unspecified cloud", state.full_name
        )
        await model.connect(state.full_name, **connect_kwargs)
        state.config = await model.get_config()
        return state

    @staticmethod
    def read_model_config(
        config_path_or_obj: Union[dict, str, os.PathLike, None],
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
        assert self.controller_name, "No controller selected for ops_test"
        if not self._controller:
            self._controller = Controller()
            await self._controller.connect(
                self.controller_name, **self._juju_connect_kwds
            )

        await self.track_model(
            alias,
            model_name=self._init_model_name or self.default_model_name,
            cloud_name=self._init_cloud_name,
            keep=self._init_model_name is not None,
            destroy_storage=self._init_destroy_storage,
            config=self.read_model_config(self._init_model_config),
        )

        self._current_alias = alias

    async def track_model(
        self,
        alias: str,
        model_name: Optional[str] = None,
        cloud_name: Optional[str] = None,
        use_existing: Optional[bool] = None,
        destroy_storage: Optional[bool] = None,
        keep: Union[ModelKeep, str, bool, None] = ModelKeep.IF_EXISTS,
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
        @param Optional[bool] destroy_storage: wether the storage should be destroyed
                                               with the model, None defaults to the
                                               pytest config flag `--destroy-storage`
        @param Optional[bool] use_existing:
               None:  True if model_name exists on this controller
               False: create a new model and keep=False, unless keep=True explicitly set
               True:  connect to a model and keep=True, unless keep=False explicitly set
        @param Optional[ModelKeep, str, bool, None] keep:
               ModelKeep  : See docs for the enum
               str        : mapped to ModelKeep values
               None       : Same as ModelKeep.IF_EXISTS
               True       : Same as ModelKeep.ALWAYS
               False      : Same as ModelKeep.NEVER, but respects keep-models flag

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

        keep_val: bool = False
        if isinstance(keep, str):
            keep = OpsTest.ModelKeep(keep.lower())
        if isinstance(keep, OpsTest.ModelKeep):
            if keep is OpsTest.ModelKeep.IF_EXISTS:
                keep_val = self._init_keep_model or bool(use_existing)
            elif keep is OpsTest.ModelKeep.ALWAYS:
                keep_val = True
            elif keep is OpsTest.ModelKeep.NEVER:
                keep_val = False
        elif isinstance(keep, bool):
            keep_val = self._init_keep_model or keep
        elif keep is None:
            keep_val = self._init_keep_model or bool(use_existing)

        destroy_storage_val = (
            self._init_destroy_storage if destroy_storage is None else destroy_storage
        )

        if use_existing:
            if not model_name:
                raise NotImplementedError(
                    "Cannot use_existing model if model_name is unspecified"
                )
            model_state = await self._connect_to_model(
                self.controller_name,
                model_name,
                keep_val,
                **self._juju_connect_kwds,
            )
        else:
            cloud_name = cloud_name or self.cloud_name
            model_name = model_name or self._generate_name(kind="model")
            model_state = await self._add_model(
                cloud_name, model_name, keep_val, destroy_storage_val, **kwargs
            )
        self._models[alias] = model_state
        if ops_cloud := self._clouds.get(cloud_name):
            ops_cloud.models.append(alias)
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
        args = ["-s", f"-m={self.model_full_name}", "-a=debug-layer", "-a=config"]
        output_directory = self.crash_dump_output
        if output_directory:
            log.debug("juju-crashdump will use output dir `%s`", output_directory)
            args.append(f"-o={output_directory}")

        user_args = shlex.split(self.crash_dump_args)
        cmd = ["juju-crashdump"] + args + user_args
        try:
            return_code, _, __ = await self.run(*cmd)
            log.info("juju-crashdump finished [%s]", return_code)
            return True
        except FileNotFoundError:
            log.info("juju-crashdump command was not found.")
            return False

    async def forget_model(
        self,
        alias: Optional[str] = None,
        timeout: Optional[Timeout] = None,
        destroy_storage: Optional[bool] = None,
        allow_failure: bool = True,
    ):
        """
        Forget a model and wait for it to be removed from the controller.
        If the model is marked as kept, ops_tests forgets about this model immediately.
        If the model is not marked as kept, ops_test will destroy the model.
        If timeout is None don't wait on the model to be completely destroyed

        @param                   str alias: alias of the model (default: current alias)
        @param Optional[float,int] timeout: how long to wait for it to be removed,
                                            if None, don't block waiting for success
        @param          bool allow_failure: if False, failures raise an exception
        @param        bool destroy_storage: destroy storage when removing model
        """
        if not self._controller:
            log.error("No access to controller, skipping...")
            return

        if not alias:
            alias = self.current_alias

        if alias not in self.models:
            raise ModelNotFoundError(f"{alias} not found")

        model_state: ModelState = self._models[alias]
        if timeout is None and model_state.timeout:
            timeout = model_state.timeout

        async def is_model_alive():
            return model_name in await self._controller.list_models()

        with self.model_context(alias) as model:
            await self.log_model()
            model_name = model.info.name

            if self.is_crash_dump_enabled():
                await self.create_crash_dump()

            if not self.keep_model:
                await self._reset(model, allow_failure, timeout=timeout)
                destroy_storage = (
                    self.destroy_storage if destroy_storage is None else destroy_storage
                )
                await self._controller.destroy_model(
                    model_name,
                    force=True,
                    destroy_storage=destroy_storage,
                    max_wait=timeout,
                )
                if timeout and await is_model_alive():
                    log.warning("Waiting for model %s to die...", model_name)
                    while await is_model_alive():
                        await asyncio.sleep(5)

            await model.disconnect()

        # stop managing this model now
        log.info(f"Forgetting model {alias}...")
        self._models.pop(alias)
        if ops_cloud := self._clouds.get(model_state.cloud_name):
            ops_cloud.models.remove(alias)
        if alias is self.current_alias:
            self._current_alias = None

    @staticmethod
    async def _reset(model: Model, allow_failure, timeout: Optional[Timeout] = None):
        # Forcibly destroy applications/machines in case any units are in error.
        async def _destroy(entity_name: str, **kwargs):
            for key, entity in getattr(model, entity_name).items():
                try:
                    log.info(f"   Destroying {entity_name} {key}")
                    await entity.destroy(**kwargs)
                except DeadEntityException as e:
                    log.warning(e)
                    log.warning(f"{entity_name.title()} already dead, skipping")
                except JujuError as e:
                    log.exception(e)
                    if not allow_failure:
                        raise
            return None

        log.info(f"Resetting model {model.info.name}...")
        await _destroy("applications")
        await _destroy("machines", force=True)

        if timeout is None:
            log.info("Not waiting on reset to complete.")
            return

        try:
            await model.block_until(
                lambda: len(model.units) == 0
                and len(model.machines) == 0
                and len(model.applications) == 0,
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
        # remove clouds from most recently made, to first made
        # each model in the cloud will be forgotten
        for cloud in reversed(self._clouds):
            await self.forget_cloud(cloud)

        # remove models from most recently made, to first made
        aliases = list(reversed(self._models.keys()))
        for model in aliases:
            await self.forget_model(model)

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

    @overload
    async def build_charm(
        self,
        charm_path,
        bases_index: Optional[int] = None,
        verbosity: Optional[
            Literal["quiet", "brief", "verbose", "debug", "trace"]
        ] = None,
        return_all: Literal[False] = False,  # default case first
    ) -> Path: ...

    @overload
    async def build_charm(
        self,
        charm_path,
        bases_index: Optional[int] = None,
        verbosity: Optional[
            Literal["quiet", "brief", "verbose", "debug", "trace"]
        ] = None,
        return_all: Literal[True] = True,
    ) -> List[Path]: ...

    async def build_charm(
        self,
        charm_path,
        bases_index: int = None,
        verbosity: Optional[
            Literal["quiet", "brief", "verbose", "debug", "trace"]
        ] = None,
        return_all: bool = False,
    ) -> Union[Path, List[Path]]:
        """Builds a single charm.

        This can handle charms using the older charms.reactive framework as
        well as charms written against the modern operator framework.

        Args:
            charm_path:  Path to the base source of the charm.
            bases_index: Index of `bases` configuration to build
                         (see charmcraft pack help)
            verbosity:   Verbosity level for charmcraft pack.
            return_all:  Return all built charms, not just the first one.

        Returns:
            Returns a Path / Paths for the built charm file.

        Raises:
            RuntimeError: If the charm build fails.
            FileNotFoundError: If no charm file is found after a successful build
        """
        charms_dst_dir = self.tmp_path / "charms"
        charms_dst_dir.mkdir(exist_ok=True)
        charm_path = Path(charm_path)
        charm_abs = Path(charm_path).absolute()
        metadata_path = charm_path / "metadata.yaml"
        layer_path = charm_path / "layer.yaml"
        charmcraft_path = charm_path / "charmcraft.yaml"
        charmcraft_yaml_exists = charmcraft_path.exists()
        charm_name = None
        if charmcraft_yaml_exists:
            charmcraft_yaml = yaml.safe_load(charmcraft_path.read_text())
            if "name" in charmcraft_yaml:
                charm_name = charmcraft_yaml["name"]
        if charm_name is None:
            charm_name = yaml.safe_load(metadata_path.read_text())["name"]
        if layer_path.exists() and not charmcraft_yaml_exists:
            # Handle older, reactive framework charms.
            # if a charmcraft.yaml file isn't defined for it
            check_deps("charm")
            cmd = ["charm", "build", "--charm-file"]
        else:
            # Handle newer, operator framework charms.
            users_groups = {grp.getgrgid(g).gr_name for g in os.getgroups()}
            cmd = ["charmcraft", "pack"]
            if bases_index is not None:
                cmd.append(f"--bases-index={bases_index}")
            if verbosity:
                cmd.append(f"--verbosity={verbosity}")
            for args in self._charmcraft_args:
                cmd.append(args)
            if self.destructive_mode:
                # host builder never requires lxd group
                cmd.append("--destructive-mode")
            elif "lxd" not in users_groups:
                # building with lxd builder and user does't already have lxd group;
                # try to build with sudo -u <user> -E charmcraft pack

                all_groups = {g.gr_name for g in grp.getgrall()}
                assert "lxd" in all_groups, (
                    "Group 'lxd' required but not available; "
                    "ensure that lxd is available or use --destructive-mode"
                )
                cmd = ["sudo", "-g", "lxd", "-E", *cmd]

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
                f"Failed to build charm at path `{charm_path}`:\n"
                f"    command used: `{' '.join(cmd)}`\n"
                f"    stdout: {stdout or '(null)'}\n"
                f"    stderr: {stderr or '(null)'}\n"
            )

        # If charmcraft.yaml has multiple bases
        # then multiple charms would be generated, rename them all
        charms = list(charm_abs.glob(f"{charm_name}*.charm"))
        for idx, charm_file_src in enumerate(charms):
            charm_file_dst = charms_dst_dir / charm_file_src.name
            charms[idx] = charm_file_src.rename(charm_file_dst)

        if not charms:
            raise FileNotFoundError(f"No such file in '{charm_path}/*.charm'")
        if charms and not return_all:
            # Even though we may have multiple *.charm file,
            # for backwards compatibility we can - only return one.
            return charms[0]
        return charms

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

    def arch_specific_resources(self, built_charm):
        return {
            name: rsc
            for name, rsc in self.charm_file_resources(built_charm).items()
            if rsc.arch
        }

    async def build_resources(self, build_script: Path, with_sudo: bool = True):
        build_script = build_script.absolute()
        if not build_script.exists():
            raise FileNotFoundError(
                f"Failed to locate resource build script {build_script}"
            )

        log.info("Build Resources...")
        dst_dir = self.tmp_path / "resources"
        dst_dir.mkdir(exist_ok=True)
        start, cmd = timer(), ("sudo " if with_sudo else "") + str(build_script)
        rc, stdout, stderr = await self.run(*shlex.split(cmd), cwd=dst_dir, check=False)
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

    async def async_render_bundles(self, *bundles: BundleOpt, **context) -> List[Path]:
        """
        Render a set of templated bundles using Jinja2.

        This can be used to populate built charm paths or config values.
        @param *bundles: objects that are YAML content, pathlike, or charmhub reference
        @param **context: Additional optional context as keyword args.
        @returns list of paths to rendered bundles.
        """
        ...
        bundles_dst_dir = self.tmp_path / "bundles"
        bundles_dst_dir.mkdir(exist_ok=True)
        re_bundlefile = re.compile(r"\.(yaml|yml)(\.j2)?$")
        to_render = []
        for bundle in bundles:
            if isinstance(bundle, str) and re_bundlefile.search(bundle):
                content = Path(bundle).read_text()
            elif isinstance(bundle, str):
                content = bundle
            elif isinstance(bundle, Path):
                content = bundle.read_text()
            elif isinstance(bundle, OpsTest.Bundle):
                filepath = f"{bundles_dst_dir}/{bundle.name}.bundle"
                await self.juju(
                    "download",
                    bundle.name,
                    *bundle.juju_download_args,
                    f"--filepath={filepath}",
                    check=True,
                    fail_msg=f"Couldn't download {bundle.name} bundle",
                )
                bundle_zip = ZipPath(filepath, "bundle.yaml")
                content = bundle_zip.read_text()
            else:
                raise TypeError(f"bundle {type(bundle)} isn't a known Type")
            to_render.append(content)
        return self.render_bundles(*to_render, **context)

    def render_bundles(self, *bundles, context=None, **kwcontext) -> List[Path]:
        """Render one or more templated bundles using Jinja2.

        This can be used to populate built charm paths or config values.

        :param *bundles (str or Path): One or more bundle Paths or YAML contents.
        :param context (dict): Optional context mapping.
        :param **kwcontext: Additional optional context as keyword args.

        Returns a list of Paths for the rendered bundles.
        """
        # Jinja2 does support async, but rendering bundles should be relatively quick.
        return [
            self.render_bundle(bundle, context=context, **kwcontext)
            for bundle in bundles
        ]

    def render_bundle(self, bundle, context=None, **kwcontext) -> Path:
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

    @contextlib.asynccontextmanager
    async def fast_forward(
        self, fast_interval: str = "10s", slow_interval: Optional[str] = None
    ):
        """Temporarily speed up update-status firing rate for the current model.

        Returns an async context manager that temporarily sets update-status
        firing rate to `fast_interval`.
        If provided, when the context exits the update-status firing rate will
        be set to `slow_interval`. Otherwise, it will be set to the previous
        value.
        """
        model = self.model
        if not model:
            raise RuntimeError("No model currently set.")

        update_interval_key = "update-status-hook-interval"
        if slow_interval:
            interval_after = slow_interval
        else:
            interval_after = (await model.get_config())[update_interval_key]

        try:
            await model.set_config({update_interval_key: fast_interval})
            yield
        finally:
            # Whatever happens, we restore the interval.
            await model.set_config({update_interval_key: interval_after})

    def is_crash_dump_enabled(self) -> bool:
        """Returns whether Juju crash dump is enabled given the current settings."""
        if self.crash_dump == "always":
            return True
        elif self.crash_dump == "on-failure" and self.request.session.testsfailed > 0:
            return True
        elif (
            self.crash_dump == "legacy"
            and self.request.session.testsfailed > 0
            and self.keep_model is False
        ):
            return True
        else:
            return False

    async def add_k8s(
        self,
        cloud_name: Optional[str] = None,
        kubeconfig: Optional[K8sConfiguration] = None,
        context: Optional[str] = None,
        skip_storage: bool = True,
        storage_class: Optional[str] = None,
    ) -> str:
        """
        Add a new k8s cloud in the existing controller.

        @param Optional[str] cloud_name:
            Name for the new cloud
            None will autogenerate a name
        @param Optional[kubernetes.client.configuration.Configuration] kubeconfig:
            Configuration object from kubernetes.config.load_config
            None will read from the usual kubeconfig locations like
                os.environ.get('KUBECONFIG', '$HOME/.kube/config')
        @param Optional[str] context:
            context to use within the kubeconfig
            None will use the default context
        @param bool skip_storage:
            True will not use cloud storage,
            False either finds storage or uses storage_class
        @param Optional[str] storage_class:
            cluster storage-class to use for juju storage
            None will look for a default storage class within the cluster

        @returns str: cloud_name

        Common Examples:
        ----------------------------------
        # make a new k8s cloud with any juju name and destroy it when the tests are over
        await ops_test.add_k8s()

        # make a cloud known to juju as "bob"
        await ops_test.add_k8s(cloud_name="my-k8s")
        ----------------------------------
        """

        if kubeconfig is None:
            # kubeconfig should be auto-detected from the usual places
            kubeconfig = type.__call__(K8sConfiguration)
            kubernetes.config.load_config(
                client_configuration=kubeconfig,
                context=context,
                temp_file_path=self.tmp_path,
            )
        juju_cloud_config = {}
        if not skip_storage and storage_class is None:
            # lookup default storage-class
            api_client = kubernetes.client.ApiClient(configuration=kubeconfig)
            cluster = k8s_client.StorageV1Api(api_client=api_client)
            for sc in cluster.list_storage_class().items:
                if (
                    sc.metadata.annotations.get(
                        "storageclass.kubernetes.io/is-default-class"
                    )
                    == "true"
                ):
                    storage_class = sc.metadata.name
        if not skip_storage and storage_class:
            juju_cloud_config["workload-storage"] = storage_class
            juju_cloud_config["operator-storage"] = storage_class

        controller = self._controller
        cloud_name = cloud_name or self._generate_name("k8s-cloud")
        log.info(f"Adding k8s cloud {cloud_name}")

        cloud_def = client.Cloud(
            auth_types=[
                "certificate",
                "clientcertificate",
                "oauth2",
                "oauth2withcert",
                "userpass",
            ],
            ca_certificates=[Path(kubeconfig.ssl_ca_cert).read_text()],
            endpoint=kubeconfig.host,
            host_cloud_region="kubernetes/ops-test",
            regions=[client.CloudRegion(endpoint=kubeconfig.host, name="default")],
            skip_tls_verify=not kubeconfig.verify_ssl,
            type_="kubernetes",
            config=juju_cloud_config,
        )

        if kubeconfig.cert_file and kubeconfig.key_file:
            auth_type = "clientcertificate"
            attrs = dict(
                ClientCertificateData=Path(kubeconfig.cert_file).read_text(),
                ClientKeyData=Path(kubeconfig.key_file).read_text(),
            )
        elif token := kubeconfig.api_key["authorization"]:
            if token.startswith("Bearer "):
                auth_type = "oauth2"
                attrs = {"Token": token.split(" ")[1]}
            elif token.startswith("Basic "):
                auth_type, userpass = "userpass", token.split(" ")[1]
                user, passwd = base64.b64decode(userpass).decode().split(":", 1)
                attrs = {"username": user, "password": passwd}
            else:
                raise ValueError("Failed to find credentials in authorization token")
        else:
            raise ValueError("Failed to find credentials in kubernetes.Configuration")

        await controller.add_cloud(cloud_name, cloud_def)
        await controller.add_credential(
            cloud_name,
            credential=client.CloudCredential(attrs, auth_type),
            cloud=cloud_name,
        )
        self._clouds[cloud_name] = CloudState(cloud_name, timeout=5 * 60)
        return cloud_name

    async def forget_cloud(self, cloud_name: str):
        if cloud_name not in self._clouds:
            raise KeyError(f"{cloud_name} not in clouds")
        for model in reversed(self._clouds[cloud_name].models):
            await self.forget_model(model, destroy_storage=True)
        log.info(f"Forgetting cloud: {cloud_name}...")
        await self._controller.remove_cloud(cloud_name)
        del self._clouds[cloud_name]
