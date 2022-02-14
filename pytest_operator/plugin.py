import asyncio
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
from functools import cached_property
from fnmatch import fnmatch
from pathlib import Path
from random import choices
from string import ascii_lowercase, digits, hexdigits
from typing import Iterable, Optional
from urllib.request import urlretrieve, urlopen
from urllib.parse import urlencode
from urllib.error import HTTPError
from zipfile import Path as ZipPath

import jinja2
import pytest
import pytest_asyncio.plugin
import yaml
from juju.client.jujudata import FileJujuData
from juju.controller import Controller
from juju.exceptions import DeadEntityException
from juju.model import Model

log = logging.getLogger(__name__)


def pytest_addoption(parser):
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


def pytest_configure(config):
    config.addinivalue_line("markers", "abort_on_fail")
    # These need to be fixed in libjuju and just clutter things up for tests using this.
    config.addinivalue_line(
        "filterwarnings", "ignore:The loop argument:DeprecationWarning"
    )
    config.addinivalue_line(
        "filterwarnings", r"ignore:'with \(yield from lock\)':DeprecationWarning"
    )


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
    await ops_test._cleanup_model()


def handle_file_delete_error(function, path, execinfo):
    log.warn(f"Failed to delete '{path}' due to {execinfo[1]}")


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

    @cached_property
    def _charm_id(self):
        params = dict(channel=self._channel)
        url = f"{self.CS_URL}/{self._name}/meta/id-revision"
        try:
            resp = json_request(url, params)
        except HTTPError as ex:
            raise RuntimeError(
                f"Charm {self._name} not found in charmstore at channel={self._channel}"
            ) from ex
        revision = resp["Revision"]
        return f"charm-{revision}"

    @property
    def exists(self):
        try:
            return bool(self._charm_id)
        except RuntimeError:
            return False

    def download_resource(self, resource, destination: Path):
        charm_id = self._charm_id
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


class OpsTest:
    """Utility class for testing Operator Charms."""

    _instance = None  # store instance so we can tell if it's been used yet

    def __init__(self, request, tmp_path_factory):
        self.request = request
        self.tmp_path: Path = tmp_path_factory.mktemp(self.default_model_name)
        log.info(f"Using tmp_path: {self.tmp_path}")

        # Flag indicating whether all subsequent tests should be aborted.
        self.aborted = False

        # Flag for using destructive mode or not for charm builds.
        self.destructive_mode = request.config.option.destructive_mode

        # These may be modified by _setup_model
        self.cloud_name = request.config.option.cloud
        self.controller_name = request.config.option.controller
        self.model_name = request.config.option.model
        self.keep_model = request.config.option.keep_models

        # These will be set by _setup_model
        self.model_full_name = None
        self.model = None
        self.jujudata = None
        self._controller = None

    @property
    def default_model_name(self):
        if not hasattr(self, "_default_model_name"):
            module_name = self.request.module.__name__.rpartition(".")[-1]
            suffix = "".join(choices(ascii_lowercase + digits, k=4))
            self._default_model_name = f"{module_name.replace('_', '-')}-{suffix}"
        return self._default_model_name

    async def run(self, *cmd, cwd=None, check=False, fail_msg=None):
        """Asynchronously run a subprocess command.

        If `check` is False, returns a tuple of the return code, stdout, and
        stderr (decoded as utf8). Otherwise, calls `pytest.fail` with
        `fail_msg` and relevant command info.
        """
        env = {**os.environ}
        if self.jujudata:
            env["JUJU_DATA"] = self.jujudata.path
        if self.model_full_name:
            env["JUJU_MODEL"] = self.model_full_name

        proc = await asyncio.create_subprocess_exec(
            *(str(c) for c in cmd),
            cwd=str(cwd or "."),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout, stderr = await proc.communicate()
        stdout, stderr = stdout.decode("utf8"), stderr.decode("utf8")
        if check and proc.returncode != 0:
            if fail_msg is None:
                fail_msg = f"Command {list(cmd)} failed"
            raise AssertionError(
                f"{fail_msg} ({proc.returncode}): {(stderr or stdout).strip()}"
            )
        return proc.returncode, stdout, stderr

    _run = run  # backward compatibility alias

    async def juju(self, *args):
        """Runs a Juju CLI command.

        Useful for cases where python-libjuju sees things differently than the Juju CLI.
        Will set `JUJU_MODEL`, so manually passing in `-m model-name` is unnecessary.
        """

        return await self.run("juju", *args)

    async def _setup_model(self):
        # TODO: We won't need this if Model.debug_log is implemented in libjuju
        self.jujudata = FileJujuData()
        if not self.controller_name:
            self.controller_name = self.jujudata.current_controller()
        if not self.model_name:
            self.model_name = self.default_model_name
            self.model_full_name = f"{self.controller_name}:{self.model_name}"
            self._controller = Controller()
            await self._controller.connect(self.controller_name)
            on_cloud = f" on cloud {self.cloud_name}" if self.cloud_name else ""
            log.info(f"Adding model {self.model_full_name}{on_cloud}")
            self.model = await self._controller.add_model(
                self.model_name, cloud_name=self.cloud_name
            )
            # NB: This call to `juju models` is needed because libjuju's
            # `add_model` doesn't update the models.yaml cache that the Juju
            # CLI depends on with the model's UUID, which the CLI requires to
            # connect. Calling `juju models` beforehand forces the CLI to
            # update the cache from the controller.
            await self.juju("models")
        else:
            self.model_full_name = f"{self.controller_name}:{self.model_name}"
            log.info(f"Connecting to model {self.model_full_name}")
            self.model = Model()
            await self.model.connect(self.model_full_name)
            self.keep_model = True  # don't cleanup models we didn't create

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

    async def _cleanup_model(self):
        if not self.model:
            return

        await self.log_model()

        if not self.keep_model:
            # Forcibly destroy machines in case any units are in error.
            for machine in self.model.machines.values():
                try:
                    log.info(f"Destroying machine {machine.id}")
                    await machine.destroy(force=True)
                except DeadEntityException as e:
                    log.warning(e)
                    log.warning("Machine already dead, skipping")
            await self.model.disconnect()
            log.info(f"Destroying model {self.model_name}")
            await self._controller.destroy_model(self.model_name)
        else:
            await self.model.disconnect()
        if self._controller:
            await self._controller.disconnect()

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

    async def build_charm(self, charm_path):
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
        returncode, stdout, stderr = await self.run(*cmd, cwd=charm_abs)

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

    async def build_charms(self, *charm_paths):
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
        rc, stdout, stderr = await self.run(
            *shlex.split(f"sudo {build_script}"), cwd=dst_dir, check=False
        )
        if rc != 0:
            log.warning(f"{build_script} failed: {(stderr or stdout).strip()}")
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
        downloader = Charmhub(charm_name, channel)
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

        cmd += ["--", "-m", self.model_name] + list(extra_args)
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
