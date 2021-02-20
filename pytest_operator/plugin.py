import asyncio
import inspect
import os
import re
import shutil
import subprocess
import sys
import textwrap
import logging
from fnmatch import fnmatch
from functools import wraps
from pathlib import Path
from random import choices
from string import hexdigits, ascii_lowercase, digits

import jinja2
import pytest
import yaml

from juju.client.jujudata import FileJujuData
from juju.controller import Controller
from juju.model import Model

from unittest import TestCase


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


def pytest_configure(config):
    config.addinivalue_line("markers", "abort_on_fail")


@pytest.fixture(scope="session")
def check_deps(autouse=True):
    missing = []
    for dep in ("juju", "charm-build", "charmcraft"):
        res = subprocess.run(["which", dep])
        if res.returncode != 0:
            missing.append(dep)
    if missing:
        raise RuntimeError(
            "Missing dependenc{}: {}".format(
                "y" if len(missing) == 1 else "ies",
                ", ".join(missing),
            )
        )


def _cls_to_model_name(cls):
    def _decamelify(match):
        prefix = f"{match.group(1)}-" if match.group(1) else ""
        if match.group(3) and len(match.group(2)) > 1:
            return (
                f"{prefix}{match.group(2)[:-1].lower()}-"
                f"{match.group(2)[-1:].lower()}{match.group(3)}"
            )
        elif match.group(3):
            return f"{prefix}{match.group(2).lower()}{match.group(3)}"
        else:
            return f"{prefix}{match.group(2).lower()}"

    camel_pat = re.compile(r"([a-z]?)([A-Z]+)([a-z]?)")
    suffix = "".join(choices(ascii_lowercase + digits, k=4))
    full_name = f"{cls.__qualname__}-{suffix}"
    return re.sub(r"[^a-z0-9-]", "-", re.sub(camel_pat, _decamelify, full_name))


def _wrap_async_tests(cls):
    def _wrap_async(async_method):
        @wraps(async_method)
        def _run_async(*args, **kwargs):
            if cls.aborted:
                pytest.xfail("aborted")
            item = cls._item_for_method(async_method)
            is_abort_on_fail = item.get_closest_marker("abort_on_fail")
            try:
                return cls.loop.run_until_complete(async_method(*args, **kwargs))
            except Exception:
                if is_abort_on_fail:
                    cls.aborted = True
                raise

        return _run_async

    for name, method in inspect.getmembers(cls, inspect.iscoroutinefunction):
        if not name.startswith("test_"):
            continue
        setattr(cls, name, _wrap_async(method))


@pytest.fixture(scope="class")
def inject_fixtures(request, tmp_path_factory):
    cls = request.cls
    cls.request = request
    cls.tmp_path = tmp_path_factory.mktemp(_cls_to_model_name(cls))
    log.info(f"Using tmp_path: {cls.tmp_path}")
    cls.loop = asyncio.new_event_loop()
    asyncio.set_event_loop(cls.loop)
    cls.loop.run_until_complete(cls.setup_model())

    _wrap_async_tests(cls)

    yield

    cls.loop.run_until_complete(cls.cleanup_model())
    cls.loop.close()


@pytest.mark.usefixtures("inject_fixtures")
class OperatorTest(TestCase):
    """Base class for testing Operator Charms."""

    # Flag indicating whether all subsequent tests should be aborted.
    aborted = False

    # This will be injected by inject_fixtures.
    request = None
    tmp_path = None
    loop = None

    # These will be set by setup_model
    cloud_name = None
    controller_name = None
    model_name = None
    model_full_name = None
    model = None
    jujudata = None

    @classmethod
    def _item_for_method(cls, method):
        for item in cls.request.session.items:
            function = getattr(item, "function", None)
            if function is method:
                return item
            if getattr(function, "__wrapped__", None) is method:
                return item
        else:
            raise ValueError(f"Item not found for {method}")

    @classmethod
    async def _run(cls, *cmd, cwd=None):
        proc = await asyncio.create_subprocess_exec(
            *(str(c) for c in cmd),
            cwd=str(cwd or "."),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=dict(os.environ, JUJU_DATA=cls.jujudata.path),
        )
        stdout, stderr = await proc.communicate()
        stdout, stderr = stdout.decode("utf8"), stderr.decode("utf8")
        return proc.returncode, stdout, stderr

    @classmethod
    async def setup_model(cls):
        cls.cloud_name = cls.request.config.getoption("--cloud")
        cls.controller_name = cls.request.config.getoption("--controller")
        cls.model_name = cls.request.config.getoption("--model")
        cls.keep_model = cls.request.config.getoption("--keep-models")
        # TODO: We won't need this if Model.debug_log is implemented in libjuju
        cls.jujudata = FileJujuData()
        if not cls.controller_name:
            cls.controller_name = cls.jujudata.current_controller()
        if not cls.model_name:
            cls.model_name = _cls_to_model_name(cls)
            cls.model_full_name = f"{cls.controller_name}:{cls.model_name}"
            controller = Controller()
            await controller.connect(cls.controller_name)
            on_cloud = f" on cloud {cls.cloud_name}" if cls.cloud_name else ""
            log.info(f"Adding model {cls.model_full_name}{on_cloud}")
            cls.model = await controller.add_model(
                cls.model_name, cloud_name=cls.cloud_name
            )
            await controller.disconnect()
        else:
            cls.model_full_name = f"{cls.controller_name}:{cls.model_name}"
            log.info(f"Connecting to model {cls.model_full_name}")
            cls.model = Model()
            await cls.model.connect(cls.model_full_name)
            cls.keep_model = True  # don't cleanup models we didn't create

    @classmethod
    async def dump_model(cls):
        if not (cls.model.units or cls.model.machines):
            log.info("Model is empty")
            return

        unit_len = max(len(unit.name) for unit in cls.model.units.values()) + 1
        unit_line = f"{{:{unit_len}}}  {{:7}}  {{:11}}  {{}}"
        machine_line = "{:<7}  {:10}  {}"

        status = [unit_line.format("Unit", "Machine", "Status", "Message")]
        for unit in cls.model.units.values():
            status.append(
                unit_line.format(
                    unit.name + ("*" if await unit.is_leader_from_status() else ""),
                    unit.machine.id,
                    unit.workload_status,
                    unit.workload_status_message,
                )
            )
        status.append("")
        status.append(machine_line.format("Machine", "Series", "Status"))
        for machine in cls.model.machines.values():
            status.append(
                machine_line.format(machine.id, machine.series, machine.status)
            )
        status = "\n".join(status)
        log.info(f"Model status:\n\n{status}")

        # TODO: Implement Model.debug_log in libjuju
        # NB: This call to `juju models` is needed because libjuju's `add_model`
        # doesn't update the models.yaml cache that `juju debug-logs` depends
        # on. Calling `juju models` beforehand forces the CLI to update the
        # cache from the controller.
        await cls._run("juju", "models")
        returncode, stdout, stderr = await cls._run(
            "juju",
            "debug-log",
            "-m",
            cls.model_full_name,
            "--replay",
            "--no-tail",
            "--level",
            "ERROR",
        )
        if returncode != 0:
            raise RuntimeError(f"Failed to get error logs:\n{stderr}\n{stdout}")
        log.info(f"Juju error logs:\n\n{stdout}")

    @classmethod
    async def cleanup_model(cls):
        if not cls.model:
            return

        await cls.dump_model()

        if not cls.keep_model:
            controller = await cls.model.get_controller()
            # Forcibly destroy machines in case any units are in error.
            for machine in cls.model.machines.values():
                log.info(f"Destroying machine {machine.id}")
                await machine.destroy(force=True)
            await cls.model.disconnect()
            log.info(f"Destroying model {cls.model_name}")
            await controller.destroy_model(cls.model_name)
            await controller.disconnect()
        else:
            await cls.model.disconnect()

    def abort(self, *args, **kwargs):
        """Fail the current test method and mark all remaining test methods as xfail.

        This can be used if a given step is required for subsequent steps to be
        successful, such as the initial deployment.

        Any args will be passed through to `pytest.fail()`.

        You can also mark a test with `@pytest.marks.abort_on_fail` to have this
        automatically applied if the marked test method fails or errors.
        """
        type(self).aborted = True
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
            cmd = ["charm-build", "-F", charm_abs]
        else:
            # Handle newer, operator framework charms.
            cmd = ["charmcraft", "build", "-f", charm_abs]

        log.info(f"Building charm {charm_name}")
        returncode, stdout, stderr = await self._run(*cmd, cwd=charms_dst_dir)

        if not layer_path.exists():
            # Clean up build dir created by charmcraft.
            build_path = charm_path / "build"
            if build_path.exists():
                shutil.rmtree(build_path)

        if returncode != 0:
            raise RuntimeError(
                f"Failed to build charm {charm_path}:\n{stderr}\n{stdout}"
            )

        return charms_dst_dir / f"{charm_name}.charm"

    async def build_charms(self, *charm_paths):
        """Builds one or more charms in parallel.

        This can handle charms using the older charms.reactive framework as
        well as charms written against the modern operator framework.

        Returns a mapping of charm names to Paths for the built charm files.
        """
        charms = await asyncio.gather(
            *(self.build_charm(charm_path) for charm_path in charm_paths)
        )
        return {charm.stem: charm for charm in charms}

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

        returncode, stdout, stderr = await self._run(
            sys.executable, "setup.py", "--fullname", cwd=lib_path_abs
        )
        if returncode != 0:
            raise RuntimeError(
                f"Failed to get library name {lib_path}:\n{stderr}\n{stdout}"
            )
        lib_name_ver = stdout.strip()
        lib_dst_path = libs_dst_dir / f"{lib_name_ver}.tar.gz"

        log.info(f"Building library {lib_path}")
        returncode, stdout, stderr = await self._run(
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
        if context is None:
            context = {}
        context.update(kwcontext)
        charm_path = Path(charm_path)
        charm_dst_path = self.tmp_path / "charms" / charm_path.name
        log.info(f"Rendering charm {charm_path}")
        shutil.copytree(
            charm_path,
            charm_dst_path,
            ignore=shutil.ignore_patterns(".git", ".bzr", "__pycache__", "*.pyc"),
        )
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
