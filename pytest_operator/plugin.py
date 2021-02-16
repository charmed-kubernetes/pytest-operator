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
            return cls.loop.run_until_complete(async_method(*args, **kwargs))

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

    # These will be injected by inject_fixtures.
    request = None
    tmp_path = None
    loop = None
    model = None

    @classmethod
    async def setup_model(cls):
        cls.cloud_name = cls.request.config.getoption("--cloud")
        cls.controller_name = cls.request.config.getoption("--controller")
        cls.model_name = cls.request.config.getoption("--model")
        if not cls.model_name:
            cls.model_name = _cls_to_model_name(cls)
            controller = Controller()
            if controller:
                await controller.connect(cls.controller_name)
            else:
                await controller.connect_current()
            log.info(f"Adding model {cls.model_name} on cloud {cls.cloud_name}")
            cls.model = await controller.add_model(
                cls.model_name, cloud_name=cls.cloud_name
            )
            await controller.disconnect()
            cls.keep_model = cls.request.config.getoption("--keep-models")
        else:
            if cls.controller_name:
                cls.model_name = f"{cls.controller_name}:{cls.model_name}"
            cls.model = Model()
            await cls.model.connect(cls.model_name)
            cls.keep_model = True  # don't cleanup models we didn't create

    @classmethod
    async def cleanup_model(cls):
        if not cls.model:
            return
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
            cmd = ["charm-build", "-F", str(charm_abs)]
        else:
            # Handle newer, operator framework charms.
            cmd = ["charmcraft", "build", "-f", str(charm_abs)]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(charms_dst_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        log.info(f"Building charm {charm_name}")
        stdout, stderr = await proc.communicate()
        stdout, stderr = stdout.decode("utf8"), stderr.decode("utf8")

        if not layer_path.exists():
            # Clean up build dir created by charmcraft.
            build_path = charm_path / "build"
            if build_path.exists():
                shutil.rmtree(build_path)

        if proc.returncode != 0:
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

        proc = await asyncio.create_subprocess_exec(
            *(sys.executable, "setup.py", "--fullname"),
            cwd=str(lib_path_abs),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        stdout, stderr = stdout.decode("utf8"), stderr.decode("utf8")
        if proc.returncode != 0:
            raise RuntimeError(
                f"Failed to get library name {lib_path}:\n{stderr}\n{stdout}"
            )
        lib_name_ver = stdout.strip()
        lib_dst_path = libs_dst_dir / f"{lib_name_ver}.tar.gz"

        log.info(f"Building library {lib_path}")
        proc = await asyncio.create_subprocess_exec(
            *(sys.executable, "setup.py", "sdist", "-d", str(libs_dst_dir)),
            cwd=str(lib_path_abs),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        stdout, stderr = stdout.decode("utf8"), stderr.decode("utf8")
        if proc.returncode != 0:
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
