import asyncio
import inspect
import re
import shutil
import subprocess
import textwrap
import logging
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
            print(match.groups())
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
            cwd=str(self.tmp_path),
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

        return self.tmp_path / f"{charm_name}.charm"

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

        Returns the Path for the rendered bundle.
        """
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
        rendered = jinja2.Template(bundle_text).render(**context)
        dst = self.tmp_path / bundle_name
        dst.write_text(rendered)
        return dst

    def render_bundles(self, *bundles, context=None, **kwcontext):
        """Render one or more templated bundles using Jinja2 in parallel.

        This can be used to populate built charm paths or config values.

        :param bundles (list[str or Path]): List of bundle Paths or YAML contents.

        Returns a list of Paths for the rendered bundles.
        """
        # Jinja2 does support async, but rendering bundles should be relatively quick.
        return [
            self.render_bundle(bundle_path, context=context, **kwcontext)
            for bundle_path in bundles
        ]
