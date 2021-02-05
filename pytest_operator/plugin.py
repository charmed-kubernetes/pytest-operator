import asyncio
import re
import shutil
import subprocess
import textwrap
from pathlib import Path
from random import choices
from string import hexdigits

import jinja2
import pytest
import yaml

from juju.controller import Controller
from juju.model import Model

from .shims import IsolatedAsyncioTestCase


class ErroredUnitError(AssertionError):
    pass


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


class OperatorTest(IsolatedAsyncioTestCase):
    """Base class for testing Operator Charms."""

    @pytest.fixture(autouse=True)
    def handle_fixtures(self, request, tmp_path):
        self.request = request
        self.tmp_path = tmp_path

    def _modelify(self, name):
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
        return re.sub(r"[^a-z0-9-]", "-", re.sub(camel_pat, _decamelify, name))

    async def asyncSetUp(self):
        self.cloud_name = self.request.config.getoption("--cloud")
        self.controller_name = self.request.config.getoption("--controller")
        self.model_name = self.request.config.getoption("--model")
        if not self.model_name:
            self.model_name = self._modelify(self.id())
            controller = Controller()
            if controller:
                await controller.connect(self.controller_name)
            else:
                await controller.connect_current()
            self.model = await controller.add_model(
                self.model_name, cloud_name=self.cloud_name
            )
            await controller.disconnect()
            self.keep_model = self.request.config.getoption("--keep-models")
        else:
            if self.controller_name:
                self.model_name = f"{self.controller_name}:{self.model_name}"
            self.model = Model()
            await self.model.connect(self.model_name)
            self.keep_model = True  # don't cleanup models we didn't create

    async def asyncTearDown(self):
        if not self.keep_model:
            controller = await self.model.get_controller()
            await self.model.disconnect()
            await controller.destroy_model(self.model_name)
            await controller.disconnect()
        else:
            await self.model.disconnect()

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

        charm_name = yaml.safe_load(metadata_path.read_text())["name"]
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
