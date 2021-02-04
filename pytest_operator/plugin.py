import asyncio
import re
import shutil
import subprocess
import textwrap
from datetime import datetime, timedelta, timezone
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


def parse_ts(ts):
    """Parse a Juju provided timestamp, which must be UTC."""
    return datetime.strptime(ts, "%d %b %Y %H:%M:%SZ").replace(tzinfo=timezone.utc)


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

    # TODO: Up-port this to libjuju.
    async def wait_for_bundle(
        self,
        bundle_path,
        raise_on_error=True,
        timeout=10 * 60,
        idle_period=15,
        check_freq=0.5,
    ):
        """Wait for the applications and units in the given bundle to settle.

        The bundle is considered "settled" when all units are simultaneously "idle"
        for at least `idle_period` seconds.

        :param bundle_path (str or Path): Path to bundle to read.

        :param raise_on_error (bool): If True, then any unit going into "error" status
            immediately raises an ErroredUnitError (which is an AssertionError).

        :param timeout (float): How long to wait, in seconds, for the bundle settles
            before raising an asyncio.TimeoutError. If None, will wait forever.

        :param idle_period (float): How long, in seconds, between agent status updates a
            unit needs to be idle for, to allow for queued hooks to start.

        :param check_freq (float): How frequently, in seconds, to check the model.
        """
        timeout = timedelta(timeout) if timeout is not None else None
        idle_period = timedelta(idle_period)
        bundle = yaml.safe_load(Path(bundle_path).read_text())
        start_time = datetime.now()
        apps = list(bundle["applications"].keys())
        status_times = {}
        while True:
            all_ready = True
            errored_units = []
            for app in apps:
                if app not in self.model.applications:
                    continue
                for unit in self.model.applications[app].units:
                    if raise_on_error and unit.workload_status == "error":
                        errored_units.append(unit.name)
                    if unit.name in status_times:
                        prev_status_time = status_times[unit.name]
                        curr_status_time = parse_ts(unit.agent_status_since)
                        if curr_status_time - prev_status_time < idle_period:
                            all_ready = False
                    status_times[unit.name] = parse_ts(unit.agent_status_since)
                expected_num_units = bundle["applications"][app]["num_units"]
                actual_num_units = len(self.model.applications[app].units)
                if actual_num_units < expected_num_units:
                    all_ready = False
            if errored_units:
                s = "s" if len(errored_units) > 1 else ""
                errored_units = ", ".join(errored_units)
                raise ErroredUnitError(f"Unit{s} in error: {errored_units}")
            if all_ready:
                break
            if timeout is not None and datetime.now() - start_time > timeout:
                raise asyncio.TimeoutError(f"Timed out waiting for {bundle_path}")
            await asyncio.sleep(check_freq)
