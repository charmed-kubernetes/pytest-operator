import os
import pytest
import shutil
import subprocess
from pathlib import Path

import asyncio
import yaml

from juju.controller import Controller
from juju.model import Model


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


class OperatorTools:
    """Utility class for accessing Juju related tools"""

    def __init__(self, request, tmp_path):
        self.tmp_path = tmp_path
        self.cloud = request.config.getoption("--cloud")
        self.controller = request.config.getoption("--controller")
        self.model = request.config.getoption("--model")
        self.keep_model = request.config.getoption("--keep-models")
        if not self.model:
            test_name = request.node.name.replace("_", "-")
            self.model = f"pytest-operator-{test_name}"
            self.create_model = True
        else:
            self.create_model = False

    @property
    def connection(self):
        if not self.model:
            return None
        elif self.controller:
            return f"{self.controller}:{self.model}"
        else:
            return self.model

    async def run(self, cmd, *args, input=None, **kwargs):
        cmd = str(cmd)
        args = [str(a) for a in args]
        kwargs.setdefault("env", os.environ.copy())
        kwargs.setdefault("stdout", asyncio.subprocess.PIPE)
        kwargs.setdefault("stderr", asyncio.subprocess.PIPE)
        if input is not None:
            kwargs["stdin"] = asyncio.subprocess.PIPE
        proc = await asyncio.create_subprocess_exec(
            cmd,
            *args,
            **kwargs,
        )

        if hasattr(input, "encode"):
            input = input.encode("utf8")

        stdout, stderr = await proc.communicate(input=input)
        if stdout is not None:
            stdout = stdout.decode("utf8")
        if stderr is not None:
            stderr = stderr.decode("utf8")
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(
                proc.returncode, [cmd] + args, stdout, stderr
            )
        return stdout, stderr

    async def juju_wait(self, *args, **kwargs):
        cmd = ["juju-wait", "-e", self.connection, "-w"]
        if args:
            cmd.extend(args)
        if "timeout_secs" in kwargs and kwargs["timeout_secs"]:
            cmd.extend(["-t", str(kwargs["timeout_secs"])])
        try:
            return await self.run(*cmd)
        except subprocess.CalledProcessError as e:
            raise AssertionError(f"juju-wait failed:\n{e.stderr}\n{e.stdout}") from e

    async def build_charm(self, src):
        """Builds a charm and returns the path to the built charm file.

        This can handle charms using the older charms.reactive framework as
        well as charms written against the modern operator framework.
        """
        src = Path(src)
        metadata_path = src / "metadata.yaml"
        layer_path = src / "layer.yaml"
        if layer_path.exists():
            # Handle older, reactive framework charms.
            cmd = ["charm-build", "-F", src]
        else:
            # Handle newer, operator framework charms.
            cmd = ["charmcraft", "build", "-f", src]
        await self.run(*cmd, cwd=str(self.tmp_path))
        if not layer_path.exists():
            # Clean up build dir created by charmcraft.
            build_path = src / "build"
            if build_path.exists():
                shutil.rmtree(build_path)
        charm_name = yaml.safe_load(metadata_path.read_text())["name"]
        return self.tmp_path / f"{charm_name}.charm"


@pytest.fixture(scope="session")
def check_deps(autouse=True):
    missing = []
    for dep in ("juju", "charm-build", "charmcraft", "juju-wait"):
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


@pytest.fixture(scope="function")
def operator_tools(request, tmp_path):
    return OperatorTools(request, tmp_path)


@pytest.fixture(scope="function")
async def operator_model(request, operator_tools):
    if operator_tools.create_model:
        controller = Controller()
        if operator_tools.controller:
            await controller.connect(operator_tools.controller)
        else:
            await controller.connect_current()
        model = await controller.add_model(
            operator_tools.model, cloud_name=operator_tools.cloud
        )
        await controller.disconnect()
    else:
        model = Model()
        await model.connect(operator_tools.connection)
    yield model
    if operator_tools.create_model and not operator_tools.keep_model:
        controller = await model.get_controller()
        await model.disconnect()
        await controller.destroy_model(operator_tools.model)
        await controller.disconnect()
    else:
        await model.disconnect()
