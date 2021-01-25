import os
import pytest
import asyncio
from pathlib import Path
from juju.model import Model as JujuModel
from aioify import aioify
from subprocess import run


def pytest_addoption(parser):

    parser.addoption("--controller", action="store", help="Juju controller to use")

    parser.addoption("--model", action="store", help="Juju model to use")

    parser.addoption(
        "--cloud", action="store", help="Juju cloud to use"
    )


class ToolsObj:
    """Utility class for accessing juju related tools"""

    def __init__(self, request):
        self._request = request

    async def _load(self):
        request = self._request
        self.controller_name = request.config.getoption("--controller")
        self.model_name = request.config.getoption("--model")
        self.cloud = request.config.getoption("--cloud")
        self.connection = None
        if self.model_name and self.cloud:
            self.connection = f"{self.controller_name}:{self.model_name}"

    async def run(self, cmd, *args, input=None):
        proc = await asyncio.create_subprocess_exec(
            cmd,
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ.copy(),
        )

        if hasattr(input, "encode"):
            input = input.encode("utf8")

        stdout, stderr = await proc.communicate(input=input)
        if proc.returncode != 0:
            raise Exception(
                f"Problem with run command {cmd} (exit {proc.returncode}):\n"
                f"stdout:\n{stdout.decode()}\n"
                f"stderr:\n{stderr.decode()}\n"
            )
        return stdout.decode("utf8"), stderr.decode("utf8")

    async def juju_wait(self, *args, **kwargs):
        cmd = ["juju-wait", "-e", self.connection, "-w"]
        if args:
            cmd.extend(args)
        if "timeout_secs" in kwargs and kwargs["timeout_secs"]:
            cmd.extend(["-t", str(kwargs["timeout_secs"])])
        return await self.run(*cmd)


@pytest.fixture(scope="session")
def check_deps(autouse=True):
    missing = []
    for dep in ("juju", "charm-build", "charmcraft", "juju-wait"):
        res = run(["which", dep])
        if res.returncode != 0:
            missing.append(dep)
    if missing:
        raise RuntimeError(
            "Missing dependenc{}: {}".format(
                "y" if len(missing) == 1 else "ies",
                ", ".join(missing),
            )
        )


async def charm_build(src):
    """Builds both legacy and new style charms"""
    async_run = aioify(obj=run)
    src = Path(src)
    layer_path = src / "layer.yaml"
    if layer_path.exists():
        return await async_run(["charm-build", "-r", "-F", str(src)])
    return await async_run(["charmcraft", "build", "-f", str(src)])


@pytest.fixture(scope="function")
async def operatortools(request):
    _tools = ToolsObj(request)
    await _tools._load()
    return _tools


@pytest.fixture(scope="function")
async def operatormodel(request, event_loop, operatortools):
    _model = JujuModel(event_loop)
    if not operatortools.connection:
        await _model.connect_current()
    else:
        await _model.connect(operatortools.connection)
    yield _model
    await _model.disconnect()
