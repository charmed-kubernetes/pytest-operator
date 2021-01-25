import pytest
from pathlib import Path
from pytest_operator.plugin import charm_build

pytest_plugins = "pytester"
fixtures_path = Path("fixtures")


@pytest.mark.asyncio
async def test_operator_connection(operatormodel, operatortools):
    assert operatormodel.applications.keys()


@pytest.mark.asyncio
async def test_build_reactive_charm():
    charm_path = fixtures_path / "test-charm"
    await charm_build(str(charm_path))
    assert Path("test-charm.charm").exists()


@pytest.mark.asyncio
async def test_build_ops_charm():
    charm_path = fixtures_path / "new-style-charm"
    await charm_build(str(charm_path))
    assert Path("new-style-charm.charm").exists()
