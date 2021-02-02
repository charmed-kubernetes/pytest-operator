import pytest
from pathlib import Path

tests_data_path = Path(__file__).parent / "data"
tests_charms_path = tests_data_path / "charms"


@pytest.mark.asyncio
async def test_reactive_charm(operator_model, operator_tools):
    src = tests_charms_path / "reactive-framework"
    dst = await operator_tools.build_charm(src)
    assert dst.exists()
    assert dst.name == "reactive-framework.charm"
    await operator_model.deploy(dst)
    await operator_tools.juju_wait()


@pytest.mark.asyncio
async def test_operator_charm(operator_model, operator_tools):
    src = tests_charms_path / "operator-framework"
    dst = await operator_tools.build_charm(src)
    assert dst.exists()
    assert dst.name == "operator-framework.charm"
    await operator_model.deploy(dst)
    await operator_tools.juju_wait()
