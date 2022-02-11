import asyncio
import pathlib
from unittest.mock import Mock, AsyncMock, ANY, MagicMock

import pytest

from pytest_operator import plugin


async def test_destructive_mode(monkeypatch, tmp_path_factory):
    patch = monkeypatch.setattr
    patch(plugin.os, "getgroups", mock_getgroups := Mock(return_value=[]))
    patch(plugin.grp, "getgrall", mock_getgrall := Mock(return_value=[]))
    patch(plugin.grp, "getgrgid", Mock(return_value=Mock(gr_name="lxd")))
    patch(plugin.OpsTest, "run", mock_run := AsyncMock(return_value=(1, "", "")))
    ops_test = plugin.OpsTest(Mock(**{"module.__name__": "test"}), tmp_path_factory)

    ops_test.destructive_mode = True
    try:
        await ops_test.build_charm("tests/data/charms/operator-framework")
    except RuntimeError as e:
        # We didn't actually build it
        assert str(e).startswith("Failed to build charm")
    assert mock_run.called
    assert mock_run.call_args[0] == ("charmcraft", "pack", "--destructive-mode")

    mock_run.reset_mock()
    ops_test.destructive_mode = False
    try:
        await ops_test.build_charm("tests/data/charms/operator-framework")
    except AssertionError as e:
        # Expected failure
        assert str(e).startswith("Group 'lxd' required")
    else:
        pytest.fail("Missing expected lxd group assertion")
    assert not mock_run.called

    mock_getgrall.return_value = [Mock(gr_name="lxd")]
    try:
        await ops_test.build_charm("tests/data/charms/operator-framework")
    except RuntimeError as e:
        # We didn't actually build it
        assert str(e).startswith("Failed to build charm")
    assert mock_run.called
    assert mock_run.call_args[0] == ("sg", "lxd", "-c", "charmcraft pack")

    mock_getgroups.return_value = [ANY]
    try:
        await ops_test.build_charm("tests/data/charms/operator-framework")
    except RuntimeError as e:
        # We didn't actually build it
        assert str(e).startswith("Failed to build charm")
    assert mock_run.called
    assert mock_run.call_args[0] == ("charmcraft", "pack")


async def test_crash_dump_mode(monkeypatch, tmp_path_factory):
    """Test running juju-crashdump in OpsTest.cleanup."""
    patch = monkeypatch.setattr
    patch(plugin.OpsTest, "run", mock_run := AsyncMock(return_value=(0, "", "")))
    ops_test = plugin.OpsTest(
        mock_request := Mock(**{"module.__name__": "test"}), tmp_path_factory
    )
    ops_test.crash_dump = True
    ops_test.keep_model = False
    ops_test.model = MagicMock()
    ops_test.model.machines.values.return_value = []
    ops_test.model.disconnect = AsyncMock()
    ops_test.model_full_name = "test-model"
    ops_test.log_model = AsyncMock()
    ops_test._controller = AsyncMock()

    # 0 tests failed
    mock_request.session.testsfailed = 0

    await ops_test._cleanup_model()

    mock_run.assert_not_called()
    mock_run.reset_mock()

    # 1 tests failed
    mock_request.session.testsfailed = 1

    await ops_test._cleanup_model()

    mock_run.assert_called_once_with(
        "juju-crashdump",
        "-s",
        "-m",
        "test-model",
        "-a",
        "debug-layer",
        "-a",
        "config",
        "-o",
        plugin._source_charm_dir(ops_test.tmp_path),
    )
    mock_run.reset_mock()


async def test_create_crash_dump(monkeypatch, tmp_path_factory):
    """Test running create crash dump."""

    async def mock_run(*cmd):
        proc = await asyncio.create_subprocess_exec("not-valid-command")
        await proc.communicate()

    patch = monkeypatch.setattr
    patch(plugin.OpsTest, "run", mock_run)
    patch(plugin, "log", mock_log := MagicMock())
    ops_test = plugin.OpsTest(Mock(**{"module.__name__": "test"}), tmp_path_factory)
    await ops_test.create_crash_dump()
    mock_log.info.assert_any_call("juju-crashdump command was not found.")


@pytest.mark.parametrize(
    "path, exp_source_path",
    [
        (pathlib.Path("/test/.tox/a/b/c/d"), pathlib.Path("/test")),
        (pathlib.Path("/test/a/b/c/d/.tox/a/b/c/d"), pathlib.Path("/test/a/b/c/d")),
        (pathlib.Path("/test/a/b/c/d"), None),
    ],
)
def test_source_charm_dir(path, exp_source_path):
    """Test getting source charm dir."""
    assert (
        plugin._source_charm_dir(path) == exp_source_path
    ), "the path {} does not contain {}".format(path, exp_source_path)
