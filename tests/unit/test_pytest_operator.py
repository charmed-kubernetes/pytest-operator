import logging
from unittest.mock import Mock, AsyncMock, ANY, patch
from pathlib import Path
from zipfile import ZipFile
import pytest

from pytest_operator import plugin

log = logging.getLogger(__name__)


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


@pytest.fixture(scope="module")
async def resource_charm(request, tmp_path_factory):
    """Creates a mock charm without building it."""
    dst_path = tmp_path_factory.mktemp(request.fixturename) / "resourced-charm.charm"
    charm_dir = Path("tests") / "data" / "charms" / "resourced-charm"
    with ZipFile(dst_path, mode="w") as zipfile:
        with zipfile.open("metadata.yaml", mode="w") as metadata:
            metadata.write((charm_dir / "metadata.yaml").read_bytes())
    yield dst_path


async def test_plugin_build_resources(tmp_path_factory):
    ops_test = plugin.OpsTest(Mock(**{"module.__name__": "test"}), tmp_path_factory)
    ops_test.jujudata = Mock()
    ops_test.jujudata.path = ""
    ops_test.model_full_name = ops_test.default_model_name

    with pytest.raises(FileNotFoundError):
        build_script = Path("tests") / "data" / "build_resources_does_not_exist.sh"
        await ops_test.build_resources(build_script)

    build_script = Path("tests") / "data" / "build_resources_errors.sh"
    resources = await ops_test.build_resources(build_script)
    assert not resources, ""

    build_script = Path("tests") / "data" / "build_resources.sh"
    resources = await ops_test.build_resources(build_script)
    assert resources and all(rsc.exists() for rsc in resources)


def test_plugin_get_resources(tmp_path_factory, resource_charm):
    ops_test = plugin.OpsTest(Mock(**{"module.__name__": "test"}), tmp_path_factory)
    resources = ops_test.arch_specific_resources(resource_charm)
    assert resources.keys() == {"resource-file-arm64", "resource-file"}
    assert resources["resource-file-arm64"].arch == "arm64"
    assert resources["resource-file"].arch == "amd64"


@patch(
    "pytest_operator.plugin.OpsTest._charm_id",
    new=Mock(return_value="resourced-charm-1"),
)
async def test_plugin_fetch_resources(tmp_path_factory, resource_charm):
    ops_test = plugin.OpsTest(Mock(**{"module.__name__": "test"}), tmp_path_factory)
    ops_test.jujudata = Mock()
    ops_test.jujudata.path = ""
    ops_test.model_full_name = ops_test.default_model_name
    arch_resources = ops_test.arch_specific_resources(resource_charm)

    def dl_rsc(charm_id, resource, dest_path):
        return dest_path

    def rename(resource, path):
        _, ext = path.name.split(".", 1)
        return path.parent / f"{resource}.{ext}"

    with patch("pytest_operator.plugin.OpsTest.download_resource", side_effect=dl_rsc):
        downloaded = await ops_test.download_resources(
            resource_charm, filter_in=lambda rsc: rsc in arch_resources, name=rename
        )

    expected_downloads = {
        "resource-file": ops_test.tmp_path / "resources" / "resource-file.tgz",
        "resource-file-arm64": ops_test.tmp_path
        / "resources"
        / "resource-file-arm64.tgz",
    }

    assert downloaded == expected_downloads
