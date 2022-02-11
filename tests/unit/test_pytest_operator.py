import logging
from unittest.mock import Mock, AsyncMock, ANY, patch, call
from urllib.error import HTTPError
from pathlib import Path
from types import SimpleNamespace
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


class TestCharmhub:
    @pytest.fixture
    def info_api(self):
        with open("tests/data/etcd_ch_api_response.json") as f:
            resp = f.read()
        with patch("pytest_operator.plugin.urlopen") as mock_url_open:
            mock_url_open.return_value = SimpleNamespace(status=200, read=lambda: resp)
            yield mock_url_open

    def test_info_api(self, info_api):
        ch = plugin.Charmhub("etcd", "latest/edge")
        assert ch.info["default-release"]["channel"]["risk"] == "edge"
        assert ch.info["default-release"]["channel"]["track"] == "latest"
        info_api.assert_called_once()

    def test_exists(self, info_api):
        ch = plugin.Charmhub("etcd", "latest/edge")
        assert ch.exists

    def test_does_not_exist(self, info_api):
        info_api.side_effect = HTTPError(url="", code=404, msg="", hdrs=None, fp=None)
        ch = plugin.Charmhub("etcd", "latest/edge")
        assert not ch.exists

    def test_resource_map(self, info_api):
        ch = plugin.Charmhub("etcd", "latest/edge")
        assert len(ch.resource_map) == 3
        assert ch.resource_map.keys() == {"core", "etcd", "snapshot"}
        info_api.assert_called_once()

    def test_download_resource(self, info_api, tmpdir):
        CH_URL = (
            "https://api.charmhub.io/api/v1/"
            "resources/download/charm_8bULztKLC5fEw4Mc9gIeerQWey1pHICv"
        )
        ch = plugin.Charmhub("etcd", "latest/edge")
        with patch("pytest_operator.plugin.urlretrieve") as mock_rtrv:
            tmpdir = Path(tmpdir)
            mock_rtrv.return_value = tmpdir, None
            for rsc in ch.resource_map:
                ch.download_resource(rsc, tmpdir)
            mock_rtrv.assert_has_calls(
                [
                    call(f"{CH_URL}.core_0", tmpdir),
                    call(f"{CH_URL}.etcd_3", tmpdir),
                    call(f"{CH_URL}.snapshot_0", tmpdir),
                ]
            )


class TestCharmstore:
    @pytest.fixture
    def info_api(self):
        def mock_api(url):
            if "id-revision" in url:
                resp = b'{"Revision": 668}'
            elif url.endswith("core") or url.endswith("snapshot"):
                resp = b'{"Revision": 0}'
            elif url.endswith("etcd"):
                resp = b'{"Revision": 3}'
            else:
                raise FileNotFoundError(f"Unexpected url: {url}")
            return SimpleNamespace(status=200, read=lambda: resp)

        with patch("pytest_operator.plugin.urlopen") as mock_url_open:
            mock_url_open.side_effect = mock_api
            yield mock_url_open

    def test_exists(self, info_api):
        ch = plugin.CharmStore("cs:etcd", "edge")
        assert ch.exists

    def test_does_not_exist(self, info_api):
        info_api.side_effect = HTTPError(url="", code=404, msg="", hdrs=None, fp=None)
        ch = plugin.CharmStore("cs:etcd", "edge")
        assert not ch.exists

    def test_download_resource(self, info_api, tmpdir):
        CH_URL = "https://api.jujucharms.com/charmstore/v5/charm-668/resource"
        ch = plugin.CharmStore("cs:etcd", "edge")
        with patch("pytest_operator.plugin.urlretrieve") as mock_rtrv:
            tmpdir = Path(tmpdir)
            mock_rtrv.return_value = tmpdir, None
            ch.download_resource("core", tmpdir)
            ch.download_resource("etcd", tmpdir)
            ch.download_resource("snapshot", tmpdir)
            mock_rtrv.assert_has_calls(
                [
                    call(f"{CH_URL}/core/0", tmpdir),
                    call(f"{CH_URL}/etcd/3", tmpdir),
                    call(f"{CH_URL}/snapshot/0", tmpdir),
                ]
            )


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
    "pytest_operator.plugin.CharmStore._charm_id",
    new=Mock(return_value="resourced-charm-1"),
)
async def test_plugin_fetch_resources(tmp_path_factory, resource_charm):
    ops_test = plugin.OpsTest(Mock(**{"module.__name__": "test"}), tmp_path_factory)
    ops_test.jujudata = Mock()
    ops_test.jujudata.path = ""
    ops_test.model_full_name = ops_test.default_model_name
    arch_resources = ops_test.arch_specific_resources(resource_charm)

    def dl_rsc(resource, dest_path):
        assert type(resource) == str
        return dest_path

    with patch(
        "pytest_operator.plugin.CharmStore.download_resource", side_effect=dl_rsc
    ):
        downloaded = await ops_test.download_resources(
            resource_charm, resources=arch_resources
        )

    base = ops_test.tmp_path / "resources"
    expected_downloads = {
        "resource-file": base / "resource-file" / "resource-file.tgz",
        "resource-file-arm64": base / "resource-file-arm64" / "resource-file.tgz",
    }

    assert downloaded == expected_downloads
