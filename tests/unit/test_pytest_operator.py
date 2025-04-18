import asyncio
import logging
import os
from unittest.mock import Mock, AsyncMock, ANY, patch, call, MagicMock, PropertyMock
from urllib.error import HTTPError
from pathlib import Path
from types import SimpleNamespace
from websockets.exceptions import ConnectionClosed
from zipfile import ZipFile
import pytest

from pytest_operator import plugin

log = logging.getLogger(__name__)

ENV = {_: os.environ.get(_) for _ in ["HOME", "TOX_ENV_DIR"]}


@patch.object(plugin, "check_deps", Mock())
@patch.object(plugin.OpsTest, "_setup_model", AsyncMock())
@patch.object(plugin.OpsTest, "_cleanup_models", AsyncMock())
def test_tmp_path_with_tox(pytester):
    pytester.makepyfile(
        f"""
        import os
        from pathlib import Path

        os.environ.update(**{ENV})
        async def test_with_tox(ops_test):
            expected_base = Path("{ENV["TOX_ENV_DIR"]}") / "tmp" / "pytest"
            common = os.path.commonpath([ops_test.tmp_path, expected_base])
            assert expected_base == Path(common)
        """
    )
    result = pytester.runpytest()
    result.assert_outcomes(passed=1)


@patch.object(plugin, "check_deps", Mock())
@patch.object(plugin.OpsTest, "_setup_model", AsyncMock())
@patch.object(plugin.OpsTest, "_cleanup_models", AsyncMock())
def test_tmp_path_without_tox(request, pytester):
    pytester.makepyfile(
        f"""
        import os
        from pathlib import Path

        os.environ.update(**{ENV})
        async def test_without_tox(request, ops_test):
            unexpected_base = Path("{ENV["TOX_ENV_DIR"]}") / "tmp" / "pytest"
            common = os.path.commonpath([ops_test.tmp_path, unexpected_base])
            assert unexpected_base != Path(common)

            expected_base = Path("/tmp/pytest")
            common = os.path.commonpath([ops_test.tmp_path, expected_base])
            assert expected_base == Path(common)
        """
    )
    result = pytester.runpytest("--basetemp=/tmp/pytest")
    result.assert_outcomes(passed=1)


@pytest.fixture()
def mock_runner(monkeypatch):
    patch = monkeypatch.setattr
    patch(plugin.os, "getgroups", mock_getgroups := Mock(return_value=[]))
    patch(plugin.grp, "getgrall", mock_getgrall := Mock(return_value=[]))
    patch(plugin.grp, "getgrgid", Mock(return_value=Mock(gr_name="lxd")))
    patch(plugin.OpsTest, "run", mock_run := AsyncMock(return_value=(0, "", "")))
    yield mock_getgroups, mock_getgrall, mock_run


async def test_build_with_args(setup_request, mock_runner, tmp_path_factory):
    mock_getgroups, _, mock_run = mock_runner
    setup_request.config.option.charmcraft_args = ["--platform=special-platform"]
    ops_test = plugin.OpsTest(setup_request, tmp_path_factory)
    build_path = "tests/data/charms/operator-framework"
    mock_getgroups.return_value = [ANY]
    ops_test.destructive_mode = False
    with pytest.raises(FileNotFoundError) as exc_info:
        await ops_test.build_charm(build_path)

    assert str(exc_info.value) == f"No such file in '{build_path}/*.charm'"
    assert mock_run.called
    assert mock_run.call_args[0] == (
        "charmcraft",
        "pack",
        "--platform=special-platform",
    )


@patch("pathlib.Path.glob")
@patch("pathlib.Path.rename")
async def test_build_return_all(
    renamed, mock_glob, setup_request, mock_runner, tmp_path_factory
):
    mock_getgroups, _, mock_run = mock_runner
    setup_request.config.option.charmcraft_args = ["--platform=special-platform"]
    ops_test = plugin.OpsTest(setup_request, tmp_path_factory)
    charm_path = Path("tests/data/charms/operator-framework")
    mock_getgroups.return_value = [ANY]
    ops_test.destructive_mode = False
    renamed.side_effect = lambda _: _
    mock_glob.return_value = [
        Path(f"tests/data/charms/operator-framework/{i}.charm") for i in range(3)
    ]
    built = await ops_test.build_charm(charm_path, return_all=True)

    assert mock_run.called
    assert mock_run.call_args[0] == (
        "charmcraft",
        "pack",
        "--platform=special-platform",
    )
    expected_dest = ops_test.tmp_path / "charms"
    assert len(built) == 3, "All built charms should be returned"
    assert all(
        str(f).startswith(str(expected_dest)) for f in built
    ), "All built charms should be in the same directory"


@patch(
    "pathlib.Path.glob",
)
@patch("pathlib.Path.rename")
async def test_build_return_one(
    renamed, mock_glob, setup_request, mock_runner, tmp_path_factory
):
    mock_getgroups, _, mock_run = mock_runner
    setup_request.config.option.charmcraft_args = ["--platform=special-platform"]
    ops_test = plugin.OpsTest(setup_request, tmp_path_factory)
    charm_path = Path("tests/data/charms/operator-framework")
    ops_test.destructive_mode = False
    renamed.side_effect = lambda _: _
    mock_glob.return_value = [
        Path(f"tests/data/charms/operator-framework/{i}.charm") for i in range(3)
    ]

    mock_getgroups.return_value = [ANY]
    ops_test.destructive_mode = False
    built = await ops_test.build_charm(charm_path)

    assert mock_run.called
    assert mock_run.call_args[0] == (
        "charmcraft",
        "pack",
        "--platform=special-platform",
    )
    expected_dest = ops_test.tmp_path / "charms" / "0.charm"
    assert built == expected_dest, "Only the first built charm should be returned"


async def test_destructive_mode(setup_request, mock_runner, tmp_path_factory):
    mock_getgroups, mock_getgrall, mock_run = mock_runner
    ops_test = plugin.OpsTest(setup_request, tmp_path_factory)

    mock_run.return_value = (1, "", "")
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
    assert mock_run.call_args[0] == ("sudo", "-g", "lxd", "-E", "charmcraft", "pack")

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
            mock_url_open.return_value.__enter__.return_value = SimpleNamespace(
                status=200, read=lambda: resp
            )
            yield mock_url_open

    def test_info_api(self, info_api):
        ch = plugin.Charmhub("etcd", "latest/edge")
        assert ch.info["default-release"]["channel"]["risk"] == "edge"
        assert ch.info["default-release"]["channel"]["track"] == "latest"
        info_api.assert_called_with(
            "https://api.charmhub.io/v2/charms/info/etcd"
            "?channel=latest%2Fedge&fields=default-release.resources"
        )

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
        info_api.assert_called_with(
            "https://api.charmhub.io/v2/charms/info/etcd"
            "?channel=latest%2Fedge&fields=default-release.resources"
        )

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
            if "meta/id" in url:
                resp = b'{"Id": "etcd-668"}'
            elif url.endswith("core") or url.endswith("snapshot"):
                resp = b'{"Revision": 0}'
            elif url.endswith("etcd"):
                resp = b'{"Revision": 3}'
            else:
                raise FileNotFoundError(f"Unexpected url: {url}")
            response = MagicMock()
            response.__enter__.return_value = SimpleNamespace(
                status=200, read=lambda: resp
            )
            return response  # returns an object to use as a context manager

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
        CH_URL = "https://api.jujucharms.com/charmstore/v5/etcd-668/resource"
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


async def test_plugin_build_resources(setup_request, tmp_path_factory):
    ops_test = plugin.OpsTest(setup_request, tmp_path_factory)
    ops_test.jujudata = Mock()
    ops_test.jujudata.path = ""
    dst_dir = ops_test.tmp_path / "resources"
    expected_resources = [1, 2, 3]

    with pytest.raises(FileNotFoundError):
        build_script = Path("tests") / "data" / "build_resources_does_not_exist.sh"
        await ops_test.build_resources(build_script)

    build_script = Path("tests") / "data" / "build_resources_errors.sh"
    with patch.object(
        ops_test, "run", AsyncMock(return_value=(1, "", "error"))
    ) as mock_run:
        resources = await ops_test.build_resources(build_script)
    assert not resources, ""
    mock_run.assert_called_once_with(
        "sudo", str(build_script.absolute()), cwd=dst_dir, check=False
    )

    build_script = Path("tests") / "data" / "build_resources.sh"
    with patch(
        "pytest_operator.plugin.Path.glob", Mock(return_value=expected_resources)
    ):
        with patch.object(
            ops_test, "run", AsyncMock(return_value=(0, "okay", ""))
        ) as mock_run:
            resources = await ops_test.build_resources(build_script)
    assert resources and resources == expected_resources


async def test_plugin_get_resources(setup_request, tmp_path_factory, resource_charm):
    ops_test = plugin.OpsTest(setup_request, tmp_path_factory)
    resources = ops_test.arch_specific_resources(resource_charm)
    assert resources.keys() == {"resource-file-arm64", "resource-file"}
    assert resources["resource-file-arm64"].arch == "arm64"
    assert resources["resource-file"].arch == "amd64"


@patch(
    "pytest_operator.plugin.CharmStore._charm_id",
    new=Mock(return_value="resourced-charm-1"),
)
async def test_plugin_fetch_resources(setup_request, tmp_path_factory, resource_charm):
    ops_test = plugin.OpsTest(setup_request, tmp_path_factory)
    ops_test.jujudata = Mock()
    ops_test.jujudata.path = ""
    arch_resources = ops_test.arch_specific_resources(resource_charm)

    def dl_rsc(resource, dest_path):
        assert isinstance(resource, str)
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


async def test_async_render_bundles(setup_request, tmp_path_factory):
    ops_test = plugin.OpsTest(setup_request, tmp_path_factory)
    ops_test.jujudata = Mock()
    ops_test.jujudata.path = ""

    with pytest.raises(TypeError):
        await ops_test.async_render_bundles(1234)

    str_template = "a: {{ num }}"
    bundles = await ops_test.async_render_bundles(str_template, num=1)
    assert bundles[0].read_text() == "a: 1"

    template_file = ops_test.tmp_path / "str_path.yml.j2"
    template_file.write_text("a: {{ num }}")
    bundles = await ops_test.async_render_bundles(str(template_file), num=1)
    assert bundles[0].read_text() == "a: 1"

    template_file = ops_test.tmp_path / "path.yaml"
    template_file.write_text("a: {{ num }}")
    bundles = await ops_test.async_render_bundles(template_file, num=1)
    assert bundles[0].read_text() == "a: 1"

    download_bundle = ops_test.Bundle("downloaded")
    (ops_test.tmp_path / "bundles").mkdir(exist_ok=True)
    with ZipFile(ops_test.tmp_path / "bundles" / "downloaded.bundle", "w") as zf:
        zf.writestr("bundle.yaml", "a: {{ num }}")
    with patch.object(ops_test, "juju", AsyncMock(return_value=(0, "", ""))):
        bundles = await ops_test.async_render_bundles(download_bundle, num=1)
    assert bundles[0].read_text() == "a: 1"


@pytest.mark.parametrize(
    "crash_dump, no_crash_dump, n_testsfailed, keep_models, expected_crashdump",
    [
        # crash_dump == always && no_crash_dump == False -> always dump
        ("always", False, 0, True, True),
        ("always", False, 0, False, True),
        ("always", False, 1, True, True),
        ("always", False, 1, False, True),
        # crash_dump == always && no_crash_dump == True -> never dump
        ("always", True, 0, True, False),
        ("always", True, 0, False, False),
        ("always", True, 1, True, False),
        ("always", True, 1, False, False),
        # crash_dump == on-failure && no_crash_dump == False -> dump on failures
        ("on-failure", False, 0, True, False),
        ("on-failure", False, 0, False, False),
        ("on-failure", False, 1, True, True),
        ("on-failure", False, 1, False, True),
        # crash_dump == legacy && no_crash_dump == False ->
        #   dump if failure and keep_model==False
        ("legacy", False, 0, True, False),
        ("legacy", False, 0, False, False),
        ("legacy", False, 1, True, False),
        ("legacy", False, 1, False, True),
        # crash_dump == never -> never dump
        ("never", False, 0, True, False),
        ("never", False, 0, False, False),
        ("never", False, 1, True, False),
        ("never", False, 1, False, False),
    ],
)
async def test_crash_dump_mode(
    setup_request,
    crash_dump,
    no_crash_dump,
    n_testsfailed,
    keep_models,
    expected_crashdump,
    monkeypatch,
    tmp_path_factory,
):
    """Test running juju-crashdump in OpsTest.cleanup."""
    patch = monkeypatch.setattr
    patch(plugin.OpsTest, "run", mock_run := AsyncMock(return_value=(0, "", "")))
    setup_request.config.option.crash_dump = crash_dump
    setup_request.config.option.no_crash_dump = no_crash_dump
    setup_request.config.option.crash_dump_args = "-c --bug=1234567"
    setup_request.config.option.keep_models = False
    ops_test = plugin.OpsTest(setup_request, tmp_path_factory)
    model = MagicMock()
    model.machines.values.return_value = []
    model.disconnect = AsyncMock()
    model.block_until = AsyncMock()
    ops_test._current_alias = "main"
    ops_test._models = {
        ops_test.current_alias: plugin.ModelState(
            model, keep_models, False, "test", "local", "model"
        )
    }
    ops_test.crash_dump_output = None
    ops_test.log_model = AsyncMock()
    ops_test._controller = AsyncMock()

    setup_request.session.testsfailed = n_testsfailed

    await ops_test._cleanup_model()

    if expected_crashdump:
        mock_run.assert_called_once_with(
            "juju-crashdump",
            "-s",
            "-m=test:model",
            "-a=debug-layer",
            "-a=config",
            "-c",
            "--bug=1234567",
        )
    else:
        mock_run.assert_not_called()


def test_crash_dump_mode_invalid_input(setup_request, monkeypatch, tmp_path_factory):
    """Test running juju-crashdump in OpsTest.cleanup."""
    patch = monkeypatch.setattr
    patch(plugin.OpsTest, "run", AsyncMock(return_value=(0, "", "")))
    setup_request.config.option.crash_dump = "not-a-real-option"
    setup_request.config.option.crash_dump_args = ""
    setup_request.config.option.no_crash_dump = False
    setup_request.config.option.keep_models = False

    with pytest.raises(ValueError):
        plugin.OpsTest(setup_request, tmp_path_factory)


async def test_create_crash_dump(monkeypatch, tmp_path_factory):
    """Test running create crash dump."""

    async def mock_run(*cmd):
        proc = await asyncio.create_subprocess_exec("not-valid-command")
        await proc.communicate()

    patch = monkeypatch.setattr
    patch(plugin.OpsTest, "run", mock_run)
    mock_request = Mock(**{"module.__name__": "test"})
    mock_request.config.option.crash_dump_args = ""
    mock_request.config.option.juju_max_frame_size = None
    patch(plugin, "log", mock_log := MagicMock())
    ops_test = plugin.OpsTest(mock_request, tmp_path_factory)
    await ops_test.create_crash_dump()
    mock_log.info.assert_any_call("juju-crashdump command was not found.")


def test_no_deploy_mode(pytester):
    """Test running no deploy mode."""
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.abort_on_fail
        @pytest.mark.skip_if_deployed
        def test_build_and_deploy():
            pass

        def test_01():
            pass

        def test_02():
            pass
    """
    )
    # test without --no-deploy option
    result = pytester.runpytest("--asyncio-mode=auto")
    result.assert_outcomes(passed=3)

    # test with --no-deploy, but without --model option
    result = pytester.runpytest("--no-deploy", "--asyncio-mode=auto")
    assert any(
        "error: must specify --model when using --no-deploy" in errline
        for errline in result.errlines
    )
    assert result.outlines == []

    # test with --no-deploy and --model
    result = pytester.runpytest(
        "--no-deploy", "--model", "test-model", "--asyncio-mode=auto"
    )
    result.assert_outcomes(passed=2, skipped=1)


@pytest.fixture(autouse=True)
def mock_juju():
    juju = SimpleNamespace()
    with patch("pytest_operator.plugin.Model", autospec=True) as MockModel, patch(
        "pytest_operator.plugin.Controller", autospec=True
    ) as MockController:
        juju.controller = MockController.return_value
        juju.model = MockModel.return_value

        juju.controller.controller_name = "this-controller"
        juju.controller.get_cloud = AsyncMock(return_value="this-cloud")
        juju.controller.add_model = AsyncMock(return_value=juju.model)
        juju.controller.model_uuids = AsyncMock(return_value={})
        juju.controller.list_models = AsyncMock(return_value=[])
        juju.model.get_controller = AsyncMock(return_value=juju.controller)
        yield juju


@pytest.fixture
def setup_request(request, mock_juju):
    mock_request = MagicMock()
    mock_request.module.__name__ = request.node.name
    mock_request.config.option.controller = mock_juju.controller.controller_name
    mock_request.config.option.model = None
    mock_request.config.option.cloud = None
    mock_request.config.option.model_alias = "main"
    mock_request.config.option.model_config = None
    mock_request.config.option.keep_models = False
    mock_request.config.option.destroy_storage = False
    mock_request.config.option.juju_max_frame_size = None
    mock_request.config.option.charmcraft_platform = None
    yield mock_request


@pytest.mark.parametrize("max_frame_size", [None, 2**16])
async def test_fixture_set_up_existing_model(
    mock_juju, setup_request, tmp_path_factory, max_frame_size
):
    setup_request.config.option.model = "this-model"
    setup_request.config.option.juju_max_frame_size = max_frame_size
    expected_kwargs = {}
    if max_frame_size:
        expected_kwargs["max_frame_size"] = max_frame_size

    ops_test = plugin.OpsTest(setup_request, tmp_path_factory)
    assert ops_test.model is None

    mock_juju.controller.list_models = AsyncMock(return_value=["this-model"])
    await ops_test._setup_model()
    mock_juju.model.connect.assert_called_with(
        "this-controller:this-model", **expected_kwargs
    )
    assert ops_test.model == mock_juju.model
    assert ops_test.model_full_name == "this-controller:this-model"
    assert ops_test.cloud_name is None
    assert ops_test.model_name == "this-model"
    assert ops_test.keep_model is True, "Model should be kept if it already exists"
    assert (
        ops_test.destroy_storage is False
    ), "Storage should not be destroyed by default"
    assert len(ops_test.models) == 1


async def test_fixture_invalid_max_frame_size(setup_request, tmp_path_factory):
    setup_request.config.option.model = "this-model"
    setup_request.config.option.juju_max_frame_size = -1
    with pytest.raises(ValueError):
        plugin.OpsTest(setup_request, tmp_path_factory)


@patch("pytest_operator.plugin.OpsTest.forget_model")
@patch("pytest_operator.plugin.OpsTest.run")
async def test_fixture_cleanup_multi_model(
    mock_run, mock_forget_model, setup_request, tmp_path_factory
):
    ops_test = plugin.OpsTest(setup_request, tmp_path_factory)
    await ops_test._setup_model()
    await ops_test.track_model("secondary")
    assert len(ops_test.models) == 2
    await ops_test._cleanup_models()
    mock_run.assert_has_calls([call("juju", "models")] * 2)
    mock_forget_model.assert_has_calls(
        [
            call("secondary"),
            call("main"),
        ],
        any_order=False,
    )


@patch("pytest_operator.plugin.OpsTest.forget_model", AsyncMock())
@patch("pytest_operator.plugin.OpsTest.run", AsyncMock())
@pytest.mark.parametrize(
    "global_flag, keep, expected",
    [
        (None, None, False),
        (None, True, True),
        (None, False, False),
        (None, plugin.OpsTest.ModelKeep.ALWAYS, True),
        (None, plugin.OpsTest.ModelKeep.NEVER, False),
        (None, plugin.OpsTest.ModelKeep.IF_EXISTS, False),
        (True, None, True),
        (True, True, True),
        (True, False, True),
        (True, plugin.OpsTest.ModelKeep.ALWAYS, True),
        (True, plugin.OpsTest.ModelKeep.NEVER, False),
        (True, plugin.OpsTest.ModelKeep.IF_EXISTS, True),
        (True, "ALWAYS", True),
    ],
)
async def test_model_keep_options(
    global_flag, keep, expected, setup_request, tmp_path_factory
):
    setup_request.config.option.keep_models = global_flag
    ops_test = plugin.OpsTest(setup_request, tmp_path_factory)
    await ops_test._setup_model()
    with ops_test.model_context("main"):
        assert ops_test.keep_model is bool(
            global_flag
        ), "main model should follow global flag"

    await ops_test.track_model("secondary", keep=keep)
    with ops_test.model_context("secondary"):
        assert (
            ops_test.keep_model is expected
        ), f"{ops_test.model_full_name} should follow configured keep"


@patch("pytest_operator.plugin.OpsTest.forget_model", AsyncMock())
@patch("pytest_operator.plugin.OpsTest.run", AsyncMock())
@pytest.mark.parametrize(
    "global_flag, destroy_storage, expected",
    [
        (False, None, False),
        (False, True, True),
        (False, False, False),
        (True, None, True),
        (True, True, True),
        (True, False, False),
    ],
)
async def test_destroy_storage_options(
    global_flag, destroy_storage, expected, setup_request, tmp_path_factory
):
    setup_request.config.option.destroy_storage = global_flag
    ops_test = plugin.OpsTest(setup_request, tmp_path_factory)
    await ops_test._setup_model()
    with ops_test.model_context("main"):
        assert ops_test.destroy_storage is bool(
            global_flag
        ), "main model should follow global flag"

    await ops_test.track_model("secondary", destroy_storage=destroy_storage)
    with ops_test.model_context("secondary"):
        assert (
            ops_test.destroy_storage is expected
        ), f"{ops_test.model_full_name} should follow configured destroy_storage"


@patch("pytest_operator.plugin.OpsTest.default_model_name", new_callable=PropertyMock)
@patch("pytest_operator.plugin.OpsTest.juju", autospec=True)
async def test_fixture_set_up_automatic_model(
    juju_cmd, mock_default_model_name, mock_juju, setup_request, tmp_path_factory
):
    model_name = "this-auto-generated-model-name"
    mock_default_model_name.return_value = model_name
    ops_test = plugin.OpsTest(setup_request, tmp_path_factory)
    assert ops_test.model is None

    await ops_test._setup_model()
    mock_juju.controller.add_model.assert_called_with(
        model_name, "this-cloud", credential_name=None, config=None
    )
    juju_cmd.assert_called_with(ops_test, "models")
    assert ops_test.model == mock_juju.model
    assert ops_test.model_full_name == f"this-controller:{model_name}"
    assert ops_test.cloud_name == "this-cloud"
    assert ops_test.model_name == model_name
    # Don't keep a model if it's automatically generated
    assert ops_test.keep_model is False
    assert len(ops_test.models) == 1


@pytest.mark.parametrize("model_name", [None, "alt-model"])
@pytest.mark.parametrize("cloud_name", [None, "alt-cloud"])
@pytest.mark.parametrize(
    "block_exception",
    [None, asyncio.TimeoutError(), ConnectionClosed(1, "test", False)],
)
@patch("pytest_operator.plugin.OpsTest.juju", autospec=True)
async def test_fixture_create_remove_model(
    juju_cmd,
    model_name,
    cloud_name,
    block_exception,
    mock_juju,
    setup_request,
    tmp_path_factory,
):
    juju_cmd.return_value = (0, "", "")
    setup_request.session.testsfailed = 0
    ops_test = plugin.OpsTest(setup_request, tmp_path_factory)
    await ops_test._setup_model()
    assert len(ops_test.models) == 1
    first_model = SimpleNamespace(
        alias=ops_test.current_alias,
        model=ops_test.model,
        controller_name=ops_test.controller_name,
        model_name=ops_test.model_name,
        cloud_name=ops_test.cloud_name,
        tmp_path=ops_test.tmp_path,
    )

    test_alias = "model-alias"
    await ops_test.track_model(test_alias, model_name=model_name, cloud_name=cloud_name)
    juju_cmd.assert_called_with(ops_test, "models")

    # Adding a model doesn't switch the current model.
    assert ops_test.current_alias == "main"
    assert len(ops_test.models) == 2

    # Now let's switch into it
    with ops_test.model_context(test_alias):
        test_model = SimpleNamespace(
            alias=ops_test.current_alias,
            model=ops_test.model,
            controller_name=ops_test.controller_name,
            model_name=ops_test.model_name,
            cloud_name=ops_test.cloud_name,
            tmp_path=ops_test.tmp_path,
            keep_model=ops_test.keep_model,
        )
        # Within this context the current alias is updated
        assert ops_test.current_alias == test_alias

    # leaving this context, current alias switches back to 'main'
    assert ops_test.current_alias == "main"
    # And all properties reflect the main model
    assert ops_test.model_name == first_model.model_name
    assert ops_test.model == first_model.model
    assert ops_test.cloud_name == first_model.cloud_name
    assert ops_test.controller_name == first_model.controller_name
    assert ops_test.tmp_path == first_model.tmp_path

    if model_name is None:
        generated = setup_request.module.__name__.replace("_", "-")
        assert test_model.model_name.startswith(generated)
    else:
        assert test_model.model_name == model_name
    assert first_model.alias != test_alias, "Model Alias must be different"

    if cloud_name is None:
        # cloud-names should match if cloud-name is None
        assert first_model.cloud_name == test_model.cloud_name

    # controller-names should match
    assert first_model.controller_name == test_model.controller_name

    # New tmp_path should be generated
    assert first_model.tmp_path != test_model.tmp_path

    # Created models shouldn't be kept
    assert not test_model.keep_model

    if block_exception:
        mock_juju.model.block_until = AsyncMock(side_effect=block_exception)
    await ops_test.forget_model(test_alias, timeout=1.0)

    # Should be back to managing only one model
    assert len(ops_test.models) == 1

    # Ensure switching to that context fails
    with pytest.raises(plugin.ModelNotFoundError):
        with ops_test.model_context(test_alias):
            pass
