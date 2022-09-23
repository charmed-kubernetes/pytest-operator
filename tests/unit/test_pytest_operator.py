import asyncio
import logging
from unittest.mock import Mock, AsyncMock, ANY, patch, call, MagicMock, PropertyMock
from urllib.error import HTTPError
from pathlib import Path
from types import SimpleNamespace
from websockets.exceptions import ConnectionClosed
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
        info_api.side_effect = HTTPError(
            url="", code=404, msg="", hdrs={}, fp=None  # type: ignore[arg-type]
        )
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
        info_api.side_effect = HTTPError(
            url="", code=404, msg="", hdrs={}, fp=None  # type: ignore[arg-type]
        )
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


async def test_plugin_build_resources(tmp_path_factory):
    ops_test = plugin.OpsTest(Mock(**{"module.__name__": "test"}), tmp_path_factory)
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


async def test_async_render_bundles(tmp_path_factory):
    ops_test = plugin.OpsTest(Mock(**{"module.__name__": "test"}), tmp_path_factory)
    ops_test.jujudata = Mock()
    ops_test._current_alias = "main"
    mock_model = ops_test._models[ops_test._current_alias] = Mock()
    mock_model.model_name = ops_test.default_model_name
    mock_model.tmp_path = None
    ops_test.jujudata.path = ""

    with pytest.raises(TypeError):
        await ops_test.async_render_bundles(1234)  # type: ignore[type-var]

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


async def test_crash_dump_mode(monkeypatch, tmp_path_factory):
    """Test running juju-crashdump in OpsTest.cleanup."""
    patch = monkeypatch.setattr
    patch(plugin.OpsTest, "run", mock_run := AsyncMock(return_value=(0, "", "")))
    ops_test = plugin.OpsTest(
        mock_request := Mock(**{"module.__name__": "test"}), tmp_path_factory
    )
    ops_test.crash_dump = True
    model = MagicMock()
    model.machines.values.return_value = []
    model.disconnect = AsyncMock()
    model.block_until = AsyncMock()
    ops_test._init_keep_model = None
    ops_test._current_alias = "main"
    ops_test._models = {
        ops_test._current_alias: plugin.ModelState(
            model, False, "test", "local", "model"
        )
    }
    ops_test.crash_dump_output = None
    patch(ops_test, "log_model", AsyncMock())
    ops_test._controller = AsyncMock()

    # 0 tests failed
    mock_request.session.testsfailed = 0

    await ops_test._cleanup_model()

    mock_run.assert_not_called()
    mock_run.reset_mock()

    # 1 tests failed
    ops_test._current_alias = "main"
    ops_test._models = {
        ops_test._current_alias: plugin.ModelState(
            model, False, "test", "local", "model"
        )
    }
    mock_request.session.testsfailed = 1

    await ops_test._cleanup_model()

    mock_run.assert_called_once_with(
        "juju-crashdump",
        "-s",
        "-m",
        "test:model",
        "-a",
        "debug-layer",
        "-a",
        "config",
    )
    mock_run.reset_mock()


async def test_create_crash_dump(monkeypatch, tmp_path_factory):
    """Test running create crash dump."""

    async def mock_run(*cmd):
        proc = await asyncio.create_subprocess_exec("not-valid-command")
        await proc.communicate()

    patch = monkeypatch.setattr
    patch(plugin.OpsTest, "run", mock_run)
    patch(plugin.OpsTest, "model_full_name", "admin:cloud/main")
    patch(plugin, "log", mock_log := MagicMock())
    ops_test = plugin.OpsTest(Mock(**{"module.__name__": "test"}), tmp_path_factory)
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


@pytest.fixture
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
    yield mock_request


@pytest.mark.parametrize(
    "attr", ["model", "model_name", "model_full_name", "model_config", "cloud_name"]
)
async def test_fixture_no_existing_model(
    mock_juju, setup_request, tmp_path_factory, attr
):
    setup_request.config.option.model = "this-model"
    ops_test = plugin.OpsTest(setup_request, tmp_path_factory)
    with pytest.raises(plugin.ModelNotFoundError, match="No model currently selected"):
        _ = getattr(ops_test, attr)

    ops_test._current_alias = "bogus"
    with pytest.raises(plugin.ModelNotFoundError, match="model 'bogus' not found"):
        _ = getattr(ops_test, attr)


async def test_fixture_set_up_existing_model(
    mock_juju, setup_request, tmp_path_factory
):
    setup_request.config.option.model = "this-model"
    ops_test = plugin.OpsTest(setup_request, tmp_path_factory)
    with pytest.raises(plugin.ModelNotFoundError):
        _ = ops_test.model

    await ops_test._setup_model()
    mock_juju.model.connect.assert_called_with("this-controller:this-model")
    assert ops_test.model == mock_juju.model
    assert ops_test.model_full_name == "this-controller:this-model"
    assert ops_test.cloud_name is None
    assert ops_test.model_name == "this-model"
    assert ops_test.keep_model is True, "Model should be kept if it already exists"
    assert len(ops_test.models) == 1


@patch("pytest_operator.plugin.OpsTest.forget_model")
@patch("pytest_operator.plugin.OpsTest.run")
async def test_fixture_cleanup_multi_model(
    mock_run, mock_forget_model, mock_juju, setup_request, tmp_path_factory
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


@patch("pytest_operator.plugin.OpsTest.default_model_name", new_callable=PropertyMock)
@patch("pytest_operator.plugin.OpsTest.juju", autospec=True)
async def test_fixture_set_up_automatic_model(
    juju_cmd, mock_default_model_name, mock_juju, setup_request, tmp_path_factory
):
    model_name = "this-auto-generated-model-name"
    mock_default_model_name.return_value = model_name
    ops_test = plugin.OpsTest(setup_request, tmp_path_factory)
    with pytest.raises(plugin.ModelNotFoundError):
        _ = ops_test.model

    await ops_test._setup_model()
    mock_juju.controller.add_model.assert_called_with(
        model_name, "this-cloud", config=None
    )
    juju_cmd.assert_called_with(ops_test, "models")
    assert ops_test.model == mock_juju.model
    assert ops_test.model_full_name == f"this-controller:{model_name}"
    assert ops_test.cloud_name == "this-cloud"
    assert ops_test.model_name == model_name
    # Don't keep a model if it's automatically generated
    assert ops_test.keep_model is False
    assert len(ops_test.models) == 1


class MockConnectionClosed(ConnectionClosed):
    """defines new type of ConnectionClosed exception.

    websockets.ConnectionClosed takes different forms
    in python 3.8, 3.9 and beyond and doesn't yield well to
    static-analysis. This exception can be raised in its stead
    by the tests but is still treated in the plugin as a
    ConnectionClosed typed exception.
    """

    def __init__(self, *_args, **_kwds) -> None:
        # intentionally don't construct the underlying exception
        pass


@pytest.mark.parametrize("model_name", [None, "alt-model"])
@pytest.mark.parametrize("cloud_name", [None, "alt-cloud"])
@pytest.mark.parametrize(
    "block_exception",
    [None, asyncio.TimeoutError(), MockConnectionClosed()],
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
