import logging
import os
from pathlib import Path

import pytest


log = logging.getLogger(__name__)


class TestPlugin:
    @pytest.mark.abort_on_fail
    async def test_build_and_deploy(self, ops_test):
        lib_path = Path(__file__).parent.parent
        pytest_operator = await ops_test.build_lib(lib_path)
        charms = ops_test.render_charms(
            "tests/data/charms/reactive-framework",
            "tests/data/charms/operator-framework",
            include=["requirements.txt"],
            context={
                "pytest_operator": pytest_operator,
            },
        )
        req_path = charms[1] / "requirements.txt"
        req_text = req_path.read_text()
        assert f"file://{pytest_operator}#egg=pytest_operator" not in req_text
        bundle = ops_test.render_bundle(
            # Normally, this would just be a filename like for the charms, rather
            # than an in-line YAML dump, but for visibility purposes in using this
            # test as an example, I included it directly here, since it's small. E.g.:
            # "tests/data/bundle.yaml",
            """
                series: focal
                applications:
                  reactive-framework:
                    charm: {{ charms["reactive-framework"] }}
                    num_units: 1
                  operator-framework:
                    charm: {{ charms["operator-framework"] }}
                    num_units: 1
            """,
            charms=await ops_test.build_charms(*charms),
        )
        log.info("Deploying bundle")
        await ops_test.model.deploy(bundle)
        await ops_test.model.wait_for_idle()
        for unit in ops_test.model.units.values():
            assert f"{unit.name}: {unit.workload_status}".endswith("active")

    async def test_0_shared_model_and_test_order(self, ops_test):
        assert ops_test.model.applications.keys() == {
            "reactive-framework",
            "operator-framework",
        }


async def test_func(ops_test):
    assert ops_test.model


def test_tmp_path(ops_test):
    tox_env_dir = Path(os.environ["TOX_ENV_DIR"]).resolve()
    assert ops_test.tmp_path.relative_to(tox_env_dir)


async def test_run(ops_test):
    assert await ops_test.run("/bin/true") == (0, "", "")
    assert await ops_test.run("/bin/false") == (1, "", "")
    with pytest.raises(AssertionError) as exc_info:
        await ops_test.run("/bin/false", check=True)
    assert str(exc_info.value) == "Command ['/bin/false'] failed (1): "
    with pytest.raises(AssertionError) as exc_info:
        await ops_test.run("/bin/false", check=True, fail_msg="test")
    assert str(exc_info.value) == "test (1): "


@pytest.mark.abort_on_fail(abort_on_xfail=True)
def test_abort():
    pytest.xfail("abort")


def test_aborted():
    pytest.fail("Should be automatically xfailed due to abort")
