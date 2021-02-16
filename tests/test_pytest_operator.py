import logging
from pathlib import Path

from pytest_operator import OperatorTest


log = logging.getLogger(__name__)


class PluginTest(OperatorTest):
    async def test_build_and_deploy(self):
        lib_path = Path(__file__).parent.parent
        pytest_operator = await self.build_lib(lib_path)
        charms = self.render_charms(
            "tests/data/charms/reactive-framework",
            "tests/data/charms/operator-framework",
            include=["requirements.txt"],
            context={
                "pytest_operator": pytest_operator,
            },
        )
        req_path = charms[1] / "requirements.txt"
        assert f"file://{pytest_operator}#egg=pytest_operator" in req_path.read_text()
        bundle = self.render_bundle(
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
            charms=await self.build_charms(*charms),
        )
        log.info("Deploying bundle")
        await self.model.deploy(bundle)
        await self.model.wait_for_idle()
        for unit in self.model.units.values():
            assert f"{unit.name}: {unit.workload_status}".endswith("active")

    async def test_shared_model_and_test_order(self):
        assert self.model.applications.keys() == {
            "reactive-framework",
            "operator-framework",
        }
