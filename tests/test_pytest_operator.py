from pytest_operator import OperatorTest


class PluginTest(OperatorTest):

    async def test_build_and_deploy(self):
        bundle = self.render_bundle(
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
            charms=await self.build_charms(
                "tests/data/charms/reactive-framework",
                "tests/data/charms/operator-framework",
            ),
        )
        await self.model.deploy(bundle)
        await self.wait_for_bundle(bundle)  # TODO: Be up-ported to libjuju
        for unit in self.model.units.values():
            assert f"{unit.name}: {unit.workload_status}".endswith("active")
