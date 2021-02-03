import asyncio

from pytest_operator import OperatorTest


class PluginTest(OperatorTest):
    async def asyncSetUp(self):
        self.charms = await self.build_charms(
            "tests/data/charms/reactive-framework",
            "tests/data/charms/operator-framework",
        )
        await super().asyncSetUp()

    async def test_build_and_deploy(self):
        await asyncio.gather(*(self.model.deploy(charm) for charm in self.charms))
        await self.model.block_until(
            lambda: len(self.model.units) == 2
            and all(
                unit.workload_status == "active" for unit in self.model.units.values()
            ),
            timeout=5 * 60,
        )
        for unit in self.model.units.values():
            assert f"{unit.name}: {unit.workload_status}".endswith("active")
