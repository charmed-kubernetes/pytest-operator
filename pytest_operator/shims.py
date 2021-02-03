import asyncio
import builtins
import inspect
import unittest


if not hasattr(builtins, "breakpoint"):
    # Shim breakpoint() builtin from PEP-0553 prior to 3.7
    def _breakpoint():
        import ipdb as ipdb

        ipdb.set_trace(inspect.currentframe().f_back)

    builtins.breakpoint = _breakpoint


if not hasattr(asyncio, "all_tasks"):
    # Shim top-level all_tasks (moved in 3.7)
    asyncio.all_tasks = asyncio.Task.all_tasks


IsolatedAsyncioTestCase = getattr(unittest, "IsolatedAsyncioTestCase", None)
if not IsolatedAsyncioTestCase:
    # Shim IsolatedAsyncioTestCase using asynctest prior to 3.8
    import asynctest

    class IsolatedAsyncioTestCase(asynctest.TestCase):
        async def setUp(self):
            await self.asyncSetUp()

        async def tearDown(self):
            await self.asyncTearDown()
