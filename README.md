# pytest-operator

PyTest plugin to make it easy to create integration tests for Operator Charms.

## Usage

Include `pytest-operator` in the `deps` of your `tox.ini` file:

```ini
[testenv]
deps =
  pytest
  pytest-operator
```

Then, just start using the `ops_test` fixture in your async tests.  This
module-scoped fixture provides a libjuju Model, helpers to build charms for
testing, and the ability to abort testing so that the remaining tests in the
module are automatically xfailed (you can also mark a test so that this happens
automatically if the test fails; this is typically used on the initial deployment
test, where subsequent tests depend on the deployment having succeeded).

As an additional nicety, you don't have to explicitly mark an async test with
`@pytest.mark.asyncio`; if it's a test function / method and it's async, it
will be marked automatically.

Example:

```python
import pytest


@pytest.mark.abort_on_fail
async def test_build_and_deploy(ops_test):
    my_charm = await ops_test.build_charm(".")
    await ops_test.model.deploy(my_charm)
    await ops_test.model.wait_for_idle()


async def test_status(ops_test):
    assert ops_test.applications["my-charm"].units[0].workload_status == "active"
```
