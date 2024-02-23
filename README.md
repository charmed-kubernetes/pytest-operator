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
    assert ops_test.model.applications["my-charm"].units[0].workload_status == "active"
```

## Building/Downloading Charm Resources
Quite often, when charms are preparing for integration tests, the charms may
need to attach resources to the charm for it to function. In these cases, 
the integration code must either build the resources or pull those from external resources.

Example:

```python
async def test_build_and_deploy(ops_test):
    charm = await ops_test.build_charm(".")

    build_script = Path.cwd() / "build-charm-resources.sh"
    resources = await ops_test.build_resources(build_script)

    if resources:
        # created a dict from list of a filenames
        resources = {rsc.stem: rsc for rsc in resources}
    else:
        arch_resources = ops_test.arch_specific_resources(charm)
        resources = await ops_test.download_resources(
            charm, resources=arch_resources
        )
        
    assert resources, "Failed to build or download charm resources."
    
    log.info("Build Bundle...")
    bundle = ops_test.render_bundle(
        "tests/data/bundle.yaml", charm=charm, **resources
    )

    log.info("Deploy Bundle...")
    juju_cmd = ["deploy", "-m", ops_test.model_full_name, str(bundle)]
    rc, stdout, stderr = await ops_test.juju(*juju_cmd)
    assert rc == 0, f"Bundle deploy failed: {(stderr or stdout).strip()}"

    await ops_test.model.wait_for_idle()
    ...
```



## Reference

More details can be found in [the reference docs](docs/reference.md).
