# Reference

## Command-line Parameters

### `--cloud`

Juju cloud to use. If not provided, it will use the default for the controller.

### `--controller`

Juju controller to use. If not provided, it will use the current controller.

### `--model`

Juju model to use. If not provided, a new model will be created for each test module.
All tests within a module will share the same model.

### `--keep-models`

Keep any automatically created models.


### `--model-config`

Path to a yaml file which will be applied to the model on creation.

 * ignored if `--model` supplied 
 * if the specified file doesn't exist, an error will be raised.

## Fixtures

### `ops_test`

This is the primary interface for the plugin, and provides an instance of the [`OpsTest`
class](#OpsTest).


### `tmp_path_factory`

This overrides the default `tmp_path_factory` fixture from pytest to relocate any
temporary directories to under `$TOX_ENV_DIR/tmp/pytest`. This is done because strictly
confined snaps, like `charmcraft`, can't access the global `/tmp`.


### `event_loop`

This overrides the default `event_loop` fixture from pytest-asyncio so that the same
event loop is used for all tests within a given test module (that is, it has `module`
scope rather than `function` scope).


## Markers

### `@pytest.mark.abort_on_fail`

If the test marked with this fails, it will cause any subsequent test in the module to
automatically `xfail`. This should be used when a given test is a requirement for any
following tests to have any possibility of succeeding, such as if the test does the
initial build and deploy.


### `@pytest.mark.asyncio`

This marker from pytest-asyncio is automatically applied to any `async` test function
which is collected, so that you don't have to decorate every single one of your tests
with it.


## Warning Filters

The asyncio library changed how the event loop is managed and the explicit parameter to
a lot of functions is now deprecated, but python-libjuju hasn't been updated to address
these (see [issue #461](https://github.com/juju/python-libjuju/issues/461)). This
generates an extreme amount of noise on every test run and makes it hard to see the test
results or reason for failure, so these warnings are automatically ignored.


## `OpsTest`

### Attributes

#### `model`

The python-libjuju `Model` instance for the test.

#### `model_full_name`

The fully qualified model name, including the controller.

#### `tmp_path`

A `pathlib.Path()` instance to an automatically created temporary directory for the
test.

#### `cloud_name`

The name of the cloud provided via the `--cloud` command-line parameter. If `None`, the
default cloud for the controller.

#### `controller_name`

The name of the controller being used.

#### `model_name`

The name of the model being used, whether it was automatically generated or provided by
the `--model` command-line parameter.

### Methods

#### `async def build_charm(self, charm_path)`

Builds a charm.

This can handle charms using the older charms.reactive framework as well as charms
written against the modern operator framework.

Returns a `pathlib.Path` for the built charm file.

#### `async def build_charms(self, *charm_paths)`

A helper which builds multiple charms at once, in parallel.

Returns a list of `pathlib.Path` instances for each charm, in the same order as the
args.

#### `async def build_bundle(self, bundle, output_bundle, destructive_mode, serial)`

A helper which builds an entire bundle at once.

Uses [juju-bundle][] to do the heavy lifting. Unlike `build_charm`, does not return a `pathlib.Path`
for the resulting bundle.yaml, as the default is `built-bundle.yaml`, and can be overridden with the
`output_bundle` argument.

#### `async def deploy_bundle(self, bundle, build, destructive_mode, serial, extra_args)`

A helper which deploys an entire bundle at once.

Uses [juju-bundle][] to do the heavy lifting. Defaults to `build=True`, as it is most commonly used
for building and deploying local bundles. For deploying straight from the charm store,
`ops_test.model.deploy()` is preferred.

#### `async def build_lib(self, lib_path)`

Build a Python library (sdist), which can then be injected into a test charm's
`wheelhouse.txt` using `render_charm()`.

Returns a `pathlib.Path` for the built library archive file.

#### `def render_charm(self, charm_path, include=None, exclude=None, context=None, **kwcontext)`

Render a templated charm using Jinja2.

This can be used to make certain files in a test charm templated, such
as a path to a library file that is built locally.

The `charm_path` param is the path (string or `pathlib.Path` object) to the charm to
build. Often, this is just `"."` to build the charm under test.

The `include` and `exclude` params can be lists of glob patterns or file paths used to
filter which files are passed through to Jinja2. Any files under the base `charm_path`
which are specifically excluded or which don't match the `include` list will be copied
over entirely unchanged.

The `context` param is the context for the template, and context can also be passed in
as keyword args.

Returns a `pathlib.Path` for the rendered charm source directory.

#### `def render_charms(self, *charm_paths, include=None, exclude=None, context=None, **kwcontext)`

A helper which renders multiple charms at once.

Returns a list of `pathlib.Path` instances for each charm, in the same order as the
args.

#### `def render_bundle(self, bundle, context=None, **kwcontext)`

Render a templated bundle using Jinja2.

This can be used to populate built charm paths or config values.

The `bundle` param can be either the path to the bundle file or a YAML string.

The `context` param is the context for the template, and context can also be passed in
as keyword args.

Returns the `pathlib.Path` for the rendered bundle file.

#### `def render_bundles(self, *bundles, context=None, **kwcontext)`

A helper which renders multiple bundles at once.

Returns a list of `pathlib.Path` instances for each bundle, in the same order as the
args.

#### `async def run(self, *cmd, cwd=None, check=False, fail_msg=None)`

Asynchronously run a subprocess command.

If `check` is False, returns a tuple of the return code, stdout, and stderr (decoded as
utf8). Otherwise, calls `pytest.fail` with `fail_msg` (if given) and relevant command
info.

#### `async def juju(self, *args)`

Runs a Juju CLI command.

Useful for cases where python-libjuju sees things differently than the Juju CLI. Will set
`JUJU_MODEL`, so manually passing in `-m model-name` is unnecessary.


#### `def abort(self, *args, **kwargs)`

Fail the current test method and mark all remaining test methods as xfail.

This can be used if a given step is required for subsequent steps to be successful, such
as the initial deployment.

Any args will be passed through to `pytest.fail()`.

You can also mark a test with `@pytest.marks.abort_on_fail` to have this automatically
applied if the marked test method fails or errors.

#### `async def log_model(self)`

Log a summary of the status of the model. This is automatically called before the model
is cleaned up.

[juju-bundle]: https://snapcraft.io/juju-bundle
