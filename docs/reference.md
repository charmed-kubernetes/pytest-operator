# Reference

## Command-line Parameters

### `--cloud`

Juju cloud to use. If not provided, it will use the default for the controller.

### `--controller`

Juju controller to use. If not provided, it will use the current controller.

### `--model`

Juju model to use. 

If not provided, a new model will be created for each test module file.
- All tests within a module will share the same model.
- The model will be destroyed at the end of the test module's scope

If provided, `ops_test` will attempt to use an existing model by this name
on the specified controller.  
* If that model does exist, it will be reused
* If that model doesn't exist, a new model by this name will be created
* The model will not be destroyed at the end of the test module's scope

### `--keep-models`

Keep any automatically created models.

### `--destroy-storage`

Destroy storage allocated on the created models.

### `--model-config`

Path to a yaml file which will be applied to the model on creation.

 * ignored if `--model` supplied
 * if the specified file doesn't exist, an error will be raised.

### `--model-alias`

Alias of the first model tracked by `ops_test`.  This alias in no way relates to the
name of the model as known by juju.  For that see `--model`.
* if not provided, the first created model alias is "main"
* if provided the tests may reference the model by a different alias.

### `--no-deploy`

Flag that guarantees skipping the function marked with `skip_if_deployed`. The skip will
only work if the `--model` parameter is also provided.

### `--no-crash-dump`

(Deprecated - use '--crash-dump=never' instead.  Overrides anything specified in 
'--crash-dump') This flag disables the automatic execution of `juju-crashdump`, 
which runs by default (if a command is available) after failed tests.

### `--crash-dump`

Sets whether to output a juju-crashdump after tests.  Options are:
* always: dumps after all tests
* on-failure: dumps after failed tests
* legacy: (DEFAULT) dumps after a failed test if '--keep-models' is False
* never: never dumps

### `--crash-dump-args`

Sets extra crashdump arguments if running crashdump
* default arguments are not revokable
* arguments will appear after the default arguments
* arguments are shlex delimited


### `--crash-dump-output`

Path to the directory where the `juju-crashdump` output will be stored. The default is
the current working directory.


### `--juju-max-frame-size`

Maximum frame size to use when connecting to a juju model. The default is None

### `--charmcraft-args`

Extra set of arguments to pass to charmcraft when packing the charm
* can be invoked multiple times to pass multiple arguments
* example)
```
pytest ... \
   --charmcraft-args '--platform=my-platform' \
   --charmcraft-args '--verbosity=debug'
```

## Fixtures

### `ops_test`

This is the primary interface for the plugin, and provides an instance of the [`OpsTest`
class](#OpsTest).


### `basetemp`

Some snap tools are dropping their `classic` snap support, and will
lose the ability to write anywhere on the filesystem. Tests should
be run to confirm they're located within the user's `HOME` directory
so strictly confined snaps can write to temporary directories.

Temp Directories can be moved with the following options:

If `basetemp` is provided as pytest configuration
   * pytest will create a directory here for temporary files
If `basetemp` is not provided as pytest configuration
   * the plugin will look to `TOX_ENV_DIR` environment variable
   * if that env var is set, `${tox_env_dir}/tmp/pytest` will be used
If `basetemp` and `TOX_ENV_DIR` are both unset
   * pytest is responsible for creating a temporary directory


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

### `@pytest.mark.skip_if_deployed`

This marker should be used for test_build_and_deploy functions, ie functions that have
the job of building a charm and then deploying it alone or in a bundle. It will ensure
that the function can be skipped using the `--no-deploy` parameter, which will help the
developer to run integration tests multiple times.

---
**NOTE**

Using the `skip_if_deployed` and` --no-deploy` parameters will not ensure build and
subsequent refresh of the charm.

---


## `OpsTest`

### Attributes

#### `ModelKeep`

Enum used to select the appropriate behavior for cleaning up models created or used by ops_test.

* `NEVER`

  This gives pytest-operator the duty to delete this model
  at the end of the test regardless of any outcome.
* `ALWAYS`

  This gives pytest-operator the duty to keep this model
  at the end of the test regardless of any outcome.
* `IF_EXISTS`

  If the model already exists before opstest encounters it,
  follow the rules defined by `track_model.use_existing`
  * respects the --keep-models flag, otherwise
  * newly created models mapped to ModelKeep.NEVER
  * existing models mapped to ModelKeep.ALWAYS

Example Usage:
  ```python
  # create a model that will always be removed
  # Regardless of the --keep-models flag
  await ops_test.track_model(
    "secondary",
    keep=ops_test.ModelKeep.ALWAYS
  )
  ```

#### `model`

The current aliased python-libjuju `Model` instance for the test.

#### `model_full_name`

The fully qualified model name, including the controller of the current aliased model.

#### `tmp_path`

A `pathlib.Path()` instance to an automatically created temporary directory for the
test of the current aliased model.

#### `cloud_name`

The name of the cloud provided via the `--cloud` command-line parameter. If `None`, the
default cloud for the controller of the current aliased model.

#### `controller_name`

The name of the controller being used.

#### `model_name`

The name of the juju model referenced by the current aliased model.
If the alias is set as the first model, that name will reflect its automatically generated
name or the name provided by the `--model` command-line parameter.

#### `Bundle`

Dataclass which represents a juju bundle.

```python
  bundle = ops_test.Bundle("charmed-kubernetes", "latest/edge")
```

### Methods

#### `async def build_charm(self, charm_path, bases_index = None, verbosity = None, return_all = False)`

Builds a charm.

This can handle charms using the older charms.reactive framework as well as charms
written against the modern operator framework.

Returns a `pathlib.Path` for the built charm file if `return_all` is `False`
Returns a `List[pathlib.Path]` for the built charms if `return_all` is `True`
Raises a `FileNotFound` exception if no charms are found after a successful build.
Raises a `RuntimeError` exception if the charm build fails.


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

#### `async def async_render_bundles(self, *bundles: BundleOpt, **context) -> List[Path]:`

A helper which renders a set of templated bundles using Jinja2.

Returns a list of `pathlib.Path` instances for each bundle, in the same order as the
overlays.

provide as many bundles as necessary from any of the follow types:
  * `str`
    * can be YAML content
    * can be a `str` that is `os.pathlike` and ends with an extension of `yaml`, `yml`, or even `.yaml.j2`
  * `Path`
    * Path to any text based file which can be loaded by `yaml.safe_load(..)`
  * `OpsTest.Bundle`
    * bundles can be downloaded from charmhub using a Bundle object.
      The bundle is downloaded, unpacked, and its `bundle.yaml` file is used as the content.

      See [ops_test.Bundle](#Bundle)


#### `async def run(self, *cmd: str, cwd: Optional[os.PathLike] = None, check: bool = False, fail_msg: Optional[str] = None, stdin: Optional[bytes] = None)`

Asynchronously run a subprocess command.

If `check` is False, returns a tuple of the return code, stdout, and stderr (decoded as
utf8). Otherwise, calls `pytest.fail` with `fail_msg` (if given) and relevant command
info.

#### `async def juju(self, *args, **kwargs)`

Runs a Juju CLI command.

Useful for cases where python-libjuju sees things differently than the Juju CLI. Will set
`JUJU_MODEL`, so manually passing in `-m model-name` is unnecessary.

#### `async def fast_forward(self, fast_interval: str = "10s", slow_interval: Optional[str] = None)`

Temporarily speed up update-status firing rate for the current model.
Returns an async context manager that temporarily sets update-status firing rate to `fast_interval`.
If provided, when the context exits the update-status firing rate will be set to `slow_interval`. Otherwise, it will be set to the previous value.

* If `fast_interval` is provided, the update-status firing rate will be set to that value upon entering the context. Default is 10s.
* If `slow_interval` is provided, after the context exits the update-status firing rate will be set to that value; otherwise, to the value it had before the context was entered.

It is effectively a shortcut for:
```python
    await ops_test.model.set_config({"update-status-hook-interval": <fast-interval>})

    # do something

    await ops_test.model.set_config({"update-status-hook-interval": <slow-interval>})
```

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

#### `async def track_model(self, alias: str, model_name: Optional[str] = None, cloud_name: Optional[str] = None, use_existing: Optional[bool] = None, keep: Optional[ModelKeep, str, bool, None] = ModelKeep.IF_EXISTS, destroy_storage: bool = False, **kwargs,) -> Model`

Indicate to `ops_test` to track a new model which is automatically created in juju or an existing juju model referenced by model_name.
This allows `ops_test` to track multiple models on various clouds by a unique alias name.

##### Key parameters:
* `alias`: Required `str` which defines the alias of the model used only by `ops_test`
* `model_name`: Name of the model referenced by juju -- automatically generated if `None`
* `cloud_name`: Name of the juju cloud where the model resides -- reuse current cloud if `None`
* `use_existing`:
  * `None` (default): `ops_test` will re-use an existing model-name if provided, otherwise False
  * `False`: `ops_test` creates a new model
  * `True`: `ops_test` won't create a new model, but will connect to an existing model by `model_name`
* `keep`:
  * `ModelKeep`  : See docs for the `ModelKeep` enum
  * `str`        : mapped to `ModelKeep` values
  * `None`       : Same as `ModelKeep.IF_EXISTS`
  * `True`       : Same as `ModelKeep.ALWAYS`
  * `False`      : Same as `ModelKeep.NEVER`, but respects `--keep-models` flag

  * `None` (default): inherit boolean value of `use_existing`
  * `False`: `ops_test` will destroy at the end of testing
  * `True`: `ops_test` won't destroy at the end of testing
* `destroy_storage`: Wether the storage should be destroyed with the model, defaults to `False`.

##### Examples:

```python
# make a new model with any juju name and destroy it when the tests are over
await ops_test.track_model("alias")

# make a new model with any juju name but don't destroy it when the tests are over
await ops_test.track_model("alias", keep=True)

# Invalid, can't reuse an existing model when the model_name isn't provided
await ops_test.track_model("alias", use_existing=True)

# connect to an existing model known to juju as "bob" and keep it when the tests are over
await ops_test.track_model("alias", model_name="bob", use_existing=True)

# connect to an existing model known to juju as "bob" but destroy it when the tests are over
await ops_test.track_model("alias", model_name="bob", use_existing=True, keep=False)

# make a new model known to juju as "bob" and destroy it when the tests are over
await ops_test.track_model("alias", model_name="bob", use_existing=False)

# make a new model known to juju as "bob" but don't destroy it when the tests are over
await ops_test.track_model("alias", model_name="bob", use_existing=False, keep=False)

"""
the following examples where `model_name` is provided, will set `use_existing` implicitly
depending on its existence in the controller.
"""
# make or reuse a model known to juju as "bob"
# don't destroy model if it existed, destroy it if it didn't already exist
await ops_test.track_model("alias", model_name="bob")

# make or reuse a model known to juju as "bob" but don't destroy it when the tests are over
await ops_test.track_model("alias", model_name="bob", keep=True)
```

#### `async def forget_model(self, alias: Optional[str] = None, timeout: Optional[Union[float, int]] = None)`

Indicate to `ops_test` to forget an existing model and `destroy` that model except under the following circumstances.

* If `--keep-models` was passed as a tox argument, no models will be destroyed.
* If `--model=<specific model>` was passed as a tox argument, this specific model will not be destroyed.

A Timeout Exception will be raised if a `timeout` value is specified and the model isn't destroyed
within that number of seconds.

it's possible to determine if the model is a candidate for destroying using
```python
    assert ops_test._init_keep_model is False # this flag is set when the keep-models argument is passed to pytest
    assert ops_test.models["main"].keep is False  # by default, we forget and destroy models
    assert ops_test.keep_model is False # the current aliased model will be kept/destroyed based on this property
```


#### `def model_context(self, alias: str) -> Generator[Model, None, None]:`

The only way to switch between tracked models is by using this method to change
the context of the model to which the tests refer.

For example, assume there are two models being tracked by ops_test: "main" and "secondary"
The following test would `PASS` due to the nature of `model_context`'s function.

```python
def test_second_model(ops_test):
    assert ops_test.current_alias == "main"
    primary_model = ops_test.model
    with ops_test.model_context("secondary") as secondary_model:
        # now this is testing the secondary model
        assert ops_test.current_alias == "secondary"
        assert ops_test.model == secondary_model

    # now this is testing the main model
    assert secondary_model != primary_model
    assert ops_test.current_alias == "main"
    assert ops_test.model == primary_model
```

The following ops_test properties will change their values within a `model_context`
* `current_alias`
* `tmp_path`
* `model_config`
* `model`
* `model_full_name`
* `model_name`
* `cloud_name`
* `keep_model`

Once the model context closes, `ops_test` returns the current_alias back to the prior model's context
assuming that it references a model alias which hasn't been forgotten.

[juju-bundle]: https://snapcraft.io/juju-bundle


#### `def add_k8s(...)`

Supports adding a k8s cloud to the connected juju controller. This can be useful for testing cross-model relations between a machine model and a kubernetes models.

For example, assuming one has access to a `kubeconfig` file, use the kubernetes client to load that `kubeconfig` and pass to ops_test via the `add_k8s` method.  After this, we can create a new model on that k8s cloud and test integrations between models. 


```python
    kubeconfig_path = ops_test.tmp_path / "kubeconfig"
    kubeconfig_path.write_text(config)
    config = type.__call__(Configuration)
    k8s_config.load_config(client_configuration=config, config_file=str(kubeconfig_path))

    k8s_cloud = await ops_test.add_k8s(kubeconfig=config, skip_storage=False)
    k8s_model = await ops_test.track_model(
        "cos", cloud_name=k8s_cloud, keep=ops_test.ModelKeep.NEVER
    )
```
