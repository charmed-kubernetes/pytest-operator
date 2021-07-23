<a name="pytest_operator.plugin"></a>

# pytest\_operator.plugin

<a name="pytest_operator.plugin.event_loop"></a>

#### event\_loop

```python
@pytest.fixture(scope="module")
event_loop()
```

Create an instance of the default event loop for each test module.

<a name="pytest_operator.plugin.pytest_collection_modifyitems"></a>

#### pytest\_collection\_modifyitems

```python
pytest_collection_modifyitems(session, config, items)
```

Automatically apply the "asyncio" marker to any async test items.

<a name="pytest_operator.plugin.pytest_runtest_makereport"></a>

#### pytest\_runtest\_makereport

```python
@pytest.hookimpl(tryfirst=True, hookwrapper=True)
pytest_runtest_makereport(item, call)
```

Make test results available to fixture finalizers.

<a name="pytest_operator.plugin.OpsTest"></a>

## OpsTest Objects

```python
class OpsTest()
```

Utility class for testing Operator Charms.

<a name="pytest_operator.plugin.OpsTest.run"></a>

#### run

```python
 | async run(*cmd, *, cwd=None, check=False, fail_msg=None)
```

Asynchronously run a subprocess command.

If `check` is False, returns a tuple of the return code, stdout, and
stderr (decoded as utf8). Otherwise, calls `pytest.fail` with
`fail_msg` and relevant command info.

<a name="pytest_operator.plugin.OpsTest.log_model"></a>

#### log\_model

```python
 | async log_model()
```

Log a summary of the status of the model.

<a name="pytest_operator.plugin.OpsTest.abort"></a>

#### abort

```python
 | abort(*args, **kwargs)
```

Fail the current test method and mark all remaining test methods as xfail.

This can be used if a given step is required for subsequent steps to be
successful, such as the initial deployment.

Any args will be passed through to `pytest.fail()`.

You can also mark a test with `@pytest.marks.abort_on_fail` to have this
automatically applied if the marked test method fails or errors.

<a name="pytest_operator.plugin.OpsTest.build_charm"></a>

#### build\_charm

```python
 | async build_charm(charm_path)
```

Builds a single charm.

This can handle charms using the older charms.reactive framework as
well as charms written against the modern operator framework.

Returns a Path for the built charm file.

<a name="pytest_operator.plugin.OpsTest.build_charms"></a>

#### build\_charms

```python
 | async build_charms(*charm_paths)
```

Builds one or more charms in parallel.

This can handle charms using the older charms.reactive framework as
well as charms written against the modern operator framework.

Returns a mapping of charm names to Paths for the built charm files.

<a name="pytest_operator.plugin.OpsTest.render_bundle"></a>

#### render\_bundle

```python
 | render_bundle(bundle, context=None, **kwcontext)
```

Render a templated bundle using Jinja2.

This can be used to populate built charm paths or config values.

:param bundle (str or Path): Path to bundle file or YAML content.
:param context (dict): Optional context mapping.
:param **kwcontext: Additional optional context as keyword args.

Returns the Path for the rendered bundle.

<a name="pytest_operator.plugin.OpsTest.render_bundles"></a>

#### render\_bundles

```python
 | render_bundles(*bundles, *, context=None, **kwcontext)
```

Render one or more templated bundles using Jinja2.

This can be used to populate built charm paths or config values.

:param *bundles (str or Path): One or more bundle Paths or YAML contents.
:param context (dict): Optional context mapping.
:param **kwcontext: Additional optional context as keyword args.

Returns a list of Paths for the rendered bundles.

<a name="pytest_operator.plugin.OpsTest.build_lib"></a>

#### build\_lib

```python
 | async build_lib(lib_path)
```

Build a Python library (sdist) for use in a test.

Returns a Path for the built library archive file.

<a name="pytest_operator.plugin.OpsTest.render_charm"></a>

#### render\_charm

```python
 | render_charm(charm_path, include=None, exclude=None, context=None, **kwcontext)
```

Render a templated charm using Jinja2.

This can be used to make certain files in a test charm templated, such
as a path to a library file that is built locally.

:param charm_path (str): Path to top-level directory of charm to render.
:include (list[str or Path]): Optional list of glob patterns or file paths
    to pass through Jinja2, relative to base charm path. (default: all files
    are passed through Jinja2)
:exclude (list[str or Path]): Optional list of glob patterns or file paths
    to exclude from passing through Jinja2, relative to the base charm path.
    (default: all files are passed through Jinja2)
:param context (dict): Optional context mapping.
:param **kwcontext: Additional optional context as keyword args.

Returns a Path for the rendered charm source directory.

<a name="pytest_operator.plugin.OpsTest.render_charms"></a>

#### render\_charms

```python
 | render_charms(*charm_paths, *, include=None, exclude=None, context=None, **kwcontext)
```

Render one or more templated charms using Jinja2.

This can be used to make certain files in a test charm templated, such
as a path to a library file that is built locally.

:param *charm_paths (str): Path to top-level directory of charm to render.
:include (list[str or Path]): Optional list of glob patterns or file paths
    to pass through Jinja2, relative to base charm path. (default: all files
    are passed through Jinja2)
:exclude (list[str or Path]): Optional list of glob patterns or file paths
    to exclude from passing through Jinja2, relative to the base charm path.
    (default: all files are passed through Jinja2)
:param context (dict): Optional context mapping.
:param **kwcontext: Additional optional context as keyword args.

Returns a list of Paths for the rendered charm source directories.

