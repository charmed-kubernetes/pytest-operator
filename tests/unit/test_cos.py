# Copyright 2024 Canonical Ltd.
# See LICENSE file for licensing details.

from unittest.mock import AsyncMock, Mock, call, patch

import pytest
from juju.action import Action
from juju.application import Application
from juju.model import Model
from juju.relation import Endpoint, Relation
from juju.unit import Unit

from pytest_operator.cos import (
    _assert_grafana_agent_status,
    _check_metrics_endpoint,
    _get_alert_rules,
    _get_app_relation_data,
    _get_metrics_endpoint,
    _get_relation,
    _get_unit_relation_data,
    _run_on_unit,
    assert_alert_rules,
    assert_logging,
    assert_metrics_endpoint,
    deploy_and_assert_grafana_agent,
    get_alert_rules,
)

EXP_GRAFANA_AGENT_MSG = (
    "Missing ['grafana-cloud-config']|['grafana-dashboards-provider'] for "
    "grafana-dashboards-consumer; ['grafana-cloud-config']|['logging-consumer'] for "
    "logging-provider; ['grafana-cloud-config']|['send-remote-write'] for "
    "metrics-endpoint"
)
MY_ALERT_1 = """
groups:
- name: my-server
  rules:
  - alert: MyAlert1
    expr: rate(http_request_duration_seconds_sum[5m]) > 7
    for: 0m
    labels:
      severity: critical
    annotations:
      summary: MyServer server {{ $labels.juju_model }}/{{ $labels.juju_unit }} requests taking longer than expected
      description: >
        The MyServer server {{ $labels.juju_model }} {{ $labels.juju_unit }} requests taking longer than 7 seconds in the last 5 min window
        LABELS = {{ $labels }}
"""  # noqa E503
MY_ALERT_2 = """
alert: MyAlert2
expr: up < 1
for: 5m
labels:
  severity: critical
annotations:
  summary: MyServer unit {{ $labels.juju_model }}/{{ $labels.juju_unit }} unavailable
  description: >
    The MyServer unit {{ $labels.juju_model }} {{ $labels.juju_unit }} is unavailable
    LABELS = {{ $labels }}
"""  # noqa E503


@pytest.mark.asyncio
async def test_deploy_and_assert_grafana_agent_invalid_app():
    """Test deploy grafana-agent-k8s along not existing app."""
    app = "my-app"
    model = AsyncMock(spec_set=Model)
    model.applications = []

    with pytest.raises(AssertionError, match=f"application {app} was not found"):
        await deploy_and_assert_grafana_agent(model, app)


@pytest.mark.parametrize(
    "msg",
    [
        EXP_GRAFANA_AGENT_MSG,
        "Missing ['grafana-cloud-config']|['grafana-dashboards-provider'] for "
        "grafana-dashboards-consumer",
        "Missing ['grafana-cloud-config']|['logging-consumer'] for logging-provider",
        "Missing ['grafana-cloud-config']|['send-remote-write'] for metrics-endpoint",
    ],
)
def test_assert_grafana_agent_msg(msg):
    """Test all possible grafana-agent-k8s expected msg."""
    _assert_grafana_agent_status(msg)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs, exp_awaits",
    [
        (
            {"metrics": True, "dashboard": True, "logging": True},
            [
                call(
                    "my-app:grafana-dashboard",
                    "grafana-agent-k8s:grafana-dashboards-consumer",
                ),
                call("my-app:metrics-endpoint", "grafana-agent-k8s:metrics-endpoint"),
                call("my-app:logging", "grafana-agent-k8s:logging-provider"),
            ],
        ),
        (
            {"metrics": True, "dashboard": True, "logging": False},
            [
                call(
                    "my-app:grafana-dashboard",
                    "grafana-agent-k8s:grafana-dashboards-consumer",
                ),
                call("my-app:metrics-endpoint", "grafana-agent-k8s:metrics-endpoint"),
            ],
        ),
        (
            {"metrics": True, "dashboard": False, "logging": True},
            [
                call("my-app:metrics-endpoint", "grafana-agent-k8s:metrics-endpoint"),
                call("my-app:logging", "grafana-agent-k8s:logging-provider"),
            ],
        ),
        (
            {"metrics": True, "dashboard": False, "logging": False},
            [
                call("my-app:metrics-endpoint", "grafana-agent-k8s:metrics-endpoint"),
            ],
        ),
        ({"dashboard": False, "metrics": False, "logging": False}, []),
    ],
)
async def test_deploy_and_assert_grafana_agent(kwargs, exp_awaits):
    """Test deploy grafana-agent-k8s along test app."""
    app = Mock(spec_set=Application)()
    app.name = "my-app"

    grafana_aget_app = Mock(spec_set=Application)()
    grafana_aget_app.name = "grafana-agent-k8s"
    grafana_aget_unit = Mock(spec_set=Unit)()
    grafana_aget_unit.workload_status_message = EXP_GRAFANA_AGENT_MSG
    grafana_aget_app.units = [grafana_aget_unit]

    model = AsyncMock(spec_set=Model)
    model.applications = {app.name: app, grafana_aget_app.name: grafana_aget_app}

    await deploy_and_assert_grafana_agent(model, app.name, **kwargs)

    model.deploy.assert_awaited_once_with("grafana-agent-k8s", channel="latest/stable")
    model.integrate.assert_has_awaits(exp_awaits)
    model.wait_for_idle.assert_awaited_once_with(
        apps=["grafana-agent-k8s"], status="blocked", timeout=300
    )


@pytest.mark.asyncio
async def test_deploy_and_assert_grafana_agent_failed():
    """Test deploy grafana-agent-k8s failed reached expected state."""
    exp_error_msg = (
        "grafana-agent-k8s did not reach expected state. 'different message' != .*"
    )
    app = Mock(spec_set=Application)()
    app.name = "my-app"

    grafana_aget_app = Mock(spec_set=Application)()
    grafana_aget_app.name = "grafana-agent-k8s"
    grafana_aget_unit = Mock(spec_set=Unit)()
    grafana_aget_unit.workload_status_message = "different message"
    grafana_aget_app.units = [grafana_aget_unit]

    model = AsyncMock(spec_set=Model)
    model.applications = {app.name: app, grafana_aget_app.name: grafana_aget_app}

    with pytest.raises(AssertionError, match=exp_error_msg):
        await deploy_and_assert_grafana_agent(model, app.name)

    model.deploy.assert_awaited_once_with("grafana-agent-k8s", channel="latest/stable")


@pytest.mark.asyncio
async def test_get_relation_no_relations():
    """Test getting not existing relation."""
    app = Mock(spec_set=Application)()
    unit = Mock(spec_set=Unit)()
    app.name = "my-app"
    app.units = [unit]
    relation = Mock(spec_set=Relation)()
    endpoint = Mock(spec_set=Endpoint)()
    endpoint.name = "different-endpoint"
    relation.endpoints = [endpoint]
    app.relations = [relation]

    with pytest.raises(AssertionError, match="metrics-endpoint is missing"):
        await _get_relation(app, "metrics-endpoint")


@pytest.mark.asyncio
async def test_get_relation_too_many():
    """Test getting relation, when there is too many of them."""
    app = Mock(spec_set=Application)()
    unit = Mock(spec_set=Unit)()
    app.name = "my-app"
    app.units = [unit]
    relation = Mock(spec_set=Relation)()
    endpoint = Mock(spec_set=Endpoint)()
    endpoint.name = "metrics-endpoint"
    relation.endpoints = [endpoint]
    app.relations = [relation] * 3  # three relations

    with pytest.raises(
        AssertionError, match="too many relations with metrics-endpoint endpoint"
    ):
        await _get_relation(app, "metrics-endpoint")


@pytest.mark.asyncio
async def test_get_relation():
    """Test getting relation."""
    app = Mock(spec_set=Application)()
    unit = Mock(spec_set=Unit)()
    app.name = "my-app"
    app.units = [unit]
    relation = Mock(spec_set=Relation)()
    endpoint = Mock(spec_set=Endpoint)()
    endpoint.name = "metrics-endpoint"
    relation.endpoints = [endpoint]
    app.relations = [relation]

    assert await _get_relation(app, "metrics-endpoint") == relation


@pytest.mark.asyncio
async def test_get_app_relation_data_no_units():
    """Test getting application data from relation data bag without units."""
    app = Mock(spec_set=Application)()
    app.name = "my-app"
    app.units = []

    with pytest.raises(AssertionError, match="application my-app has no units"):
        await _get_app_relation_data(app, "metrics-endpoint")


@pytest.mark.asyncio
@patch("pytest_operator.cos._get_relation")
@patch("pytest_operator.cos._run_on_unit")
@patch("pytest_operator.cos.yaml")
async def test_get_app_relation_data(mock_yaml, mock_run_on_unit, mock_get_relation):
    """Test getting application data from relation data bag."""
    relation = Mock(spec_set=Relation)()
    relation.entity_id = relation_id = 7
    mock_get_relation.return_value = relation
    app = Mock(spec_set=Application)()
    unit = Mock(spec_set=Unit)()
    app.name = "my-app"
    app.units = [unit]
    mock_run_on_unit.return_value = result = Mock(spec_set=Action)()
    result.results = {"stdout": "test"}

    data = await _get_app_relation_data(app, "metrics-endpoint")

    mock_run_on_unit.assert_awaited_once_with(
        unit, f"relation-get --format=yaml -r {relation_id} --app - {app.name}"
    )
    mock_yaml.safe_load.assert_called_once_with("test")
    assert data == mock_yaml.safe_load.return_value


@pytest.mark.asyncio
@patch("pytest_operator.cos._get_relation")
@patch("pytest_operator.cos._run_on_unit")
@patch("pytest_operator.cos.yaml")
async def test_get_unit_relation_data(mock_yaml, mock_run_on_unit, mock_get_relation):
    """Test getting unit data from relation data bag."""
    relation = Mock(spec_set=Relation)()
    relation.entity_id = relation_id = 7
    mock_get_relation.return_value = relation
    app = Mock(spec_set=Application)()
    unit = Mock(spec_set=Unit)()
    unit.name = "my-app/0"
    app.name = "my-app"
    app.units = [unit]
    mock_run_on_unit.return_value = result = Mock(spec_set=Action)()
    result.results = {"stdout": "test"}

    data = await _get_unit_relation_data(app, "metrics-endpoint")

    mock_run_on_unit.assert_awaited_once_with(
        unit, f"relation-get --format=yaml -r {relation_id} - {unit.name}"
    )
    mock_yaml.safe_load.assert_called_once_with("test")
    assert data == {unit.name: mock_yaml.safe_load.return_value}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metrics_endpoint, exp_cmd",
    [
        ("*:5000/metrics", "curl -m 5 -sS http://my-app.my-model.svc:5000/metrics"),
        ("1.2.3.4:5000/metrics", "curl -m 5 -sS http://1.2.3.4:5000/metrics"),
    ],
)
@patch("pytest_operator.cos._run_on_unit")
async def test_check_metrics_endpoint(mock_run_on_unit, metrics_endpoint, exp_cmd):
    """Test check metrics endpoints."""
    grafana_agent_k8s_app = Mock(spec_set=Application)()
    unit = Mock(spec_set=Unit)()
    grafana_agent_k8s_app.units = [unit]

    app = Mock(spec_set=Application)()
    app.name = "my-app"
    app.model = AsyncMock(spec_set=Model)
    app.model.name = "my-model"
    app.model.applications = {"grafana-agent-k8s": grafana_agent_k8s_app, "my-app": app}

    await _check_metrics_endpoint(app, metrics_endpoint)

    mock_run_on_unit.assert_awaited_once_with(unit, exp_cmd)


@pytest.mark.parametrize(
    "data, exp_alert_rules",
    [
        ('{"groups": [{"rules": [{"alert": "my-alert"}]}]}', {"my-alert"}),
        ("alert: my-alert\nexpr: up < 1\nfor: 5m", {"my-alert"}),
        (
            "groups:\n- name: my-group1\n  rules:\n  - alert: my-alert1\n"
            "- name: my-group2\n  rules:\n  - alert: my-alert2",
            {"my-alert1", "my-alert2"},
        ),
    ],
)
def test_get_alert_rules_from_string(data, exp_alert_rules):
    """Test helper function to get alert rules from string."""
    assert _get_alert_rules(data) == exp_alert_rules


@pytest.mark.parametrize(
    "data, exp_metrics_endpoint",
    [
        (
            '[{"metrics_path": "/metrics", "static_configs": [{"targets": '
            '["*:5000","*:8000"]}]}]',
            {"*:5000/metrics", "*:8000/metrics"},
        ),
    ],
)
def test_get_metrics_endpoint(data, exp_metrics_endpoint):
    """Test helper function to get metrics endpoints from string."""
    assert _get_metrics_endpoint(data) == exp_metrics_endpoint


@pytest.mark.asyncio
async def test_run_on_unit_fail():
    """Test run cmd on unit with failure."""
    unit = Mock(spec_set=Unit)()
    unit.name = "my-app/0"
    unit.run = mock_result = AsyncMock(spec_set=Unit.run)
    mock_result.return_value.results = {"return-code": 1}

    with pytest.raises(AssertionError, match="cmd `test` failed with error `None`"):
        await _run_on_unit(unit, "test")


@pytest.mark.asyncio
async def test_run_on_unit():
    """Test run cmd on unit."""
    unit = Mock(spec_set=Unit)()
    unit.name = "my-app/0"
    unit.run = AsyncMock()
    unit.run.return_value.results = {"return-code": 0}

    result = await _run_on_unit(unit, "test")

    assert result == unit.run.return_value
    unit.run.assert_awaited_once_with("test", block=True)


@pytest.mark.asyncio
@patch("pytest_operator.cos._get_app_relation_data")
@patch("pytest_operator.cos._get_alert_rules")
async def test_assert_alert_rules_no_data(
    mock_get_alert_rules, mock_get_app_relation_data
):
    """Test assert function for alert rules with empty data bag."""
    app = Mock(spec_set=Application)()
    mock_get_app_relation_data.return_value = {}

    with pytest.raises(
        AssertionError, match="metrics-endpoint relation is missing 'alert_rules'"
    ):
        await assert_alert_rules(app, {})

    mock_get_app_relation_data.assert_awaited_once_with(app, "metrics-endpoint")
    mock_get_alert_rules.assert_not_called()


@pytest.mark.asyncio
@patch("pytest_operator.cos._get_app_relation_data")
@patch("pytest_operator.cos._get_alert_rules")
async def test_assert_alert_rules(mock_get_alert_rules, mock_get_app_relation_data):
    """Test assert function for alert rules."""
    app = Mock(spec_set=Application)()
    mock_get_app_relation_data.return_value = {"alert_rules": "..."}
    mock_get_alert_rules.return_value = exp_alert_rules = {"my-alert1", "my-alert2"}

    await assert_alert_rules(app, exp_alert_rules)

    mock_get_app_relation_data.assert_awaited_once_with(app, "metrics-endpoint")
    mock_get_alert_rules.assert_called_once_with("...")


@pytest.mark.asyncio
@patch("pytest_operator.cos._get_app_relation_data")
@patch("pytest_operator.cos._get_alert_rules")
async def test_assert_alert_rules_fail(
    mock_get_alert_rules, mock_get_app_relation_data
):
    """Test assert function for alert rules failing."""
    app = Mock(spec_set=Application)()
    mock_get_app_relation_data.return_value = {"alert_rules": "..."}
    mock_get_alert_rules.return_value = {"my-alert1", "my-alert2"}

    with pytest.raises(AssertionError):
        await assert_alert_rules(app, {"different-alert"})

    mock_get_app_relation_data.assert_awaited_once_with(app, "metrics-endpoint")
    mock_get_alert_rules.assert_called_once_with("...")


@pytest.mark.asyncio
@patch("pytest_operator.cos._get_app_relation_data")
@patch("pytest_operator.cos._get_metrics_endpoint")
@patch("pytest_operator.cos._check_metrics_endpoint")
async def test_assert_metrics_endpoint_empty_data_bag(
    mock_check_metrics_endpoint, mock_get_metrics_endpoint, mock_get_app_relation_data
):
    """Test assert function for metrics endpoint with empty data bag."""
    exp_error_msg = (
        "metrics-endpoint relation is missing 'scrape_jobs'\nassert 'scrape_jobs' in {}"
    )
    app = Mock(spec_set=Application)()
    mock_get_app_relation_data.return_value = {}

    with pytest.raises(AssertionError, match=exp_error_msg):
        await assert_metrics_endpoint(app, metrics_port=8000, metrics_path="/metrics")

    mock_get_app_relation_data.assert_awaited_once_with(app, "metrics-endpoint")
    mock_get_metrics_endpoint.assert_not_called()
    mock_check_metrics_endpoint.assert_not_awaited()


@pytest.mark.asyncio
@patch("pytest_operator.cos._get_app_relation_data")
@patch("pytest_operator.cos._get_metrics_endpoint")
@patch("pytest_operator.cos._check_metrics_endpoint")
async def test_assert_metrics_endpoint(
    mock_check_metrics_endpoint, mock_get_metrics_endpoint, mock_get_app_relation_data
):
    """Test assert function for metrics endpoint."""
    app = Mock(spec_set=Application)()
    mock_get_app_relation_data.return_value = {"scrape_jobs": "..."}
    mock_get_metrics_endpoint.return_value = {"*:5000/metrics"}

    await assert_metrics_endpoint(app, metrics_port=5000, metrics_path="/metrics")

    mock_get_app_relation_data.assert_awaited_once_with(app, "metrics-endpoint")
    mock_get_metrics_endpoint.assert_called_once_with("...")
    mock_check_metrics_endpoint.assert_awaited_once_with(app, "*:5000/metrics")


@pytest.mark.asyncio
@patch("pytest_operator.cos._get_app_relation_data")
@patch("pytest_operator.cos._get_metrics_endpoint")
@patch("pytest_operator.cos._check_metrics_endpoint")
async def test_assert_metrics_endpoint_fail(
    mock_check_metrics_endpoint, mock_get_metrics_endpoint, mock_get_app_relation_data
):
    """Test assert function for metrics endpoint failing."""
    app = Mock(spec_set=Application)()
    mock_get_app_relation_data.return_value = {"scrape_jobs": "..."}
    mock_get_metrics_endpoint.return_value = {"*:5000/metrics"}

    with pytest.raises(AssertionError):
        await assert_metrics_endpoint(app, metrics_port=8000, metrics_path="/metrics")

    mock_get_app_relation_data.assert_awaited_once_with(app, "metrics-endpoint")
    mock_get_metrics_endpoint.assert_called_once_with("...")
    mock_check_metrics_endpoint.assert_not_awaited()


@pytest.mark.asyncio
@patch("pytest_operator.cos._get_unit_relation_data")
async def test_assert_logging(mock_get_unit_relation_data):
    """Test assert function for logging endpoint."""
    app = Mock(spec_set=Application)()
    mock_get_unit_relation_data.return_value = {"my-app/0": {"endpoint": "..."}}

    await assert_logging(app)

    mock_get_unit_relation_data.assert_awaited_once_with(app, "logging")


@pytest.mark.asyncio
@patch("pytest_operator.cos._get_unit_relation_data")
async def test_assert_logging_fail(mock_get_unit_relation_data):
    """Test assert function for logging endpoint."""
    app = Mock(spec_set=Application)()
    mock_get_unit_relation_data.return_value = {"my-app/0": {}}

    with pytest.raises(AssertionError):
        await assert_logging(app)

    mock_get_unit_relation_data.assert_awaited_once_with(app, "logging")


def test_get_alert_rules(tmp_path):
    """Test load alert rules from directory."""
    exp_alert_rules = {"MyAlert1", "MyAlert2"}
    # create two example file
    (tmp_path / "test_cos_alert_rules.rules").write_text(MY_ALERT_1)
    (tmp_path / "test_cos_alert_rules.rule").write_text(MY_ALERT_2)

    assert get_alert_rules(tmp_path) == exp_alert_rules
