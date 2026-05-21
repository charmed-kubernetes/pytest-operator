# Copyright 2024 Canonical Ltd.
# See LICENSE file for licensing details.

import pytest

import pytest_operator.cos
from pytest_operator.cos import (
    GRAFANA_AGENT_APP,
    assert_alert_rules,
    assert_logging,
    assert_metrics_endpoint,
)

# Note(rgildein): Change app metrics endpoint, since blackbox-exporter-k8s is not using
# 'metrics-endpoint'.
pytest_operator.cos.APP_METRICS_ENDPOINT = "self-metrics-endpoint"


@pytest.mark.abort_on_fail
@pytest.mark.skip_if_deployed
async def test_build_and_deploy(ops_test):
    """Test deployment of grafana-agent-k8s and relate it with charm."""
    await ops_test.model.deploy(
        "blackbox-exporter-k8s", channel="latest/stable", trust=True
    )
    await ops_test.model.wait_for_idle(raise_on_blocked=True)

    await ops_test.deploy_and_assert_grafana_agent(
        "blackbox-exporter-k8s",
        metrics=True,
        dashboard=True,
        logging=True,
    )


async def test_alert_rules(ops_test):
    """Test alert_rules are defined in relation data bag."""
    app = ops_test.model.applications["blackbox-exporter-k8s"]
    await assert_alert_rules(
        app,
        {
            "BlackboxJobMissing",
            "BlackboxExporterSSLCertExpiringSoon",
            "BlackboxExporterUnitIsUnavailable",
            "BlackboxExporterUnitIsDown",
        },
    )


async def test_metrics_endpoints(ops_test):
    """Test metrics_endpoints are defined in relation data bag."""
    app = ops_test.model.applications["blackbox-exporter-k8s"]
    await assert_metrics_endpoint(
        app,
        metrics_port=9115,
        metrics_path="/metrics",
        metrics_target="blackbox-exporter-k8s-0.blackbox-exporter-k8s-endpoints."
        f"{ops_test.model.name}.svc.cluster.local",
    )


async def test_logging(ops_test):
    """Test logging is defined in relation data bag."""
    app = ops_test.model.applications[GRAFANA_AGENT_APP]
    await assert_logging(app)
