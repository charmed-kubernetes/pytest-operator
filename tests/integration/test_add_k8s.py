# test that pytest operator supports adding a k8s to an existing controller
# This is a new k8s cloud created/managed by pytest-operator

from kubernetes.config.config_exception import ConfigException
import pytest
from pytest_operator.plugin import OpsTest


async def test_add_k8s(ops_test: OpsTest):
    try:
        k8s_cloud = await ops_test.add_k8s(skip_storage=False)
    except (ConfigException, TypeError):
        pytest.skip("No Kubernetes config found to add-k8s")

    k8s_model = await ops_test.track_model(
        "secondary", cloud_name=k8s_cloud, keep=ops_test.ModelKeep.NEVER
    )
    with ops_test.model_context("secondary"):
        await k8s_model.deploy("grafana-k8s", trust=True)
        await k8s_model.wait_for_idle(apps=["grafana-k8s"], status="active")
