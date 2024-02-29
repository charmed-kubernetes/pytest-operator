# test that pytest operator supports adding a k8s to an existing controller
# This is a new k8s cloud created/managed by pytest-operator

from pytest_operator.plugin import OpsTest
from kubernetes import config as k8s_config
from kubernetes.client import Configuration

async def test_add_k8s_cloud(ops_test: OpsTest):
    config = type.__call__(Configuration)
    k8s_config.load_config(client_configuration=config)
    k8s_cloud = await ops_test.add_k8s(config, skip_storage=True, storage_class=None)
    k8s_model = await ops_test.track_model("secondary", cloud_name=k8s_cloud, keep=ops_test.ModelKeep.NEVER)
    with ops_test.model_context("secondary"):
        await k8s_model.deploy("coredns", trust=True)
        await k8s_model.wait_for_idle(apps=["coredns"], status="active")
