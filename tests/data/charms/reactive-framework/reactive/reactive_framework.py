from charms.reactive import when_not
from charms import layer


@when_not("test-charm.installed")
def set_status():
    layer.status.active("")
