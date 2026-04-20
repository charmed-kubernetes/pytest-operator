#!/usr/bin/env python3
import json

from ops.charm import CharmBase
from ops.main import main
from ops.model import ActiveStatus


class OperatorFrameworkCharm(CharmBase):
    def __init__(self, *args):
        super().__init__(*args)
        self.unit.status = ActiveStatus()
        self.framework.observe(self.on.operator_relation_joined, self.on_relation_joined)

    def on_relation_joined(self, event):
        event.relation.data[self.model.unit]["units"] = json.dumps([_.name for _ in event.relation.units])


if __name__ == "__main__":
    main(OperatorFrameworkCharm)
