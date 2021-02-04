#!/usr/bin/env python3

from ops.charm import CharmBase
from ops.main import main
from ops.model import ActiveStatus


class OperatorFrameworkCharm(CharmBase):
    def __init__(self, *args):
        super().__init__(*args)
        self.unit.status = ActiveStatus()


if __name__ == "__main__":
    main(OperatorFrameworkCharm)
