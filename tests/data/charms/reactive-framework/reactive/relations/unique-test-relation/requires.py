from charms.reactive import when, Endpoint


class UniqueTestRelationRequires(Endpoint):
    @when('endpoint.{endpoint_name}.joined')
    def joined(self):
        for relation in self.relations:
            relation.to_publish["units"] = [_.unit_name for _ in relation.joined_units]

