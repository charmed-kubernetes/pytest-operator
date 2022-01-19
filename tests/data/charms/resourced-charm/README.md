## Resource Charm
This charm is to be used by unit test and integration tests
representative of a charm with arch based file resources attached.

### Building
To make changes, update this charm and rebuild to use in tests.

```bash
cd tests/data/charms/resourced-charm
charmcraft pack
rm build
mv *.charm ../resourced-charm.charm
```

