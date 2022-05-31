import asyncio
from dataclasses import dataclass
from subprocess import PIPE, Popen
from typing import Dict

import yaml

_JUJU_DATA_CACHE = {}
_JUJU_KEYS = ('egress-subnets', 'ingress-address', 'private-address')


def _purge(data: dict):
    for key in _JUJU_KEYS:
        if key in data:
            del data[key]


async def _get_unit_info(unit_name: str) -> dict:
    """Returns unit-info data structure.

     for example:

    traefik-k8s/0:
      opened-ports: []
      charm: local:focal/traefik-k8s-1
      leader: true
      relation-info:
      - endpoint: ingress-per-unit
        related-endpoint: ingress
        application-data:
          _supported_versions: '- v1'
        related-units:
          prometheus-k8s/0:
            in-scope: true
            data:
              egress-subnets: 10.152.183.150/32
              ingress-address: 10.152.183.150
              private-address: 10.152.183.150
      provider-id: traefik-k8s-0
      address: 10.1.232.144
    """
    if cached_data := _JUJU_DATA_CACHE.get(unit_name):
        return cached_data

    proc = Popen(f'juju show-unit {unit_name}'.split(' '), stdout=PIPE)
    raw_data = proc.stdout.read().decode('utf-8').strip()
    if not raw_data:
        raise ValueError(
            f"no unit info could be grabbed for {unit_name}; "
            f"are you sure it's a valid unit name?"
        )

    data = yaml.safe_load(raw_data)
    _JUJU_DATA_CACHE[unit_name] = data
    return data


def _get_relation_by_endpoint(relations, endpoint, remote_obj):
    relations = [r for r in relations if
                 r['endpoint'] == endpoint and
                 remote_obj in r['related-units']]
    if not relations:
        raise ValueError(f'no relations found with endpoint=='
                         f'{endpoint}')
    if len(relations) > 1:
        raise ValueError('multiple relations found with endpoint=='
                         f'{endpoint}')
    return relations[0]


@dataclass
class UnitRelationData:
    unit_name: str
    endpoint: str
    leader: bool
    application_data: Dict[str, str]
    unit_data: Dict[str, str]


async def _get_endpoint_content(obj: str, other_obj,
                                include_default_juju_keys: bool = False) -> UnitRelationData:
    """Get the content of the databag of `obj`, relative to `other_obj`."""
    endpoint = None
    other_unit_name = other_obj.split(':')[0] if ':' in other_obj else other_obj
    if ':' in obj:
        unit_name, endpoint = obj.split(':')
    else:
        unit_name = obj
    data = (await _get_unit_info(unit_name))[unit_name]
    is_leader = data['leader']

    relation_infos = data.get('relation-info')
    if not relation_infos:
        raise RuntimeError(f'{unit_name} has no relations')

    if not endpoint:
        relation_data_raw = relation_infos[0]
        endpoint = relation_data_raw['endpoint']
    else:
        relation_data_raw = _get_relation_by_endpoint(
            relation_infos, endpoint, other_unit_name)

    related_units_data_raw = relation_data_raw['related-units']

    other_unit_name = next(iter(related_units_data_raw.keys()))
    other_unit_info = await _get_unit_info(other_unit_name)
    other_unit_relation_infos = other_unit_info[other_unit_name]['relation-info']
    remote_data_raw = _get_relation_by_endpoint(
        other_unit_relation_infos, relation_data_raw['related-endpoint'],
        unit_name)
    this_unit_data = remote_data_raw['related-units'][unit_name]['data']
    this_app_data = remote_data_raw['application-data']

    if not include_default_juju_keys:
        _purge(this_unit_data)

    return UnitRelationData(
        unit_name, endpoint, is_leader,
        this_app_data, this_unit_data
    )


@dataclass
class RelationData:
    provider: UnitRelationData
    requirer: UnitRelationData


async def get_relation_data(provider_endpoint: str, requirer_endpoint: str,
                            include_juju_keys: bool = False) -> RelationData:
    """Get relation databag contents for both sides of a juju relation.

    Usage:
    >>> data: RelationData = await ops_test.get_relation_data('prometheus/0:ingress', 'traefik/1:ingress-per-unit')
    >>> assert data.provider.application_data['key'] = 'foo'
    """
    provider_data, requirer_data = await asyncio.gather(
        _get_endpoint_content(provider_endpoint, requirer_endpoint, include_juju_keys),
        _get_endpoint_content(requirer_endpoint, provider_endpoint, include_juju_keys)
    )
    return RelationData(provider_data, requirer_data)
