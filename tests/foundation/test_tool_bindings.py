"""Cross-plane v3 policies: typed per-agent settings never grant global access."""
import copy
import json
import subprocess
from unittest.mock import AsyncMock

import pytest
from test_contract import CONTRACT, ROOT
from voice_agent.contract import validate_snapshot
from voice_agent.tools import ToolRegistry


def snapshot():
    value = json.loads((CONTRACT / 'published-v1.example.json').read_text())
    value['contract_version'] = 3
    value['fallback_destination_id'] = 'hangup'
    value.pop('enabled_tools')
    value.pop('approved_destination_ids')
    value['tool_bindings'] = [
        {'tool_id': 'directory.find_destinations', 'tool_version': 2,
         'enabled': True, 'config': {'max_results': 1}},
        {'tool_id': 'telephony.transfer', 'tool_version': 2,
         'enabled': True, 'config': {'approved_destination_ids': ['extension:203', 'extension:204']}},
    ]
    return value


def test_v3_contract_and_manifest_copies():
    validate_snapshot(snapshot())
    for filename in ('published-v3.schema.json', 'tools-v2.json'):
        assert (CONTRACT / filename).read_bytes() == (ROOT / 'satellite/voice_agent' / filename).read_bytes()


@pytest.mark.parametrize('invalid', [None, 'boolean', 'limit', 'duplicate', 'version', 'unknown', 'queue'])
def test_v3_php_python_configuration_parity(invalid):
    value = snapshot()
    if invalid == 'boolean': value['tool_bindings'][0]['enabled'] = 1
    if invalid == 'limit': value['tool_bindings'][0]['config']['max_results'] = 101
    if invalid == 'duplicate': value['tool_bindings'].append(copy.deepcopy(value['tool_bindings'][0]))
    if invalid == 'version': value['tool_bindings'][0]['tool_version'] = 999
    if invalid == 'unknown': value['tool_bindings'][0]['config']['arbitrary'] = True
    if invalid == 'queue': value['tool_bindings'][1]['config']['approved_destination_ids'] = ['queue:600']
    result = subprocess.run(['php', str(ROOT / 'ns8-nethvoice/freepbx/tests/agents/validate.php')],
                            input=json.dumps(value), text=True, capture_output=True)
    assert result.returncode == (1 if invalid else 0)
    if invalid:
        with pytest.raises(ValueError): validate_snapshot(value)
    else:
        assert validate_snapshot(value) == value


@pytest.mark.asyncio
async def test_v3_config_pins_limit_and_transfer_visibility():
    value = snapshot()
    targets = lambda: [{'id': 'extension:203', 'name': 'Allowed'},
                       {'id': 'extension:204', 'name': 'Also allowed'},
                       {'id': 'extension:205', 'name': 'Denied'}]
    controller = AsyncMock()
    registry = ToolRegistry(value, targets, controller)
    value['tool_bindings'][1]['config']['approved_destination_ids'] = []
    assert len(registry.advertised()) == 2
    assert await registry.invoke('search', 'nv_find_destinations_v2', {}) == {'destinations': [{'id': 'extension:203', 'name': 'Allowed'}]}
    assert await registry.invoke('deny', 'nv_transfer_v2', {'destination_id': 'extension:205'}) == {'error': 'destination_denied'}
    controller.transfer.assert_not_awaited()
    disabled = snapshot()
    disabled['tool_bindings'][1]['enabled'] = False
    registry = ToolRegistry(disabled, targets, controller)
    assert len(registry.advertised()) == 1
    assert await registry.invoke('empty', 'directory.find_destinations', {}) == {'destinations': []}
    assert await registry.invoke('guess', 'telephony.transfer', {'destination_id': 'extension:203'}) == {'error': 'tool_denied'}
    disabled['tool_bindings'][0]['enabled'] = False
    registry = ToolRegistry(disabled, targets, controller)
    assert registry.advertised() == []
    assert await registry.invoke('disabled', 'directory.find_destinations', {}) == {'error': 'tool_denied'}
