import copy
import json
from unittest.mock import AsyncMock

import pytest
from test_contract import CONTRACT
from voice_agent.tools import ToolRegistry


@pytest.mark.asyncio
async def test_tools_enforce_pinned_grants_and_idempotency():
    snapshot = json.loads((CONTRACT / 'published-v1.example.json').read_text())
    controller = AsyncMock()
    targets = lambda: [{'id': 'extension:203', 'name': 'Test extension'}, {'id': 'extension:204', 'name': 'Other test extension'}]
    registry = ToolRegistry(snapshot, targets, controller)
    assert await registry.invoke('one', 'directory.find_destinations', {}) == {'destinations': [{'id': 'extension:203', 'name': 'Test extension'}]}
    assert (await registry.invoke('two', 'telephony.transfer', {'destination_id': 'extension:204'}))['error'] == 'destination_denied'
    assert (await registry.invoke('three', 'telephony.transfer', {'context': 'arbitrary', 'destination_id': 'extension:203'}))['error'] == 'invalid_arguments'
    controller.transfer.return_value = {'outcome': 'transferred'}
    assert (await registry.invoke('four', 'telephony.transfer', {'destination_id': 'extension:203'}))['outcome'] == 'transferred'
    await registry.invoke('four', 'telephony.transfer', {'destination_id': 'extension:203'})
    controller.transfer.assert_awaited_once_with('extension:203')
    disabled = copy.deepcopy(snapshot)
    disabled['enabled_tools'] = []
    denied = ToolRegistry(disabled, targets, controller)
    assert (await denied.invoke('five', 'telephony.transfer', {'destination_id': 'extension:203'}))['error'] == 'tool_denied'
    snapshot['enabled_tools'] = []
    assert registry.snapshot['enabled_tools'] != []
