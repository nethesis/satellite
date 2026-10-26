"""No vendor traffic: both media boundaries use real local WebSockets."""
import asyncio
import base64
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestServer
import pytest

from diagnostics.primary_bridge import PrimaryBridgeAdapter
from test_agent_models import snapshot
from test_provider_emulators import eventually
from voice_agent.providers.realtime import ProviderError


@pytest.mark.asyncio
async def test_primary_bridge_preserves_live_startup_grants_and_carries_both_audio_directions():
    connection_id = '12345678-1234-1234-1234-123456789abc'
    accepted, input_frames, pbx_frames = [], [], []

    async def provider(request):
        assert request.headers['Authorization'] == 'Bearer synthetic-provider-key'
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for message in ws:
            event = json.loads(message.data)
            if event['type'] == 'session.start':
                accepted.append(event['session'])
                await ws.send_json({'type': 'session.started', 'session': {'id': 'live_primary_test'}})
            elif event['type'] == 'session.input_audio.append':
                input_frames.append(base64.b64decode(event['audio']))
            elif event['type'] == 'session.commentary.append':
                await ws.send_json({'type': 'session.commentary.appended', 'client_event_id': event['event_id']})
                # Odd chunks must be reframed without dropping or duplicating samples.
                for frame in (bytes([128] * 100), bytes([128] * 60)):
                    await ws.send_json({'type': 'session.output_audio.delta', 'delta': base64.b64encode(frame).decode()})
            elif event['type'] == 'session.close':
                await ws.send_json({'type': 'session.closed'})
        return ws

    async def media(request):
        assert request.headers['Authorization'] == aiohttp.encode_basic_auth('test-ari', 'test-password')
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_json({'event': 'MEDIA_START', 'channel_id': 'test-leg', 'connection_id': connection_id,
                            'format': 'ulaw', 'optimal_frame_size': 160})
        await ws.send_bytes(bytes([255] * 160))
        async for message in ws:
            if message.type == aiohttp.WSMsgType.BINARY:
                pbx_frames.append(message.data)
        return ws

    app = web.Application()
    app.router.add_get('/live', provider)
    app.router.add_get('/media/' + connection_id, media)
    server = TestServer(app)
    await server.start_server()
    controller_session = aiohttp.ClientSession(headers={
        'Authorization': aiohttp.encode_basic_auth('test-ari', 'test-password')})
    controller = SimpleNamespace(url=str(server.make_url('')).rstrip('/'), _session=controller_session,
                                 get_variable=AsyncMock(return_value=connection_id))
    storage = SimpleNamespace(accept=Mock())
    call = SimpleNamespace(terminal=False, leg_id='test-leg', provider_call_id=None)
    recorded = []
    adapter = PrimaryBridgeAdapter({'api_key': 'synthetic-provider-key'}, call, controller, storage,
                                    record=recorded.append)
    adapter.primary_ws_base = str(server.make_url('/live')).replace('http:', 'ws:')
    profile = snapshot('gpt-live-1')['conversation']
    profile.update(prompt='Synthetic instructions.', _opening_instructions='Synthetic full opening.')
    tools = [{'type': 'function', 'name': 'allowed_lookup', 'description': 'Synthetic lookup',
              'parameters': {'type': 'object', 'properties': {}}}]
    try:
        await adapter.accept(None, profile, tools)
        assert call.provider_call_id == 'live_primary_test'
        storage.accept.assert_called_once_with(call, 'live_primary_test')
        await adapter.connect(call.provider_call_id)
        await eventually(lambda: len(input_frames) == 1)
        await adapter.respond('Greet now.')
        await eventually(lambda: len(pbx_frames) == 1)
        assert input_frames == [bytes([255] * 160)]
        assert pbx_frames == [bytes([128] * 160)]
        session = accepted[0]
        assert session['model'] == 'gpt-live-1' and session['store'] is False
        assert session['audio']['format'] == {'type': 'audio/pcmu', 'rate': 8000}
        assert 'Synthetic full opening.' in session['input'][0]['content'][0]['text']
        assert session['delegation']['responses']['tools'][0]['name'] in adapter._wire_tools
        assert adapter.input_stats.samples == adapter.output_stats.samples == 160
    finally:
        await adapter.close()
        await controller_session.close()
        await server.close()
    assert len(recorded) == 1 and recorded[0]['variant'] == 'primary-ws'
    assert adapter._closed_event.is_set()
    assert 'synthetic-provider-key' not in json.dumps(recorded)


@pytest.mark.asyncio
async def test_primary_bridge_rejects_malformed_audio_and_bounds_backlog():
    call = SimpleNamespace(terminal=False, leg_id='test-leg', provider_call_id=None)
    adapter = PrimaryBridgeAdapter({'api_key': 'synthetic'}, call, None, None)
    for value in ['private malformed data', base64.b64encode(bytes(16001)).decode()]:
        with pytest.raises(ProviderError):
            await adapter._handle({'type': 'session.output_audio.delta', 'delta': value})
    assert adapter._audio.empty() and adapter.output_stats.samples == 0
    await adapter.close()


@pytest.mark.asyncio
async def test_closed_media_socket_reports_failure_instead_of_waiting_for_call_deadline():
    async def closed_messages():
        if False:
            yield

    call = SimpleNamespace(terminal=False, leg_id='test-leg', provider_call_id=None)
    adapter = PrimaryBridgeAdapter({'api_key': 'synthetic'}, call, None, None)
    adapter._media_ws = closed_messages()
    await adapter._read_media()
    assert not adapter._queue.empty(), 'media EOF must trigger the runtime fallback'
    assert adapter._queue.get_nowait() == {'type': 'error', 'code': 'provider_unavailable'}
    adapter._media_ws = None
    await adapter.close()


@pytest.mark.asyncio
async def test_close_discards_queued_audio_and_does_not_report_a_spurious_media_failure():
    call = SimpleNamespace(terminal=False, leg_id='test-leg', provider_call_id=None)
    adapter = PrimaryBridgeAdapter({'api_key': 'synthetic'}, call, None, None)
    await adapter._handle({'type': 'session.output_audio.delta',
                           'delta': base64.b64encode(bytes([128] * 160)).decode()})
    await adapter.close()
    assert adapter._audio.empty() and adapter._buffered_bytes == 0
    assert adapter._queue.empty()


@pytest.mark.asyncio
@pytest.mark.parametrize('selected', [False, True])
async def test_scoped_primary_origination_replaces_only_the_owned_synthetic_provider_leg(monkeypatch, selected):
    from diagnostics import scoped_runner
    from test_runtime import fixtures

    _, storage, controller, _ = fixtures()
    controller.app = 'voice-agent'
    original_originate = controller.originate_local
    runtime = scoped_runner.ScopedRuntime(storage, controller)
    call = SimpleNamespace(tasks=set())
    runtime.calls['owned-session'] = call
    runtime.converse = AsyncMock()
    monkeypatch.setattr(scoped_runner, 'read_selection', lambda candidate:
        {'variant': 'primary-ws'} if candidate is call and selected else None)
    await controller.originate_local('owned-session', 'owned-leg', {'__AGENT_SESSION_ID': 'owned-session'})
    await asyncio.sleep(0)
    if selected:
        original_originate.assert_not_awaited()
        controller.request.assert_awaited_once()
        params = controller.request.await_args.kwargs['params']
        assert params['channelId'] == 'owned-leg' and params['data'] == 'provider,owned-session'
        assert params['app'] == 'voice-agent' and params['external_host'] == 'INCOMING'
        assert params['transport'] == 'websocket' and params['connection_type'] == 'server'
        runtime.converse.assert_awaited_once_with(call)
    else:
        original_originate.assert_awaited_once()
        controller.request.assert_not_awaited()
        runtime.converse.assert_not_awaited()
    await asyncio.gather(*runtime.tasks, return_exceptions=True)
