"""Local Live HTTP/WebSocket fixtures; never contact a provider."""

import asyncio
import base64
import hashlib
import hmac
import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from agent.providers import create_adapter
from agent.providers.live import BACKEND_MODEL, OpenAILiveAdapter
from agent.providers.realtime import GrokAdapter, OpenAIAdapter, ProviderError
from agent.runtime import AgentRuntime, CallState
from agent.workflows.voice import VoiceWorkflows
from tests.test_agent_openai_emulator import eventually
from tests.test_agent_voice import setup, admit, run_async, signed_event
from tests.test_workflow_voice import call_fixture, consultation_runtime


TOOL = {"type": "function", "name": "lookup", "description": "Read a record",
        "parameters": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}}


class LiveEmulator:
    def __init__(self):
        self.accepted = []
        self.received = []
        self.ws = None
        self.reject = None
        self.app = web.Application()
        self.app.router.add_post('/v1/live/sessions/{session_id}/accept', self.accept)
        self.app.router.add_post('/v1/live/sessions/{session_id}/hangup', self.hangup)
        self.app.router.add_get('/v1/live/sessions/{session_id}/attach', self.attach)

    async def accept(self, request):
        assert request.headers['Authorization'] == 'Bearer key'
        assert request.match_info['session_id'] == 'live_test'
        body = await request.json()
        assert body['session']['type'] == 'live'
        assert 'format' not in body['session']['audio']
        self.accepted.append(body)
        return web.Response()

    async def attach(self, request):
        assert request.headers['Authorization'] == 'Bearer key'
        assert not request.query
        self.ws = web.WebSocketResponse()
        await self.ws.prepare(request)
        async for message in self.ws:
            event = json.loads(message.data)
            self.received.append(event)
            assert event['type'] != 'session.start'
            if self.reject == event['type']:
                await self.ws.send_json({'type': 'error', 'error': {
                    'client_event_id': event['event_id'], 'message': 'PRIVATE ERROR', 'code': 'invalid'}})
            elif event['type'] == 'session.update':
                assert set(event['session']) == {'delegation'}
                await self.ws.send_json({'type': 'session.updated', 'client_event_id': event['event_id']})
            elif event['type'].endswith('.append'):
                assert len(event['content'].encode()) <= 400
                await self.ws.send_json({'type': event['type'] + 'ed', 'client_event_id': event['event_id']})
        return self.ws

    async def hangup(self, request):
        await self.ws.send_json({'type': 'session.closed', 'usage': {'seconds': 15}})
        return web.Response()

    async def event(self, kind, *, response='r1', delegation='d1', **fields):
        payload = {'type': kind, **fields}
        if kind in ('response.created', 'response.completed'):
            payload.setdefault('response', {'id': response, 'status': 'completed', 'output': []})
        # output_item.done deliberately omits response_id, like Responses events.
        await self.ws.send_json({'type': 'response.event', 'delegation_id': delegation, 'event': payload})

    async def tool(self, wire, *, invocation='c1', arguments=None, response='r1', delegation='d1', complete=True):
        await self.event('response.created', response=response, delegation=delegation)
        item = {'type': 'function_call', 'name': wire, 'call_id': invocation,
                'arguments': json.dumps(arguments or {'id': '42'})}
        await self.event('response.output_item.done', item=item, delegation=delegation)
        if complete:
            await self.event('response.completed', response=response, delegation=delegation)
        return item


@asynccontextmanager
async def connected(profile=None, tools=None):
    emulator = LiveEmulator()
    server = TestServer(emulator.app)
    await server.start_server()
    adapter = OpenAILiveAdapter({'api_key': 'key'})
    adapter.http_base = str(server.make_url('/v1/live/sessions')).rstrip('/')
    adapter.ws_base = adapter.http_base.replace('http:', 'ws:')
    try:
        await adapter.accept('live_test', profile or {'model': 'gpt-live-1', 'language': 'it'}, tools or [TOOL])
        await adapter.connect('live_test')
        yield adapter, emulator
    finally:
        await adapter.close()
        await server.close()


@pytest.mark.parametrize('provider,model,adapter_class', [
    ('openai', '', OpenAIAdapter), ('openai', 'gpt-realtime', OpenAIAdapter),
    ('openai', 'gpt-live-1', OpenAILiveAdapter), ('grok', 'gpt-live-1', GrokAdapter)])
def test_model_selects_adapter_without_changing_provider(provider, model, adapter_class):
    assert type(create_adapter({'provider': provider, 'api_key': 'key'}, {'model': model})) is adapter_class


@pytest.mark.asyncio
async def test_live_accepts_sip_with_separate_backend_and_greeting():
    async with connected({'model': 'gpt-live-1', 'language': 'it', 'prompt': 'Use lookup for facts.'}) as (adapter, emulator):
        session = emulator.accepted[0]['session']
        backend = session['delegation']['responses']
        assert backend['model'] == BACKEND_MODEL
        assert not backend['parallel_tool_calls'] and not session['store']
        assert backend['tools'][0]['name'] in backend['instructions']
        assert session['audio']['output']['voice'] == 'marin'
        await adapter.respond('Salve, come posso aiutarti?')
        assert emulator.received[-1]['type'] == 'session.instructions.append'
        await adapter.hangup('live_test')
        assert adapter._closed_event.is_set()


@pytest.mark.asyncio
async def test_completed_items_drive_tools_and_continuation_exactly_once():
    async with connected() as (adapter, emulator):
        wire = next(iter(adapter._wire_tools))
        item = await emulator.tool(wire, complete=False)
        await eventually(lambda: 'c1' in adapter._invocations)
        tool_event = await anext(adapter.events())
        assert tool_event['name'] == 'lookup' and tool_event['arguments'] == {'id': '42'}
        await emulator.event('response.output_item.done', item=item)
        await adapter.send_result('c1', {'found': True})
        continuation = asyncio.create_task(adapter.respond(response_id='r1'))
        await asyncio.sleep(0)
        assert not continuation.done()
        await emulator.event('response.completed')
        await asyncio.wait_for(continuation, 1)
        await adapter.respond(response_id='r1')
        await eventually(lambda: any(e['type'] == 'response.create' for e in emulator.received))
        assert sum(e['type'] == 'response.create' for e in emulator.received) == 1
        assert len(adapter._invocations) == 1
        results = [e for e in emulator.received if e['type'] == 'response.item.create']
        assert json.loads(results[0]['item']['output']) == {'found': True}


@pytest.mark.asyncio
async def test_step_update_rejects_old_tool_even_with_reused_business_name():
    async with connected() as (adapter, emulator):
        old_wire = next(iter(adapter._wire_tools))
        await adapter.workflow_update('Use lookup for the next step.', [TOOL])
        assert old_wire not in adapter._wire_tools
        await emulator.tool(old_wire)
        await eventually(lambda: any(e['type'] == 'response.item.create' for e in emulator.received))
        assert not any(e.get('type') == 'tool_call' for e in list(adapter._queue._queue))
        output = next(e['item']['output'] for e in emulator.received if e['type'] == 'response.item.create')
        assert json.loads(output)['error'] == 'stale_workflow_tool'


@pytest.mark.asyncio
async def test_prompt_ready_uses_tool_and_acknowledged_text_without_audio_events():
    async with connected() as (adapter, emulator):
        await adapter.workflow_update('Read back the amount and request DTMF.', [], auto_response=False)
        await adapter.workflow_input({'amount': '42 EUR'})
        task = asyncio.create_task(adapter.workflow_respond(wait=True))
        await eventually(lambda: adapter._prompt_tool is not None and bool(adapter._wire_tools))
        await eventually(lambda: any(e['type'] == 'response.create' for e in emulator.received))
        assert not task.done()
        text = 'Confermi 42 EUR? Premi 1 o 2. ' + 'è' * 450
        await emulator.tool(next(iter(adapter._wire_tools)), arguments={'message': text})
        await asyncio.wait_for(task, 2)
        chunks = [e['content'] for e in emulator.received if e['type'] == 'session.commentary.append']
        assert ''.join(chunks) == text
        assert adapter._prompt_tool is None


@pytest.mark.asyncio
async def test_command_errors_are_correlated_and_redacted():
    async with connected() as (adapter, emulator):
        emulator.reject = 'session.update'
        with pytest.raises(ProviderError, match='command rejected') as error:
            await adapter.workflow_update('step', [])
        assert 'PRIVATE' not in str(error.value)
        assert not adapter._acks


@pytest.mark.asyncio
async def test_transcripts_capture_policy_and_cumulative_usage():
    async with connected({'model': 'gpt-live-1', '_capture_transcripts': True}) as (adapter, emulator):
        event = {'type': 'session.input_transcript.delta', 'event_id': 't1', 'delta': '  ripeti', 'start_ms': 20, 'end_ms': 50}
        await adapter._handle(event)
        await adapter._handle(event)
        transcript = await anext(adapter.events())
        assert transcript['text'] == '  ripeti' and transcript['position'] == 20
        await adapter.set_capture(False)
        await adapter._handle(event | {'event_id': 't2'})
        assert adapter._observation_queue.empty()
        await adapter._handle({'type': 'session.usage.updated', 'usage': {'seconds': 12}})
        await adapter._handle({'type': 'session.usage.updated', 'usage': {'seconds': 15}})
        assert (await anext(adapter.events()))['usage']['seconds'] == 12
        assert (await anext(adapter.events()))['usage']['seconds'] == 15


def live_envelope(call, kind='live.transport.incoming', **changes):
    envelope = signed_event(call)
    event = json.loads(base64.b64decode(envelope['raw_body']))
    event['type'] = kind
    event['data'].pop('call_id')
    event['data'].update(type='sip', session_id='live_test')
    event['data'].update(changes)
    raw = json.dumps(event).encode()
    headers = envelope['headers']
    signature = hmac.new(b'test-secret-for-agent',
        (headers['webhook-id'] + '.' + headers['webhook-timestamp'] + '.').encode() + raw, hashlib.sha256).digest()
    envelope['raw_body'] = base64.b64encode(raw).decode()
    headers['webhook-signature'] = 'v1,' + base64.b64encode(signature).decode()
    return envelope


@run_async
async def test_live_webhook_requires_selected_api_and_sip_and_remains_pinned(setup):
    _, runtime, _, adapter = setup
    runtime.store.snapshot['profiles']['internal']['model'] = 'gpt-live-1'
    call = await admit(setup)
    runtime.store.snapshot['profiles']['internal']['model'] = 'gpt-realtime'
    assert await runtime.provider_event('openai', signed_event(call)) == {'status': 'ignored'}
    assert await runtime.provider_event('openai', live_envelope(call, type='webrtc')) == {'status': 'ignored'}
    assert await runtime.provider_event('openai', live_envelope(call)) == {'status': 'accepted'}
    assert await runtime.provider_event('openai', live_envelope(call, kind='live.call.incoming')) == {'status': 'duplicate'}
    await eventually(lambda: bool(adapter.accepted))
    assert adapter.accepted == ['live_test']


@run_async
async def test_realtime_ignores_live_notification_before_accepting_its_own(setup):
    call = await admit(setup)
    runtime = setup[1]
    assert await runtime.provider_event('openai', live_envelope(call)) == {'status': 'ignored'}
    assert await runtime.provider_event('openai', signed_event(call)) == {'status': 'accepted'}


@pytest.mark.asyncio
async def test_dtmf_confirmation_still_requires_caller_digit_after_prompt_ready():
    runtime = AgentRuntime()
    call = call_fixture()
    ready = asyncio.get_running_loop().create_future()
    runtime.workflow_speak = AsyncMock(side_effect=lambda *args: None)
    async def speak(*args):
        await ready
    runtime.workflow_speak.side_effect = speak
    task = asyncio.create_task(runtime.voice_workflows.confirm(call, 'Approve ticket 42?'))
    await asyncio.sleep(0)
    digit = {'type': 'ChannelDtmfReceived', 'channel': {'id': call.caller_id}, 'digit': '1'}
    assert not await runtime.voice_workflows.event(digit)
    ready.set_result(None)
    await eventually(lambda: call.caller_id in runtime.voice_workflows.digits)
    assert not task.done()
    await runtime.voice_workflows.event(digit | {'digit': '2'})
    assert await task is False


@pytest.mark.asyncio
async def test_private_replacement_preserves_call_and_pinned_workflow():
    runtime = AgentRuntime(controller=AsyncMock())
    call = call_fixture()
    call.state = CallState.CONSULTING
    call.workflow = {'nodes': ['pinned']}
    call.workflow_task = asyncio.current_task()
    call.greeting_started = True
    call.binding.update(trunk_name='AgentTrunk_1', provider_user='proj_test', provider_host='sip.api.openai.com')
    runtime.calls[call.session_id] = call
    runtime.by_local[call.local_id] = call.session_id
    original = (call.run_id, call.workflow, call.deadline, call.permissions, call.caller_id)
    old_adapter, old_local, old_leg = call.adapter, call.local_id, call.leg_id
    async def originate(session, local, variables):
        assert local != old_local and variables['__AGENT_PROVIDER_LEG_ID'] != old_leg
        assert 'summary' not in str(variables)
        call.provider_ready = call.local_answered = True
        await runtime._maybe_bridge(call)
    runtime.controller.originate_local.side_effect = originate
    await runtime.replace_private_provider(call)
    assert original == (call.run_id, call.workflow, call.deadline, call.permissions, call.caller_id)
    assert call.state == CallState.CONVERSING and call.provider_replacement is None
    old_adapter.hangup.assert_awaited_once_with('provider-call')
    old_adapter.close.assert_awaited_once()
    runtime.controller.hangup.assert_awaited_once_with(old_local)
    assert old_local not in runtime.by_local


@pytest.mark.asyncio
async def test_cancelled_replacement_still_releases_every_private_resource():
    runtime = AgentRuntime(controller=AsyncMock())
    call = call_fixture()
    call.state = CallState.CONSULTING
    old_adapter, old_local, old_bridge = call.adapter, call.local_id, call.bridge_id
    entered, release = asyncio.Event(), asyncio.Event()
    async def hangup(*args):
        entered.set()
        await release.wait()
        raise ProviderError('synthetic cleanup failure')
    old_adapter.hangup.side_effect = hangup
    task = asyncio.create_task(runtime.replace_private_provider(call))
    await entered.wait()
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    old_adapter.close.assert_awaited_once()
    runtime.controller.hangup.assert_awaited_once_with(old_local)
    runtime.controller.destroy_bridge.assert_awaited_once_with(old_bridge)
    runtime.controller.originate_local.assert_not_awaited()
    assert call.provider_replacement is None


@pytest.mark.asyncio
async def test_queued_old_step_events_cannot_spend_turns_or_execute_tools():
    runtime = AgentRuntime(controller=AsyncMock())
    call = call_fixture()
    call.adapter._generation = 2
    future = asyncio.get_running_loop().create_future()
    call.workflow_step = {'turns': 0, 'max_turns': 1, 'future': future}
    async def events():
        yield {'type': 'response_done', 'response_id': 'old', 'generation': 1}
    call.adapter.events = events
    await runtime._provider_events(call)
    assert call.workflow_step['turns'] == 0 and not future.done()
    runtime.tools.dispatch = AsyncMock()
    await runtime._tool_call(call, {'name': 'lookup', 'invocation_id': 'old', 'generation': 1})
    runtime.tools.dispatch.assert_not_awaited()
    call.adapter.send_result.assert_awaited_once_with('old', {'error': 'stale_workflow_tool'})


@pytest.mark.asyncio
@pytest.mark.parametrize('decision', ['declined', 'unavailable', 'no_answer'])
async def test_private_live_consultation_replaces_session_only_after_private_summary(decision):
    call = call_fixture()
    call.adapter.api = 'live'
    runtime = consultation_runtime()
    runtime.replace_private_provider = AsyncMock()
    async def release_operator(*args):
        assert call.private_consultation is (decision != 'no_answer')
    runtime.controller.hangup.side_effect = release_operator
    voice = VoiceWorkflows(runtime)
    task = asyncio.create_task(voice.consult(call, 'extension:201', 'Private ticket summary',
                                           {'ring_seconds': 5, 'consult_seconds': 5}))
    await eventually(lambda: bool(voice.consultations))
    attempt = next(iter(voice.consultations.values()))
    if decision == 'no_answer':
        await voice.event({'type': 'ChannelDestroyed', 'channel': {'id': attempt.channel_id}, 'cause': 19})
    else:
        await voice.event({'type': 'StasisStart', 'channel': {'id': attempt.channel_id},
                           'args': ['consult', call.session_id, attempt.attempt_id]})
        await eventually(lambda: attempt.accepting)
        if decision == 'declined':
            await voice.event({'type': 'ChannelDtmfReceived', 'channel': {'id': attempt.channel_id}, 'digit': '2'})
        else:
            await voice.event({'type': 'ChannelDestroyed', 'channel': {'id': attempt.channel_id}})
    assert (await task)['status'] == decision
    if decision == 'no_answer':
        runtime.replace_private_provider.assert_not_awaited()
    else:
        runtime.replace_private_provider.assert_awaited_once_with(call)
        assert not any(args.args[0] == call.bridge_id for args in runtime.controller.add_to_bridge.await_args_list)


@pytest.mark.asyncio
async def test_closed_socket_releases_pending_commands_and_cleanup_keeps_final_usage():
    async with connected() as (adapter, emulator):
        pending = asyncio.create_task(adapter._command('unanswered.command', 'unanswered.reply'))
        await eventually(lambda: bool(adapter._acks))
        await emulator.ws.close()
        with pytest.raises(ProviderError, match='sideband closed'):
            await asyncio.wait_for(pending, 1)
    async with connected() as (adapter, emulator):
        runtime = AgentRuntime(controller=AsyncMock())
        call = call_fixture()
        call.adapter, call.provider_call_id = adapter, 'live_test'
        runtime.monitoring.enqueue = lambda kind, value: records.append((kind, value))
        records = []
        await runtime._cleanup(call, 'workflow_completed', caller_action=None)
        usage = next(value['usage'] for kind, value in records if kind == 'usage')
        assert usage['seconds'] == 15 and usage['final'] is True


@pytest.mark.asyncio
async def test_live_completion_advances_current_workflow_with_validated_data():
    from agent.workflows.templates import node, conversation
    async with connected() as (adapter, emulator):
        runtime = AgentRuntime(controller=AsyncMock())
        call = call_fixture()
        call.adapter = adapter
        call.workflow = {'tool_grants': []}
        call.workflow_context_data = runtime._context(call)
        runtime.workflows.conversation_tools = AsyncMock(return_value=[])
        step = node('collect', 'conversation.collect', conversation('Collect a required ticket summary.',
                    {'summary': {'type': 'string', 'minLength': 1}}))
        events = asyncio.create_task(runtime._provider_events(call))
        collect = asyncio.create_task(runtime.workflow_conversation(call, step, {}))
        try:
            await eventually(lambda: call.workflow_step is not None and bool(adapter._wire_tools))
            await eventually(lambda: any(e['type'] == 'response.create' for e in emulator.received))
            wire = next(iter(adapter._wire_tools))
            await emulator.tool(wire, arguments={'outcome': 'success', 'data': {'summary': 'Printer offline'}})
            assert await asyncio.wait_for(collect, 2) == {'outcome': 'success', 'data': {'summary': 'Printer offline'}}
            assert call.workflow_step is None
            await eventually(lambda: any(e['type'] == 'response.item.create' and
                e['item']['type'] == 'function_call_output' for e in emulator.received))
            results = [json.loads(e['item']['output']) for e in emulator.received
                       if e['type'] == 'response.item.create' and e['item']['type'] == 'function_call_output']
            assert results == [{'status': 'step_completed'}]
        finally:
            events.cancel()
            collect.cancel()
            await asyncio.gather(events, collect, return_exceptions=True)
