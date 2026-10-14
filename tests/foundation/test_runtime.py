import asyncio
import base64
import copy
import hashlib
import hmac
import json
import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from test_contract import CONTRACT
from voice_agent.runtime import Runtime, Transfer
from voice_agent.runner import create_app
from voice_agent.tools import DEFINITIONS
from fastapi.testclient import TestClient


def fixtures():
    snapshot = json.loads((CONTRACT / 'published-v1.example.json').read_text())
    storage = MagicMock()
    storage.snapshot.return_value = snapshot
    storage.binding.return_value = {'binding_id':'provider_test','provider':'openai','openai_project_id':'proj_test','trunk_name':'VoiceAgentTrunk_1','api_key':'synthetic-api','webhook_secret':'whsec_' + base64.b64encode(b'synthetic-signing-secret').decode()}
    storage.targets.return_value = [{'id':'extension:203','name':'Test extension'}]
    storage.tools.return_value = copy.deepcopy(DEFINITIONS)
    controller = AsyncMock()
    controller.connected = True
    controller.get_variable.return_value = 'synthetic-linkedid'
    runtime = Runtime(storage, controller, MagicMock())
    return runtime, storage, controller, snapshot


async def admit(runtime):
    await runtime.admit({'id':'synthetic-caller','dialplan':{'context':'voice-agent-basic_voice'}},'basic_voice')
    return next(iter(runtime.calls.values()))


def sign(call, data, *, age=0):
    body = json.dumps({'id':'synthetic-event','type':'realtime.call.incoming','data':data}).encode()
    stamp = str(int(time.time()) - age)
    signature = base64.b64encode(hmac.new(b'synthetic-signing-secret', b'wh_test.' + stamp.encode() + b'.' + body, hashlib.sha256).digest()).decode()
    return body, {'webhook-id':'wh_test','webhook-timestamp':stamp,'webhook-signature':'v1,'+signature}


@pytest.mark.asyncio
async def test_admission_pins_configuration_and_persists_before_origination():
    runtime, storage, controller, snapshot = fixtures()
    call = await admit(runtime)
    storage.admit.assert_called_once_with(call)
    controller.originate_local.assert_awaited_once()
    snapshot['enabled_tools'] = []
    assert call.snapshot['enabled_tools'] != []
    await runtime.finish(call,'fallback','test_finished',fallback=True)
    controller.continue_channel.assert_awaited_with('synthetic-caller',{'context':'voice-agent-transfer','extension':'203','priority':1})
    assert not runtime.calls


@pytest.mark.asyncio
async def test_signed_webhook_requires_live_pinned_pending_call_and_deduplicates():
    runtime, storage, controller, snapshot = fixtures()
    call = await admit(runtime)
    runtime.converse = AsyncMock()
    data = {'call_id':'synthetic/provider:call','sip_headers':[
        {'name':'X-OS-Session-ID','value':call.session_id},
        {'name':'X-OS-Provider-Leg-ID','value':call.leg_id},
        {'name':'X-OS-Agent-ID','value':'basic_voice'},
        {'name':'Via','value':'synthetic-hop-1'},{'name':'Via','value':'synthetic-hop-2'}]}
    body, headers = sign(call, data)
    assert await runtime.webhook('openai',body,headers) == {'status':'accepted'}
    assert await runtime.webhook('openai',body,headers) == {'status':'duplicate'}
    await asyncio.sleep(0)
    runtime.converse.assert_awaited_once_with(call)
    storage.accept.assert_called_once_with(call,'synthetic/provider:call')
    with pytest.raises(ValueError):
        await runtime.webhook('grok',body,headers)
    with pytest.raises(ValueError):
        await runtime.webhook('openai',body+b' ',headers)
    stale, stale_headers = sign(call,data,age=301)
    with pytest.raises(ValueError):
        await runtime.webhook('openai',stale,stale_headers)
    unrelated = copy.deepcopy(data)
    unrelated['sip_headers'][0]['value'] = 'unrelated-session'
    raw, auth = sign(call,unrelated)
    assert await runtime.webhook('openai',raw,auth) == {'status':'ignored'}
    await runtime.finish(call,'completed')


@pytest.mark.asyncio
async def test_transfer_waits_for_confirmed_ari_release():
    runtime, storage, controller, snapshot = fixtures()
    call = await admit(runtime)
    transfer = asyncio.create_task(Transfer(runtime,call).transfer('extension:203'))
    await asyncio.sleep(0)
    assert not transfer.done()
    await runtime.event({'type':'StasisEnd','channel':{'id':call.caller_id,'dialplan':{'context':'voice-agent-transfer','exten':'203'}}})
    assert await transfer == {'outcome':'transferred'}
    events = list(runtime.monitoring.queue._queue)
    assert any(e['event_type']=='transfer_committed' for e in events)


@pytest.mark.asyncio
async def test_hangup_racing_transfer_is_not_success():
    runtime, storage, controller, snapshot = fixtures()
    call = await admit(runtime)
    transfer = asyncio.create_task(Transfer(runtime,call).transfer('extension:203'))
    await asyncio.sleep(0)
    await runtime.event({'type':'ChannelDestroyed','channel':{'id':call.caller_id}})
    assert await transfer == {'error':'transfer_failed'}
    assert not any(e['event_type']=='transfer_committed' for e in runtime.monitoring.queue._queue)


def test_webhook_http_body_boundaries_and_public_route_isolation():
    runtime = MagicMock()
    runtime.webhook = AsyncMock(return_value={'status':'ignored'})
    client = TestClient(create_app('synthetic-token',runtime))
    assert client.post('/satellite-webhooks/openai', content=b'x'*(256*1024+1)).status_code == 413
    assert client.post('/satellite-webhooks/unexpected',content=b'{}').status_code == 404
    assert client.post('/satellite-webhooks/openai/unexpected',content=b'{}').status_code == 404
    assert client.post('/satellite-webhooks/openai/',content=b'{}').status_code == 404
    assert client.get('/internal/ready').status_code == 401


@pytest.mark.asyncio
async def test_parallel_tool_batch_does_not_block_event_consumption():
    runtime, storage, controller, snapshot = fixtures()
    call = await admit(runtime)
    call.provider_call_id = 'synthetic-provider-call'
    call.provider_answered.set()
    completed = asyncio.Event()
    results = []
    class Adapter:
        async def accept(self,*args): pass
        async def connect(self,*args): pass
        async def close(self): pass
        async def respond(self,*args,**kwargs):
            # A continuation must wait for both function results.
            if kwargs.get('response_id'):
                await completed.wait()
        async def send_result(self,identifier,result):
            results.append((identifier,result))
            if len(results)==2: completed.set()
        async def events(self):
            for identifier in ('one','two'):
                yield {'type':'tool_call','invocation_id':identifier,'name':'nv_find_destinations_v1','arguments':{},'response_id':'batch'}
            await asyncio.Event().wait()
    runtime.adapter_factory=lambda binding: Adapter()
    conversation=runtime.spawn(runtime.converse(call),call)
    await asyncio.wait_for(completed.wait(),timeout=1)
    assert {identifier for identifier,result in results}=={'one','two'}
    assert not call.terminal
    await runtime.finish(call,'completed')
    await asyncio.gather(conversation,return_exceptions=True)


@pytest.mark.asyncio
async def test_caller_hangup_cancels_admission_before_provider_origination():
    import threading
    runtime, storage, controller, snapshot = fixtures()
    waiting = threading.Event()
    released = threading.Event()
    def slow_snapshot(identifier):
        waiting.set()
        released.wait(timeout=1)
        return snapshot
    storage.snapshot.side_effect=slow_snapshot
    await runtime.event({'type':'StasisStart','args':['caller','basic_voice'],'channel':{'id':'early-caller','dialplan':{'context':'voice-agent-basic_voice'}}})
    try:
        for _ in range(20):
            if waiting.is_set(): break
            await asyncio.sleep(.01)
        task=runtime.admissions['early-caller']
        await runtime.event({'type':'StasisEnd','channel':{'id':'early-caller'}})
        await asyncio.gather(task,return_exceptions=True)
        controller.originate_local.assert_not_awaited()
        assert not runtime.calls
    finally:
        released.set()


@pytest.mark.asyncio
async def test_restart_recovery_touches_only_owned_stasis_channels():
    runtime, storage, controller, snapshot = fixtures()
    controller.list_channels.return_value=[
        {'id':'unrelated','dialplan':{'context':'voice-agent-internal','app_name':'Dial','app_data':'PJSIP/test'}},
        {'id':'owned','dialplan':{'context':'voice-agent-internal','app_name':'Stasis','app_data':'voice-agent,caller,internal'}},
        {'id':'old-provider','dialplan':{'context':'voice-agent-provider','app_name':'Stasis','app_data':'voice-agent,provider,old'}},
    ]
    await runtime.event({'type':'AgentAriConnected'})
    controller.continue_channel.assert_awaited_once_with('owned',{'context':'voice-agent-internal','extension':'s','label':'fallback'})
    controller.hangup.assert_awaited_once_with('old-provider')


@pytest.mark.asyncio
async def test_failed_durable_admission_emits_no_orphan_event_or_provider_leg():
    runtime,storage,controller,snapshot=fixtures()
    storage.admit.side_effect=RuntimeError('synthetic_db_failure')
    await runtime.admit({'id':'synthetic-caller','dialplan':{'context':'voice-agent-basic_voice'}},'basic_voice')
    controller.originate_local.assert_not_awaited()
    assert not runtime.monitoring.pending
    assert not runtime.calls


@pytest.mark.asyncio
async def test_full_monitoring_queue_denies_new_admission_before_sip():
    runtime,storage,controller,snapshot=fixtures()
    runtime.monitoring.pending={str(number):{} for number in range(192)}
    await runtime.admit({'id':'synthetic-caller','dialplan':{'context':'voice-agent-basic_voice'}},'basic_voice')
    storage.admit.assert_not_called()
    controller.originate_local.assert_not_awaited()
    controller.continue_channel.assert_awaited_once()


@pytest.mark.asyncio
async def test_v2_fallback_pins_native_wrapper_without_model_dialplan():
    runtime,storage,controller,snapshot=fixtures()
    snapshot['contract_version']=2
    snapshot['fallback_destination_id']='native:ivr-7,s,1'
    call=await admit(runtime)
    snapshot['fallback_destination_id']='hangup'
    await runtime.finish(call,'fallback','test',fallback=True)
    import hashlib
    digest=hashlib.sha256(('basic_voice\0native:ivr-7,s,1').encode()).hexdigest()[:24]
    controller.continue_channel.assert_awaited_with('synthetic-caller',{'context':'voice-agent-fallback-'+digest,'extension':'s','priority':1})


@pytest.mark.parametrize('bridge_allocated', [False, True])
@pytest.mark.asyncio
async def test_caller_hangup_before_provider_connection_is_interrupted_not_completed(bridge_allocated):
    runtime,storage,controller,snapshot=fixtures()
    call=await admit(runtime)
    if bridge_allocated: call.bridge_id='synthetic-pending-bridge'
    await runtime.event({'type':'StasisEnd','channel':{'id':call.caller_id}})
    await asyncio.sleep(0.01)
    events=list(runtime.monitoring.queue._queue)
    terminal=next(e for e in events if e.get('outcome'))
    assert terminal['outcome']=='interrupted'
    assert terminal['error_code']=='caller_cancelled_before_connect'


@pytest.mark.asyncio
async def test_connection_phases_and_response_metadata_are_bounded_and_deduplicated():
    runtime, storage, controller, snapshot = fixtures()
    call = await admit(runtime)
    call.provider_call_id = 'synthetic-provider-call'
    call.provider_answered.set()
    class Adapter:
        async def accept(self, *args): pass
        async def connect(self, *args): pass
        async def close(self): pass
        async def respond(self, *args, **kwargs): pass
        async def events(self):
            for number in range(40):
                event = {'type':'response_done','status':'completed','response_id':f'synthetic-{number}'}
                yield event
                yield event
            yield {'type':'response_done','status':'failed','response_id':'failed-response'}
            await asyncio.Event().wait()
    runtime.adapter_factory = lambda binding: Adapter()
    conversation = runtime.spawn(runtime.converse(call), call)
    for _ in range(30):
        if len(call.observed_responses) == 32: break
        await asyncio.sleep(.01)
    events = list(runtime.monitoring.queue._queue)
    assert [event['event_type'] for event in events[:3]] == ['admitted','provider_connected','media_connected']
    assert sum(event['event_type'] == 'response_completed' for event in events) == 32
    assert len(call.observed_responses) == 32
    assert all('response_id' not in event and 'transcript' not in event for event in events)
    assert call.media_connected
    await runtime.event({'type':'StasisEnd','channel':{'id':call.caller_id}})
    await asyncio.gather(conversation, return_exceptions=True)
    await asyncio.sleep(.01)
    assert any(event.get('outcome') == 'completed' for event in runtime.monitoring.queue._queue)
