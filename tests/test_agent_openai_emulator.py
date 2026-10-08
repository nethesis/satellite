"""Signed HTTP webhooks plus a local OpenAI HTTP/WebSocket simulator; no real calls."""

import asyncio
import copy
import json
import time
from contextlib import asynccontextmanager

import httpx
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from fastapi import FastAPI

from agent.api import create_router
from agent.providers.realtime import OpenAIAdapter
from agent.prompt import execution_profile
from agent.runtime import AgentRuntime, CallState
from agent.tools import ToolRegistry
from tests.test_agent_voice import FakeAri, FakeStore, signed_event


@pytest.fixture(autouse=True)
def isolated_configuration(monkeypatch):
    monkeypatch.setenv('API_TOKEN', 'emulator-private-token')
    monkeypatch.delenv('SATELLITE_AGENT_STATE_PATH', raising=False)


class OpenAIEmulator:
    def __init__(self):
        self.accepted = []
        self.received = []
        self.hangups = []
        self.ws = None
        self.app = web.Application()
        self.app.router.add_post('/v1/realtime/calls/{call_id}/accept', self.accept)
        self.app.router.add_post('/v1/realtime/calls/{call_id}/hangup', self.hangup)
        self.app.router.add_get('/v1/realtime', self.connect)

    async def accept(self, request):
        assert request.headers['Authorization'] == 'Bearer key'
        self.accepted.append(await request.json())
        return web.Response(status=200)

    async def hangup(self, request):
        self.hangups.append(request.match_info['call_id'])
        return web.Response(status=200)

    async def connect(self, request):
        assert request.query['call_id'] == 'provider-1'
        self.ws = web.WebSocketResponse()
        await self.ws.prepare(request)
        await self.ws.send_json({'type': 'session.updated', 'session': {'type': 'realtime'}})
        async for message in self.ws:
            event = json.loads(message.data)
            self.received.append(event)
            if event['type'] == 'session.update':
                await self.ws.send_json({'type': 'session.updated', 'session': event['session']})
        return self.ws

    async def tool(self, name, arguments, call_id='tool-1', response_id='response-1', repeat=False):
        item = {'type': 'function_call', 'name': name, 'arguments': json.dumps(arguments), 'call_id': call_id}
        await self.ws.send_json({'type': 'response.created', 'response': {'id': response_id}})
        await self.ws.send_json({**item, 'type': 'response.function_call_arguments.done', 'response_id': response_id})
        # OpenAI can emit the same call in the arguments event and response.done.
        await self.ws.send_json({'type': 'response.done', 'response': {'id': response_id, 'status': 'completed', 'output': [item]}})
        if repeat:
            await self.ws.send_json({**item, 'type': 'response.function_call_arguments.done', 'response_id': response_id})

    def results(self, call_id):
        return [json.loads(e['item']['output']) for e in self.received if e['type'] == 'conversation.item.create' and e['item'].get('call_id') == call_id]


async def eventually(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(.01)


@asynccontextmanager
async def harness(*, external=False, disabled=False, denied=False):
    emulator = OpenAIEmulator()
    server = TestServer(emulator.app)
    await server.start_server()
    store, ari = FakeStore(), FakeAri()
    profile = store.snapshot['profiles']['internal']
    profile.update(prompt='Be concise.', language='it', greeting='Hello', company={'company_name': 'Test Company'}, calendar_services={'support': '1'})
    profile['tools'] = {key: 'enabled' for key in ('directory.find_destinations','company.get_information','calendar.get_opening_hours','telephony.handoff')}
    profile['permissions'].update({'company.public_information': 'allow', 'calendar.opening_hours': 'allow'})
    if disabled:
        profile['tools']['telephony.handoff'] = 'disabled'
    if denied:
        profile['permissions']['telephony.transfer.extension'] = 'deny'
    store.snapshot['profiles']['external'] = copy.deepcopy(profile)
    names = {'extension': 'Alice Example', 'queue': 'Customer Support', 'ivr': 'Sales Menu'}
    for resource in store.snapshot['directory']:
        resource.update(name=names[resource['type']], description='Synthetic test destination', synonyms=['test alias'])
    store.snapshot['directory'].append({'id':'extension:999','type':'extension','name':'Private Destination','internal_allowed':False,'external_allowed':False,'target':{'context':'private-secret-routing','exten':'999','priority':1}})
    store.snapshot['calendars'] = {'1': {'timezone':'UTC','rules':['00:00-23:59|*|*|*'],'override':'auto','observed_at':int(time.time()),'supported':True}}
    if external:
        ari.vars['caller-1']['AGENT_CALL_ORIGIN'] = 'external'
        store.snapshot['profiles']['external']['permissions']['telephony.transfer.extension'] = 'deny'
    def adapter(binding, profile=None):
        value = OpenAIAdapter(binding)
        value.http_base = str(server.make_url('/v1/realtime')).rstrip('/')
        value.ws_base = str(server.make_url('/v1/realtime')).replace('http:', 'ws:')
        return value
    runtime = AgentRuntime(store=store, tools=ToolRegistry(), controller=ari, adapter_factory=adapter)
    ari.owner = runtime
    await runtime.start()
    app = FastAPI(); app.include_router(create_router(runtime, api_token='emulator-private-token'))
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://emulator')
    try:
        await runtime._admit('caller-1', ['caller','42','builtin_internal'])
        call = next(iter(runtime.calls.values()))
        await runtime._provider_stasis(call.local_id, ['provider',call.session_id])
        yield emulator, runtime, ari, call, client
    finally:
        await runtime.stop()
        await client.aclose()
        await server.close()


async def webhook(client, call, envelope=None, authorized=True):
    return await client.post('/api/agent/v1/provider-events/openai',json=envelope or signed_event(call),headers={'Authorization':'Bearer emulator-private-token'} if authorized else {})


async def connect(client, call):
    response = await webhook(client, call)
    assert response.status_code == 200 and response.json()['status'] == 'accepted'
    await eventually(lambda: call.state == CallState.CONVERSING)


@pytest.mark.asyncio
@pytest.mark.parametrize('kind,name', [('extension','Alice Example'),('queue','Customer Support'),('ivr','Sales Menu')])
async def test_named_transfer_through_signed_webhook_and_sideband(kind,name):
    async with harness() as (provider,runtime,ari,call,client):
        await connect(client,call)
        accepted = provider.accepted[0]
        assert 'Enabled capabilities' in accepted['instructions'] and name in accepted['instructions']
        assert 'Speak and respond using this language code or name: it' in accepted['instructions']
        assert 'Private Destination' not in accepted['instructions'] and 'private-secret-routing' not in accepted['instructions']
        tool = next(t for t in accepted['tools'] if t['name']=='nv_handoff_v1')
        assert name in tool['parameters']['properties']['destination_id']['description']
        await provider.tool('nv_handoff_v1',{'destination_id':kind+':100','reason':'Transfer to '+name},repeat=True)
        await eventually(lambda: call.terminal and not runtime.calls)
        assert ari.continued == [('caller-1',{'context':'satellite-agent-handoff','exten':'s','priority':1})]
        assert ari.vars['caller-1']['AGENT_HANDOFF_TARGET_ID'] == kind+':100'
        assert 'caller-1' not in ari.hungup
        assert call.local_id in ari.hungup


@pytest.mark.asyncio
async def test_all_read_tools_and_duplicate_delivery_return_one_result():
    async with harness() as (provider,runtime,ari,call,client):
        await connect(client,call)
        requests = [
            ('nv_find_destinations_v1',{'query':'Alice Example'},'directory'),
            ('nv_company_information_v1',{'fields':['company_name']},'company'),
            ('nv_opening_hours_v1',{'service_id':'support','date':'2026-10-04'},'calendar')]
        for name,args,invocation in requests:
            await provider.tool(name,args,invocation,'response-'+invocation,repeat=True)
            await eventually(lambda: provider.results(invocation))
            assert len(provider.results(invocation)) == 1
            assert provider.results(invocation)[0]['ok'] is True
        assert provider.results('directory')[0]['result']['matches'][0]['name'] == 'Alice Example'
        assert provider.results('company')[0]['result']['fields']['company_name'] == 'Test Company'
        assert provider.results('calendar')[0]['result']['status'] == 'known'
        assert not ari.continued


@pytest.mark.asyncio
@pytest.mark.parametrize('policy', ['disabled','denied','external'])
async def test_unadvertised_transfer_is_denied_even_if_model_requests_it(policy):
    async with harness(**{policy:True}) as (provider,runtime,ari,call,client):
        await connect(client,call)
        tool = next((t for t in provider.accepted[0]['tools'] if t['name']=='nv_handoff_v1'),None)
        if policy=='disabled':
            assert tool is None and 'nv_handoff_v1' not in provider.accepted[0]['instructions']
        else:
            assert 'extension:100' not in tool['parameters']['properties']['destination_id']['enum']
            assert 'Alice Example' not in tool['parameters']['properties']['destination_id']['description']
        await provider.tool('nv_handoff_v1',{'destination_id':'extension:100','reason':'Policy test'})
        await eventually(lambda: provider.results('tool-1'))
        assert provider.results('tool-1')[0]['ok'] is False
        assert not ari.continued and call.state == CallState.CONVERSING


@pytest.mark.asyncio
async def test_webhook_auth_signature_correlation_and_replay():
    async with harness() as (provider,runtime,ari,call,client):
        assert (await webhook(client,call,authorized=False)).status_code == 401
        bad = signed_event(call); bad['headers']['webhook-signature']='v1,invalid'
        assert (await webhook(client,call,bad)).status_code == 403
        assert (await webhook(client,call,signed_event(call,leg='wrong'))).json()['status']=='ignored'
        assert not provider.accepted
        await connect(client,call)
        assert (await webhook(client,call)).json()['status']=='duplicate'
        assert len(provider.accepted)==1


@pytest.mark.asyncio
@pytest.mark.parametrize('args', [{'destination_id':'extension:100'}, {'destination_id':'extension:999','reason':'Private'}, {'destination_id':'extension:100','reason':'OK','target':'arbitrary'}])
async def test_malformed_or_private_tool_arguments_cannot_transfer(args):
    async with harness() as (provider,runtime,ari,call,client):
        await connect(client,call)
        await provider.tool('nv_handoff_v1',args)
        await eventually(lambda: provider.results('tool-1'))
        assert provider.results('tool-1')[0]['ok'] is False and not ari.continued


@pytest.mark.asyncio
async def test_rejected_ari_continuation_keeps_caller_owned():
    async with harness() as (provider,runtime,ari,call,client):
        await connect(client,call)
        ari.fail_continue = True
        await provider.tool('nv_handoff_v1',{'destination_id':'extension:100','reason':'Failure test'})
        await eventually(lambda: provider.results('tool-1'))
        assert provider.results('tool-1')[0]['ok'] is False
        assert call.state == CallState.CONVERSING and not call.terminal
        assert not ari.hungup


def test_execution_prompt_does_not_mutate_saved_profile():
    profile={'prompt':'Original instructions','language':'it'}
    assert execution_profile(profile,[]) == profile
    enriched=execution_profile(profile,[{'name':'test','description':'Test capability','parameters':{'type':'object'}}])
    assert enriched['prompt'].startswith('Original instructions')
    assert profile['prompt']=='Original instructions'
