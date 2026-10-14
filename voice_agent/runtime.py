"""Basic Voice call ownership with pinned MariaDB snapshots and bounded handoff."""
import asyncio
import copy
import hmac
import hashlib
import logging
import re
import time
import uuid
from dataclasses import dataclass, field

from .monitoring import Monitoring
from .providers import create_adapter
from .providers.webhook import verify_webhook
from .storage import trusted_target
from .tools import ToolRegistry, WIRE_IDS


LOG=logging.getLogger(__name__)


@dataclass
class Call:
    session_id: str
    run_id: str
    leg_id: str
    caller_id: str
    snapshot: dict
    binding: dict
    linkedid: str | None
    deadline: float = 0
    provider_call_id: str | None = None
    adapter: object | None = None
    tools: object | None = None
    bridge_id: str | None = None
    media_connected: bool = False
    provider_answered: asyncio.Event = field(default_factory=asyncio.Event)
    transfer_done: asyncio.Event = field(default_factory=asyncio.Event)
    transfer_target: dict | None = None
    transfer_outcome: str | None = None
    transfer_token: str | None = None
    transfer_released: bool = False
    transfer_confirmed: bool = False
    terminal: bool = False
    admitted: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    tasks: set = field(default_factory=set)
    observed_responses: set = field(default_factory=set)


class Transfer:
    def __init__(self, runtime, call):
        self.runtime, self.call = runtime, call

    async def transfer(self, destination_id):
        call = self.call
        target = trusted_target(destination_id)
        async with call.lock:
            if call.terminal or call.transfer_target:
                return {'error': 'transfer_unavailable'}
            call.transfer_target = target
            call.transfer_token = uuid.uuid4().hex
        self.runtime.monitoring.emit(call.run_id, 'transfer_requested',tool_id='telephony.transfer',destination_id=destination_id)
        try:
            # Preserve a passive subscription after Stasis ends. The native wrapper
            # acknowledges the exact destination with this one-use private token.
            await self.runtime.controller.subscribe_channel(call.caller_id)
            await self.runtime.controller.set_variable(call.caller_id,'VOICE_AGENT_TRANSFER_TOKEN',call.transfer_token)
            if call.bridge_id:
                await self.runtime.controller.remove_from_bridge(call.bridge_id,[call.caller_id])
            await self.runtime.controller.continue_channel(call.caller_id, target)
            await asyncio.wait_for(call.transfer_done.wait(), timeout=10)
        except asyncio.CancelledError:
            self.runtime.monitoring.emit(call.run_id,'transfer_uncertain',error='transfer_cancelled')
            self.runtime.spawn(self.runtime.finish(call,'failed','transfer_cancelled',fallback=True),call)
            raise
        except Exception:
            self.runtime.monitoring.emit(call.run_id, 'transfer_uncertain', error='transfer_unconfirmed')
            await self.runtime.finish(call, 'failed', 'transfer_unconfirmed', fallback=True)
            return {'error': 'transfer_uncertain'}
        if call.transfer_outcome == 'transferred':
            self.runtime.monitoring.emit(call.run_id, 'transfer_committed', outcome='transferred',tool_id='telephony.transfer',destination_id=destination_id)
            await self.runtime.finish(call, 'transferred', fallback=False)
            return {'outcome': 'transferred'}
        self.runtime.monitoring.emit(call.run_id, 'transfer_failed', error='transfer_failed')
        await self.runtime.finish(call, 'failed', 'transfer_failed', fallback=False)
        return {'error': 'transfer_failed'}


class Runtime:
    def __init__(self, storage, controller, adapter_factory=create_adapter):
        self.storage, self.controller, self.adapter_factory = storage, controller, adapter_factory
        self.monitoring = Monitoring(storage)
        self.calls = {}
        self.admitting = set()
        self.admissions = {}
        self.tasks = set()
        self.controller.on_event = self.event
        self.controller.on_disconnect = self.disconnected
        self.stopping = False

    def spawn(self, coroutine, call=None):
        task = asyncio.create_task(coroutine)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        if call:
            call.tasks.add(task)
            task.add_done_callback(call.tasks.discard)
        return task

    async def start(self):
        try:
            await asyncio.wait_for(asyncio.to_thread(self.storage.recover), timeout=10)
        except Exception:
            self.monitoring.degraded = True
        await self.monitoring.start()
        await self.controller.start()
        self.spawn(self.expire_pending())

    async def stop(self):
        self.stopping = True
        await asyncio.gather(*(self.finish(c, 'interrupted', 'runtime_stopped', fallback=True) for c in list(self.calls.values())))
        for task in list(self.tasks):
            task.cancel()
        await asyncio.gather(*list(self.tasks), return_exceptions=True)
        await self.controller.stop()
        await self.monitoring.stop()

    async def event(self, event):
        kind = event.get('type')
        if kind == 'AgentAriConnected':
            # Only recover channels belonging to this app and these generated contexts.
            for channel in await self.controller.list_channels():
                dialplan = channel.get('dialplan') or {}
                context = dialplan.get('context', '')
                if dialplan.get('app_name') != 'Stasis' or not str(dialplan.get('app_data','')).startswith('voice-agent,'):
                    continue
                if context == 'voice-agent-provider':
                    await self.controller.hangup(channel['id'])
                    continue
                if not re.fullmatch(r'voice-agent-[a-z][a-z0-9_-]{0,47}', context):
                    continue
                if channel['id'] not in {c.caller_id for c in self.calls.values()}:
                    await self.controller.continue_channel(channel['id'], {'context':context,'extension':'s','label':'fallback'})
            return
        channel = event.get('channel') or {}
        channel_id = channel.get('id')
        if kind == 'StasisStart':
            args = event.get('args') or []
            if not isinstance(args,list) or not all(isinstance(value,str) for value in args):
                await self.controller.continue_channel(channel_id)
                return
            if len(args) == 2 and args[0] == 'caller':
                if channel_id not in self.admitting and not any(c.caller_id == channel_id for c in self.calls.values()):
                    if len(self.calls) + len(self.admitting) >= 32 or self.stopping:
                        await self.controller.continue_channel(channel_id)
                        return
                    self.admitting.add(channel_id)
                    self.admissions[channel_id] = self.spawn(self.admit(channel, args[1]))
            elif len(args) == 2 and args[0] == 'provider':
                call = self.calls.get(args[1])
                if call and channel_id == call.leg_id:
                    call.provider_answered.set()
                else:
                    await self.controller.hangup(channel_id)
            else:
                await self.controller.continue_channel(channel_id)
        elif kind == 'ChannelUserevent' and event.get('eventname') == 'VoiceAgentTransfer':
            call = next((c for c in self.calls.values() if channel_id == c.caller_id), None)
            body = event.get('userevent') or {}
            if call and not call.terminal and call.transfer_target and isinstance(body,dict):
                token = body.get('token')
                if isinstance(token,str) and token.isascii() and call.transfer_token and hmac.compare_digest(token,call.transfer_token) and body.get('destination_id') == 'extension:' + call.transfer_target['extension']:
                    call.transfer_confirmed = True
                    if call.transfer_released:
                        call.transfer_outcome = 'transferred'
                        call.transfer_done.set()
        elif kind in ('StasisEnd', 'ChannelDestroyed'):
            call = next((c for c in self.calls.values() if channel_id in (c.caller_id, c.leg_id)), None)
            if not call or call.terminal:
                if channel_id in self.admissions:
                    self.admissions[channel_id].cancel()
                return
            if channel_id == call.caller_id and call.transfer_target:
                if kind == 'StasisEnd':
                    call.transfer_released = True
                    if call.transfer_confirmed:
                        call.transfer_outcome = 'transferred'
                        call.transfer_done.set()
                elif not (call.transfer_released and call.transfer_confirmed):
                    call.transfer_outcome = 'failed'
                    call.transfer_done.set()
            elif channel_id == call.caller_id:
                if call.media_connected:
                    self.spawn(self.finish(call, 'completed', fallback=False), call)
                else:
                    self.spawn(self.finish(call,'interrupted','caller_cancelled_before_connect',fallback=False),call)
            else:
                reason='provider_disconnected'
                cause=event.get('cause')
                if isinstance(cause,int) and 0<cause<128: reason=f'provider_hangup_{cause}'
                LOG.warning('Voice agent phase=provider_channel error_code=%s',reason)
                self.spawn(self.finish(call, 'fallback', reason, fallback=True), call)

    async def admit(self, channel, agent_id):
        call = None
        stage='snapshot'
        try:
            if not re.fullmatch(r'[a-z][a-z0-9_-]{0,47}', agent_id):
                raise ValueError('invalid_agent')
            if (channel.get('dialplan') or {}).get('context') != 'voice-agent-' + agent_id:
                raise ValueError('invalid_entrypoint')
            snapshot = await asyncio.wait_for(asyncio.to_thread(self.storage.snapshot, agent_id), timeout=5)
            if len(self.monitoring.pending)>=192:
                raise ValueError('monitoring_capacity_exceeded')
            stage='binding'
            binding = await asyncio.wait_for(asyncio.to_thread(self.storage.binding, snapshot['provider_binding_id']), timeout=5)
            if binding['provider'] != snapshot['provider']:
                raise ValueError('provider_mismatch')
            linkedid = await self.controller.get_variable(channel['id'], 'CHANNEL(linkedid)')
            if linkedid is not None and (not isinstance(linkedid, str) or len(linkedid) > 128 or not linkedid.isascii()):
                linkedid = None
            call = Call(str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4()), channel['id'], copy.deepcopy(snapshot), copy.deepcopy(binding), linkedid)
            call.deadline = time.monotonic()+snapshot['max_call_duration_seconds']
            # Durable correlation exists before SIP origination or webhook acceptance.
            stage='run_storage'
            await asyncio.wait_for(asyncio.to_thread(self.storage.admit, call), timeout=10)
            call.admitted=True
            self.calls[call.session_id] = call
            if asyncio.current_task() in self.tasks:
                call.tasks.add(asyncio.current_task())
            stage='tool_catalog'
            definitions=await asyncio.wait_for(asyncio.to_thread(self.storage.tools),timeout=5)
            call.tools = ToolRegistry(snapshot, self.storage.targets, Transfer(self, call),definitions)
            self.monitoring.emit(call.run_id, 'admitted')
            user = binding.get('openai_project_id') if binding['provider'] == 'openai' else binding.get('grok_phone_number')
            if not isinstance(user, str) or not re.fullmatch(r'proj_[A-Za-z0-9_-]+|\+[1-9][0-9]{1,14}', user):
                raise ValueError('invalid_provider_user')
            variables = {'__AGENT_SESSION_ID':call.session_id,'__AGENT_PROVIDER_LEG_ID':call.leg_id,
                         '__VOICE_AGENT_ID':agent_id,'__AGENT_PROVIDER_HOST':'sip.api.openai.com' if binding['provider']=='openai' else 'sip.voice.x.ai','__AGENT_PROVIDER_USER':user,'__AGENT_PROVIDER_TRUNK':binding['trunk_name']}
            await self.controller.set_variable(call.caller_id, 'VOICE_AGENT_ID', agent_id)
            if call.terminal:
                return
            stage='origination'
            await self.controller.originate_local(call.session_id, call.leg_id, variables)
            self.spawn(self.deadline(call, 30, setup=True), call)
            self.spawn(self.deadline(call, max(0,call.deadline-time.monotonic())), call)
        except asyncio.CancelledError:
            if call:
                await self.finish(call, 'interrupted', 'setup_cancelled', fallback=True)
            raise
        except Exception as error:
            LOG.warning('Voice agent phase=%s error_class=%s',stage,type(error).__name__)
            if call:
                await self.finish(call, 'fallback', 'setup_'+stage+'_failed', fallback=True)
            else:
                try:
                    await self.controller.continue_channel(channel['id'])
                except Exception:
                    pass
        finally:
            self.admitting.discard(channel['id'])
            self.admissions.pop(channel['id'],None)

    async def expire_pending(self):
        while not self.stopping:
            await asyncio.sleep(10)
            try:
                await asyncio.wait_for(asyncio.to_thread(self.storage.expire_pending),timeout=5)
            except Exception:
                self.monitoring.degraded=True

    async def deadline(self, call, seconds, setup=False):
        await asyncio.sleep(seconds)
        if not call.terminal and (not setup or not call.media_connected):
            await self.finish(call, 'fallback', 'setup_timeout' if setup else 'duration_exceeded', fallback=True)

    async def webhook(self, provider, raw, headers):
        if provider not in ('openai', 'grok'):
            raise ValueError('invalid_provider')
        verified = []
        envelope = None
        for call in list(self.calls.values()):
            if call.terminal or call.binding['provider'] != provider:
                continue
            try:
                envelope = verify_webhook(call.binding['webhook_secret'], raw, headers)
                verified.append(call)
            except ValueError:
                pass
        if envelope is None:
            raise ValueError('invalid_webhook')
        if envelope.get('type') != 'realtime.call.incoming':
            return {'status':'ignored'}
        data = envelope.get('data') or {}
        if not isinstance(data, dict) or not isinstance(data.get('sip_headers'), list):
            return {'status':'ignored'}
        sip = {}
        if len(data['sip_headers'])>128: return {'status':'ignored'}
        correlation_headers={'x-os-session-id','x-os-provider-leg-id','x-os-agent-id'}
        for item in data['sip_headers']:
            if not isinstance(item, dict) or not isinstance(item.get('name'),str) or not isinstance(item.get('value'),str):
                return {'status':'ignored'}
            name = item['name'].lower()
            if name not in correlation_headers: continue
            if name in sip:
                return {'status':'ignored'}
            sip[name] = item['value']
        call = self.calls.get(sip.get('x-os-session-id'))
        call_id = data.get('call_id')
        if call not in verified or sip.get('x-os-provider-leg-id') != call.leg_id or sip.get('x-os-agent-id') != call.snapshot['agent_id'] or not isinstance(call_id,str) or not 1<=len(call_id)<=128 or not call_id.isascii() or any(ord(char)<32 or ord(char)==127 for char in call_id):
            return {'status':'ignored'}
        async with call.lock:
            if call.terminal:
                return {'status':'ignored'}
            if call.provider_call_id:
                return {'status':'duplicate' if hmac.compare_digest(call.provider_call_id,call_id) else 'ignored'}
            try:
                await asyncio.wait_for(asyncio.to_thread(self.storage.accept,call,call_id),timeout=5)
            except Exception:
                return {'status':'ignored'}
            call.provider_call_id = call_id
        self.spawn(self.converse(call), call)
        return {'status':'accepted'}

    async def converse(self, call):
        stage='provider_accept'
        try:
            call.adapter = self.adapter_factory(call.binding)
            profile = dict(call.snapshot['conversation'])
            profile['prompt'] = profile.pop('instructions')
            await asyncio.wait_for(call.adapter.accept(call.provider_call_id, profile, call.tools.advertised()), timeout=15)
            stage='provider_sideband'
            await asyncio.wait_for(call.adapter.connect(call.provider_call_id), timeout=15)
            self.monitoring.emit(call.run_id,'provider_connected')
            stage='provider_answer'
            await asyncio.wait_for(call.provider_answered.wait(),timeout=20)
            if call.terminal:
                return
            stage='media_bridge'
            call.bridge_id = str(uuid.uuid4())
            await self.controller.create_bridge(call.bridge_id)
            await self.controller.add_to_bridge(call.bridge_id, [call.caller_id,call.leg_id])
            await self.controller.set_variable(call.caller_id,'TIMEOUT(absolute)',max(1,int(call.deadline-time.monotonic())))
            call.media_connected = True
            self.monitoring.emit(call.run_id,'media_connected')
            self.spawn(self.greet(call,profile['greeting']),call)
            async for event in call.adapter.events():
                if call.terminal:
                    break
                if event.get('type')=='response_done' and event.get('status')=='completed':
                    response_id=event.get('response_id')
                    if isinstance(response_id,str) and len(response_id)<=128 and response_id not in call.observed_responses and len(call.observed_responses)<32:
                        call.observed_responses.add(response_id)
                        self.monitoring.emit(call.run_id,'response_completed',key='response:'+response_id)
                if event.get('type') == 'error':
                    raise ValueError('provider_error')
                if event.get('type') == 'tool_call':
                    if len(call.tasks)>=32 or len(self.tasks)>=512:
                        raise ValueError('tool_capacity_exceeded')
                    self.spawn(self.handle_tool(call,event),call)
            if not call.terminal:
                await self.finish(call,'fallback','provider_disconnected',fallback=True)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            status=getattr(error,'http_status',None)
            code=stage+'_failed'
            if isinstance(status,int) and 400<=status<600: code='provider_http_'+str(status)
            LOG.warning('Voice agent phase=%s error_code=%s error_class=%s',stage,code,type(error).__name__)
            await self.finish(call,'fallback',code,fallback=True)

    async def greet(self,call,greeting):
        try:
            await call.adapter.respond('Say this greeting: '+greeting if greeting else None)
        except Exception:
            await self.finish(call,'fallback','provider_unavailable',fallback=True)

    async def handle_tool(self,call,event):
        try:
            result=await call.tools.invoke(event.get('invocation_id'),event.get('name'),event.get('arguments'))
            if result.get('error'):
                kind='tool_denied' if result['error'] in ('tool_denied','destination_denied','invalid_arguments') else 'tool_failed'
                self.monitoring.emit(call.run_id,kind,error=result['error'],key=str(event.get('invocation_id')))
            else:
                self.monitoring.emit(call.run_id,'tool_completed',tool_id=WIRE_IDS.get(event['name'],event['name']),key=str(event.get('invocation_id')))
            if not call.terminal:
                await call.adapter.send_result(event['invocation_id'],result)
                await call.adapter.respond(response_id=event.get('response_id'))
        except Exception:
            await self.finish(call,'fallback','tool_execution_failed',fallback=True)

    async def finish(self, call, outcome, error=None, fallback=False):
        async with call.lock:
            if call.terminal:
                return
            call.terminal = True
        for task in list(call.tasks):
            if task is not asyncio.current_task():
                task.cancel()
        if call.tools:
            await call.tools.close(exclude=asyncio.current_task())
        if call.transfer_target:
            try:
                await self.controller.unsubscribe_channel(call.caller_id)
            except Exception:
                pass
        if fallback and call.transfer_released:
            # ARI cannot continue a channel outside Stasis. Fail closed if the
            # native owner could not be positively confirmed after release.
            try:
                await self.controller.hangup(call.caller_id)
            except Exception:
                pass
        elif fallback:
            try:
                if call.snapshot['contract_version']>=2:
                    await self.controller.continue_channel(call.caller_id,{'context':'voice-agent-fallback-'+hashlib.sha256((call.snapshot['agent_id']+'\0'+call.snapshot['fallback_destination_id']).encode()).hexdigest()[:24],'extension':'s','priority':1})
                    target=None
                else:
                    target = trusted_target(call.snapshot['fallback_destination_id'])
                if target is None:
                    pass
                elif target['extension'] in [row['id'].split(':')[1] for row in await asyncio.to_thread(self.storage.targets)]:
                    await self.controller.continue_channel(call.caller_id,target)
                else:
                    await self.controller.continue_channel(call.caller_id)
            except Exception:
                try:
                    await self.controller.continue_channel(call.caller_id)
                except Exception:
                    try:
                        await self.controller.hangup(call.caller_id)
                    except Exception:
                        pass
        try:
            await self.controller.hangup(call.leg_id)
            if call.bridge_id:
                await self.controller.destroy_bridge(call.bridge_id)
        except Exception:
            pass
        if call.adapter:
            try:
                await asyncio.wait_for(call.adapter.close(),timeout=3)
            except Exception:
                pass
        kind = outcome if outcome in ('completed','fallback','failed','interrupted') else 'completed'
        if call.admitted:
            self.monitoring.emit(call.run_id,kind,outcome=outcome,error=error)
        self.calls.pop(call.session_id,None)

    async def disconnected(self):
        await asyncio.gather(*(self.finish(call,'interrupted','ari_disconnected',fallback=True) for call in list(self.calls.values())))
