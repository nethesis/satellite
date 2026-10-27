"""Consent, isolation, one deadline and exact native-handoff boundaries."""
import asyncio
import copy
import json
import subprocess
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from test_contract import CONTRACT
from test_runtime import fixtures, admit
from voice_agent.contract import validate_snapshot
from voice_agent.consultation import Consultations
from voice_agent.consultation_phone import PrivateTools
from voice_agent.consultation_transport import ConsultationClient, private_url


def publication(role='public'):
    value = json.loads((CONTRACT/'published-v8.example.json').read_text())
    value.update(contract_version=9, role=role, chat_model=None,
        consultation={'mode':'direct','phone':False,'chat':False,'assistant_id':None})
    for binding in value['tool_bindings']:
        if binding['tool_id']=='pbx.transfer':
            binding['tool_version']=3
        if role=='consultation':
            binding['enabled']=False
    if role=='consultation':
        value['fallback_destination_id']='hangup'
    return value


def recipient(**changes):
    return dict(version=1, recipient_id='alice', mainextension='203',
        effective_presence='online', device_state='online', chat_allowed=True,
        observed_at=datetime.now(timezone.utc).isoformat(), **changes)


@pytest.mark.parametrize('role',['public','consultation'])
def test_v9_contract_is_identical_in_php_and_python(role):
    value=publication(role)
    cases=[(value,True)]
    invalid=copy.deepcopy(value);invalid['consultation']['mode']='attended';cases.append((invalid,False))
    invalid=copy.deepcopy(value);invalid['role']='arbitrary';cases.append((invalid,False))
    invalid=copy.deepcopy(value);invalid['chat_model']='https://private.example';cases.append((invalid,False))
    invalid=copy.deepcopy(value);invalid['tool_bindings'][1]['tool_version']=2;cases.append((invalid,False))
    if role=='consultation':
        invalid=copy.deepcopy(value);invalid['tool_bindings'][0]['enabled']=True;cases.append((invalid,False))
        invalid=copy.deepcopy(value);invalid['fallback_destination_id']='extension:203';cases.append((invalid,False))
    else:
        valid=copy.deepcopy(value);valid['consultation'].update(mode='attended',phone=True,assistant_id='internal');cases.append((valid,True))
        invalid=copy.deepcopy(valid);invalid['consultation']['assistant_id']=value['agent_id'];cases.append((invalid,False))
    for case,valid in cases:
        # Use the paired validator already exercised by the foundation suite.
        from test_contract import ROOT
        php=subprocess.run(['php',str(ROOT/'ns8-nethvoice/freepbx/tests/agents/validate.php')],input=json.dumps(case),text=True,capture_output=True)
        assert (php.returncode==0)==valid,php.stderr
        if valid:
            assert validate_snapshot(case)==case
        else:
            with pytest.raises(ValueError):validate_snapshot(case)


def test_v9_assets_are_mirrored_without_modifying_v8():
    from pathlib import Path
    root=Path(__file__).resolve().parents[2]/'voice_agent'
    for name in ('published-v9.schema.json','tools-v6.json','consultation-tools-v1.json'):
        assert (root/name).read_bytes()==(CONTRACT/name).read_bytes()


async def prepared(phone=True,chat=False):
    runtime,storage,controller,_=fixtures()
    call=await admit(runtime)
    call.snapshot=publication()
    call.snapshot['tool_bindings'][1]['config'].update(extensions_enabled=True,extension_allowlist=['extension:203'])
    call.snapshot['consultation'].update(mode='attended',phone=phone,chat=chat,assistant_id='internal')
    call.assistant_snapshot=publication('consultation')
    call.assistant_binding={'provider':'openai','api_key':'synthetic'}
    call.adapter=SimpleNamespace(consultation_say=AsyncMock(),close=AsyncMock())
    call.bridge_id='caller-only-bridge'
    call.media_connected=True
    client=SimpleNamespace(recipient=AsyncMock(return_value=recipient()),send=AsyncMock(return_value='a'*32),replies=AsyncMock(return_value=[]),close_attempt=AsyncMock())
    runtime.consultations=Consultations(runtime,client)
    return runtime,storage,controller,call,client


@pytest.mark.asyncio
async def test_internal_admission_is_rejected_before_provider_usage():
    runtime,storage,controller,_=fixtures()
    storage.snapshot.return_value=publication('consultation')
    await runtime.admit({'id':'untrusted','dialplan':{'context':'voice-agent-basic_voice'}},'basic_voice')
    storage.binding.assert_not_called();storage.admit.assert_not_called()
    controller.originate_local.assert_not_awaited()


@pytest.mark.asyncio
async def test_grants_recipient_overlap_and_no_automatic_same_person_retry():
    runtime,storage,controller,call,client=await prepared()
    coordinator=runtime.consultations
    try:
        assert await coordinator.start(call,'extension:204','reason')=={'error':'destination_denied'}
        client.recipient.assert_not_awaited()
        assert (await coordinator.start(call,'extension:203','reason'))['outcome']=='consulting'
        c=call.consultation
        assert (await coordinator.start(call,'extension:203','reason'))['outcome']=='unavailable'
        await coordinator.cleanup(c)
        assert (await coordinator.start(call,'extension:203','reason'))['outcome']=='unavailable'
        controller.originate_consult.assert_not_awaited()
    finally:await runtime.finish(call,'completed')


@pytest.mark.asyncio
@pytest.mark.parametrize('presence',['dnd','busy','ringing','offline','voicemail','cellphone','callforward'])
async def test_unavailable_presence_never_rings_or_blind_transfers(presence):
    runtime,storage,controller,call,client=await prepared()
    client.recipient.return_value['effective_presence']=presence
    try:
        await runtime.consultations.start(call,'extension:203','reason')
        c=call.consultation
        await runtime.consultations.begin(call)
        await c.task
        controller.originate_consult.assert_not_awaited()
        controller.continue_channel.assert_not_awaited()
        assert not call.terminal and call.consultation is None
        controller.remove_from_bridge.assert_awaited_with('caller-only-bridge',[call.caller_id])
        controller.add_to_bridge.assert_awaited_with('caller-only-bridge',[call.caller_id])
    finally:await runtime.finish(call,'completed')


class Phone:
    rings=0
    decision='decline'
    def __init__(self,runtime,coordinator,c):
        self.c=c;self.channel='answered-colleague';self.answered=True;self.private=None
        self.detach=AsyncMock();self.close=AsyncMock();self.answer=AsyncMock()
    async def ring(self,seconds,connect_only=False):
        type(self).rings+=1
        assert 0<seconds<=20
        if not connect_only:self.c.responses.put_nowait({'decision':self.decision,'text':''})
        return True


@pytest.mark.asyncio
async def test_decline_is_terminal_and_does_not_fall_back_to_chat():
    runtime,storage,controller,call,client=await prepared(chat=True)
    runtime.consultations.phone_factory=Phone;Phone.decision='decline';Phone.rings=0
    try:
        await runtime.consultations.start(call,'extension:203','reason')
        c=call.consultation;await runtime.consultations.begin(call);await c.task
        assert c.state=='declined' and Phone.rings==1
        client.send.assert_not_awaited()
        assert not call.terminal
    finally:await runtime.finish(call,'completed')


@pytest.mark.asyncio
async def test_phone_acceptance_requires_same_native_bridge_and_both_releases_without_redial():
    runtime,storage,controller,call,client=await prepared()
    runtime.consultations.phone_factory=Phone;Phone.decision='accept';Phone.rings=0
    async def handoff(c,channel):
        assert channel=='answered-colleague'
        c.released={call.caller_id,channel}
        controller.list_bridges.return_value=[{'bridge_class':'basic','channels':[call.caller_id,channel]}]
    controller.native_handoff.side_effect=handoff
    try:
        await runtime.consultations.start(call,'extension:203','reason')
        c=call.consultation;await runtime.consultations.begin(call);await c.task
        assert c.transferred and Phone.rings==1 and call.terminal
        controller.native_handoff.assert_awaited_once()
        controller.hangup.assert_awaited_once_with(call.leg_id)
        assert not runtime.consultations.recipients
    finally:await runtime.finish(call,'completed')


@pytest.mark.asyncio
async def test_one_clarification_answer_keeps_original_deadline_and_private_role():
    runtime,storage,controller,call,client=await prepared()
    try:
        await runtime.consultations.start(call,'extension:203','reason')
        c=call.consultation;c.deadline=time.monotonic()+5;c.state='phone'
        c.phone=Phone(runtime,runtime.consultations,c)
        original=c.deadline
        tools=PrivateTools(runtime.consultations,c)
        assert [t['name'] for t in tools.advertised()]==['nv_consultation_decide_v1']
        assert await tools.invoke('bad','pbx.transfer',{})=={'error':'tool_denied'}
        task=asyncio.create_task(runtime.consultations.decision(c,{'decision':'question','text':'Which document?'}))
        for _ in range(30):
            if c.state=='clarification':break
            await asyncio.sleep(.001)
        assert await runtime.consultations.answer(call,'The signed copy.')=={'outcome':'answer_submitted'}
        await task
        assert c.deadline==original and c.clarified
        c.phone.answer.assert_awaited_once_with('The signed copy.')
        assert await runtime.consultations.submit(c,'question','Again?','phone')=={'error':'clarification_limit'}
        assert await runtime.consultations.answer(call,'Late')=={'error':'consultation_unavailable'}
        assert await runtime.consultations.submit(c,'accept','','chat')=={'error':'consultation_unavailable'}
    finally:await runtime.finish(call,'completed')


@pytest.mark.asyncio
async def test_shared_deadline_and_component_failure_preserve_caller():
    runtime,storage,controller,call,client=await prepared()
    call.deadline=time.monotonic()+.02
    class Stalled(Phone):
        async def ring(self,*args):await asyncio.Event().wait()
    runtime.consultations.phone_factory=Stalled
    try:
        await runtime.consultations.start(call,'extension:203','reason')
        c=call.consultation;await runtime.consultations.begin(call);await c.task
        assert c.state=='timed_out' and not call.terminal
        assert not runtime.consultations.recipients and call.consultation is None
    finally:await runtime.finish(call,'completed')


@pytest.mark.parametrize('url',['https://127.0.0.1:3','http://example.org:3','http://127.0.0.1:3/path','http://user@127.0.0.1:3','http://127.0.0.1:3?token=secret'])
def test_private_transports_reject_nonlocal_or_injected_urls(url):
    with pytest.raises(ValueError):private_url(url)


@pytest.mark.asyncio
@pytest.mark.parametrize('failure',['no_answer','provider_failure'])
async def test_phone_failure_uses_authorized_chat_and_preserves_caller(failure):
    runtime,storage,controller,call,client=await prepared(chat=True)
    coordinator=runtime.consultations
    class FailedPhone(Phone):
        async def ring(self,*args):
            if failure=='provider_failure':raise RuntimeError('synthetic provider outage')
            return False
    coordinator.phone_factory=FailedPhone
    async def chat(c):
        assert c.state=='chat'
        await coordinator.resolve(c,'replied','Please call tomorrow.')
    coordinator.chat=AsyncMock(side_effect=chat)
    try:
        await coordinator.start(call,'extension:203','reason')
        c=call.consultation;await coordinator.begin(call);await c.task
        assert c.state=='replied' and not call.terminal
        coordinator.chat.assert_awaited_once()
        assert not coordinator.recipients
    finally:await runtime.finish(call,'completed')


@pytest.mark.asyncio
async def test_dnd_chat_acceptance_requires_new_available_state_before_ringing():
    runtime,storage,controller,call,client=await prepared(phone=False,chat=True)
    coordinator=runtime.consultations
    try:
        await coordinator.start(call,'extension:203','reason')
        c=call.consultation;c.deadline=time.monotonic()+5;c.state='chat'
        client.recipient.return_value['effective_presence']='dnd'
        coordinator.phone=AsyncMock()
        assert not await coordinator.decision(c,{'decision':'accept','text':''})
        coordinator.phone.assert_not_awaited()
        client.send.assert_awaited_once()
        assert c.state=='chat' and not call.terminal
        client.recipient.return_value['effective_presence']='online'
        await coordinator.decision(c,{'decision':'accept','text':''})
        coordinator.phone.assert_awaited_once_with(c,connect_only=True)
    finally:await runtime.finish(call,'completed')


@pytest.mark.asyncio
async def test_phone_tool_rejects_acceptance_before_colleague_speech_and_duplicate_invocation():
    runtime,storage,controller,call,client=await prepared()
    try:
        await runtime.consultations.start(call,'extension:203','reason')
        c=call.consultation;c.deadline=time.monotonic()+5;c.state='phone'
        tools=PrivateTools(runtime.consultations,c)
        assert await tools.invoke('first','nv_consultation_decide_v1',{'decision':'accept','text':''})=={'error':'recipient_not_heard'}
        c.colleague_spoke=True
        assert await tools.invoke('first','nv_consultation_decide_v1',{'decision':'accept','text':''})=={'error':'duplicate_decision'}
        assert await tools.invoke('next','nv_consultation_decide_v1',{'decision':'accept','text':''})=={'outcome':'decision_submitted'}
    finally:await runtime.finish(call,'completed')


@pytest.mark.asyncio
async def test_capacity_includes_pending_public_admissions():
    runtime,storage,controller,call,client=await prepared()
    try:
        runtime.admitting={str(i) for i in range(31)}
        assert (await runtime.consultations.start(call,'extension:203','reason'))['outcome']=='unavailable'
        assert not runtime.consultations.recipients
    finally:await runtime.finish(call,'completed')


@pytest.mark.asyncio
async def test_uncertain_native_release_never_resumes_or_hangs_up_human_legs():
    runtime,storage,controller,call,client=await prepared()
    coordinator=runtime.consultations
    try:
        await coordinator.start(call,'extension:203','reason')
        c=call.consultation;c.deadline=time.monotonic()+5;c.state='phone';c.phone=Phone(runtime,coordinator,c)
        async def uncertain(c,channel):
            c.native_started=True
            raise RuntimeError('connection lost after release')
        controller.native_handoff.side_effect=uncertain
        with pytest.raises(RuntimeError):await coordinator.connect(c)
        assert c.uncertain and call.terminal
        assert all(args.args[0] not in (call.caller_id,c.phone.channel) for args in controller.hangup.await_args_list)
        controller.continue_channel.assert_not_awaited()
        controller.add_to_bridge.assert_not_awaited()
    finally:await runtime.finish(call,'completed')


@pytest.mark.asyncio
async def test_restart_reconciles_only_persisted_owned_stasis_consultations():
    runtime,storage,controller,_=fixtures()
    storage.consultation_ownership.return_value={'old-session'}
    storage.media_ownership.return_value={}
    def channel(id,app='Stasis',name='Local/203@voice-agent-consult-00000001;1'):
        return {'id':id,'name':name,'dialplan':{'context':'voice-agent-consult','app_name':app,'app_data':'voice-agent,consult,old-session,attempt'}}
    controller.list_channels.return_value=[channel('owned'),channel('human','Bridge'),channel('foreign'),channel('secondary',name='Local/203@voice-agent-consult-00000001;2')]
    async def variable(id,name):
        return {'AGENT_ROLE':'consult','AGENT_SESSION_ID':'old-session' if id!='foreign' else 'unowned','AGENT_CONSULT_ATTEMPT':'a'*32}.get(name)
    controller.get_variable.side_effect=variable
    await runtime.event({'type':'AgentAriConnected'})
    controller.hangup.assert_awaited_once_with('owned')
    controller.continue_channel.assert_not_awaited()


@pytest.mark.asyncio
async def test_primary_input_activity_is_content_free_and_silence_is_not_consent():
    from voice_agent.providers.live_primary import OpenAILiveWebSocketSession
    session=OpenAILiveWebSocketSession({'api_key':'synthetic'})
    session._send=AsyncMock()
    try:
        await session.send_audio(b'\xff'*160)
        assert not any(e['type']=='input_speech' for e in session._queue._queue)
        await session.send_audio(b'\x80'*160)
        signals=[e for e in session._queue._queue if e['type']=='input_speech']
        assert signals==[{'type':'input_speech'}]
    finally:await session.close()



@pytest.mark.asyncio
async def test_parent_cancellation_cannot_interrupt_owned_resource_cleanup():
    runtime,storage,controller,call,client=await prepared()
    coordinator=runtime.consultations
    release=asyncio.Event();entered=asyncio.Event()
    async def close(**kwargs):
        entered.set();await release.wait()
    try:
        await coordinator.start(call,'extension:203','reason')
        c=call.consultation;c.phone=Phone(runtime,coordinator,c);c.phone.close=close
        task=asyncio.create_task(coordinator.cleanup(c))
        await entered.wait();task.cancel();await asyncio.gather(task,return_exceptions=True)
        assert not c.cleanup_task.done()
        release.set();await coordinator.cleanup(c)
        assert call.consultation is None and not coordinator.recipients and c.cleanup_task.done()
    finally:
        release.set();await runtime.finish(call,'completed')



@pytest.mark.asyncio
async def test_active_consultation_blocks_competing_direct_queue_transfer():
    from voice_agent.runtime import Transfer
    runtime,storage,controller,call,client=await prepared()
    try:
        await runtime.consultations.start(call,'extension:203','reason')
        assert await Transfer(runtime,call).transfer('queue:600',reason='different path')=={'error':'transfer_unavailable'}
        controller.continue_channel.assert_not_awaited()
    finally:await runtime.finish(call,'completed')



@pytest.mark.asyncio
async def test_correlated_native_failure_rejoins_receptionist_media():
    runtime,storage,controller,call,client=await prepared()
    coordinator=runtime.consultations
    try:
        await coordinator.start(call,'extension:203','reason')
        c=call.consultation;c.deadline=time.monotonic()+5;c.state='phone';c.phone=Phone(runtime,coordinator,c)
        async def failed(c,channel):
            c.native_started=True;c.native_failure.set()
        controller.native_handoff.side_effect=failed
        await coordinator.connect(c)
        assert c.state=='unavailable' and not call.terminal and not c.releasing and not c.held
        controller.add_to_bridge.assert_awaited_once_with('caller-only-bridge',[call.caller_id])
        assert controller.set_variable.await_args.args[:2]==(call.caller_id,'TIMEOUT(absolute)')
        assert all(args.args[0] != call.caller_id for args in controller.hangup.await_args_list)
    finally:await runtime.finish(call,'completed')



@pytest.mark.asyncio
async def test_partial_native_preparation_unsubscribes_only_owned_channels():
    from voice_agent.ari import AriController
    runtime,storage,controller,call,client=await prepared()
    coordinator=runtime.consultations
    ari=AriController('http://127.0.0.1:8088','test','synthetic')
    ari.request=AsyncMock(return_value={'state':'Up','name':'Local/203@voice-agent-consult-00000001;1'})
    ari.subscribe_channel=AsyncMock(side_effect=[None,RuntimeError('synthetic disconnect')])
    ari.set_variable=AsyncMock()
    try:
        await coordinator.start(call,'extension:203','reason')
        c=call.consultation
        with pytest.raises(RuntimeError):await ari.native_handoff(c,'answered-colleague')
        assert c.subscriptions=={call.caller_id} and not c.native_started
        await coordinator.cleanup(c)
        controller.unsubscribe_channel.assert_awaited_once_with(call.caller_id)
    finally:await runtime.finish(call,'completed')



@pytest.mark.asyncio
async def test_failed_peer_release_recovers_only_when_caller_release_was_never_requested():
    runtime,storage,controller,call,client=await prepared()
    coordinator=runtime.consultations
    try:
        await coordinator.start(call,'extension:203','reason')
        c=call.consultation;c.deadline=time.monotonic()+5;c.state='phone';c.phone=Phone(runtime,coordinator,c)
        async def failed(c,channel):
            c.native_started=True
            raise RuntimeError('peer release acknowledgement lost')
        controller.native_handoff.side_effect=failed
        controller.handoff_state.return_value='caller_owned'
        await coordinator.connect(c)
        assert c.state=='unavailable' and not c.uncertain and not call.terminal
        controller.handoff_state.assert_awaited_once_with(c,'answered-colleague')
        controller.add_to_bridge.assert_awaited_once_with('caller-only-bridge',[call.caller_id])
    finally:await runtime.finish(call,'completed')



@pytest.mark.asyncio
@pytest.mark.parametrize('decision_before_reply',[False,True])
async def test_chat_opening_uses_internal_language_prompt_and_cannot_decide(monkeypatch,decision_before_reply):
    from voice_agent.consultation_transport import ChatConversation
    runtime,storage,controller,call,client=await prepared(chat=True)
    value=publication('consultation');value['chat_model']='synthetic-text-model';value['conversation']['language']='fr'
    value['conversation']['instructions']='Configured private assistant prompt.'
    captured=[]
    output={'role':'assistant','content':'Bonjour. Voulez-vous accepter cet appel ?'}
    if decision_before_reply:output['tool_calls']=[{'function':{'name':'nv_consultation_decide_v1','arguments':'{"decision":"accept","text":""}'}}]
    class Response:
        status=200
        content=SimpleNamespace(readexactly=AsyncMock(side_effect=asyncio.IncompleteReadError(json.dumps({'choices':[{'message':output}],'usage':{'prompt_tokens':10,'completion_tokens':5,'total_tokens':15}}).encode(),65537)))
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
    class Session:
        def __init__(self,**kwargs):pass
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
        def post(self,url,**kwargs):captured.append(kwargs['json']);return Response()
    monkeypatch.setattr('voice_agent.consultation_transport.aiohttp.ClientSession',Session)
    try:
        await runtime.consultations.start(call,'extension:203','A concise reason')
        c=call.consultation;c.deadline=time.monotonic()+5
        conversation=ChatConversation(value,{'provider':'openai','api_key':'synthetic'},c)
        if decision_before_reply:
            with pytest.raises(ValueError,match='invalid_chat_opening'):await conversation.opening()
        else:assert await conversation.opening()==output['content']
        assert captured[0]['model']=='synthetic-text-model' and 'tools' not in captured[0]
        assert 'Configured private assistant prompt.' in captured[0]['messages'][0]['content']
        assert 'configured language fr' in captured[0]['messages'][0]['content']
        assert c.responses.empty() and c.state=='pending'
    finally:await runtime.finish(call,'completed')



@pytest.mark.asyncio
async def test_private_json_waits_for_split_network_body_and_bounds_response_size():
    import aiohttp
    from voice_agent.consultation_transport import bounded_json
    stream=aiohttp.StreamReader(MagicMock(connected=True,_reading_paused=False),limit=65537)
    stream.feed_data(b'{"version":')
    task=asyncio.create_task(bounded_json(SimpleNamespace(content=stream)))
    await asyncio.sleep(0)
    assert not task.done()
    stream.feed_data(b'1}');stream.feed_eof()
    assert await task=={'version':1}
    oversized=aiohttp.StreamReader(MagicMock(connected=True,_reading_paused=False),limit=65537)
    oversized.feed_data(b'x'*65537);oversized.feed_eof()
    with pytest.raises(ValueError):await bounded_json(SimpleNamespace(content=oversized))



@pytest.mark.asyncio
async def test_shutdown_during_native_release_never_starts_competing_fallback():
    runtime,storage,controller,call,client=await prepared()
    coordinator=runtime.consultations
    coordinator.phone_factory=Phone;Phone.decision='accept'
    entered=asyncio.Event()
    async def handoff(c,channel):
        c.native_started=True;c.caller_release_requested=True;entered.set()
        await asyncio.Event().wait()
    controller.native_handoff.side_effect=handoff
    controller.handoff_state.return_value='bridged'
    try:
        await coordinator.start(call,'extension:203','reason')
        c=call.consultation;await coordinator.begin(call)
        await asyncio.wait_for(entered.wait(),1)
        await runtime.finish(call,'interrupted','runtime_stopped',fallback=True)
        assert c.uncertain and call.terminal and not coordinator.recipients
        controller.continue_channel.assert_not_awaited()
        assert all(args.args[0] not in (call.caller_id,c.phone.channel) for args in controller.hangup.await_args_list)
    finally:await runtime.finish(call,'completed')


@pytest.mark.asyncio
async def test_pending_consultation_blocks_competing_agent_hangup():
    from voice_agent.runtime import Transfer
    runtime,storage,controller,call,client=await prepared()
    call.snapshot['tool_bindings'][3]['enabled']=True
    try:
        await runtime.consultations.start(call,'extension:203','reason')
        assert await Transfer(runtime,call).hangup('conversation_complete')=={'error':'hangup_unavailable'}
        assert not call.ending_by_agent
    finally:await runtime.finish(call,'completed')
