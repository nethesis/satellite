"""State and race checks for the built-in ARI voice owner."""

import asyncio
import base64
import hashlib
import hmac
import json
import time
from functools import wraps

import pytest

from agent.runtime import AgentRuntime, CallState


SECRET = "whsec_" + base64.b64encode(b"test-secret-for-agent").decode()


def test_default_tools_share_call_event_sequence():
    runtime = AgentRuntime()
    assert runtime.tools.events is runtime.events
    runtime.events.emit("call_started", run_id="run-1")
    runtime.tools.events.emit("tool.start", run_id="run-1")
    runtime.events.emit("call_ended", run_id="run-1")
    assert [event["sequence"] for event in runtime.events.recent()] == [1, 2, 3]


class FakeStore:
    revision = 7
    payload_hash = "a" * 64

    def __init__(self):
        profile = {"trunk_id": "1", "flow": "Internal", "greeting": "Hello",
                   "max_call_duration_seconds": 60,
                   "permissions": {"telephony.transfer.extension": "allow",
                                   "telephony.transfer.queue": "allow",
                                   "telephony.transfer.ivr": "allow",
                                   "directory.extensions": "allow",
                                   "directory.queues": "allow",
                                   "directory.ivrs": "allow"},
                   "tools": {"telephony.handoff": "enabled"}}
        self.snapshot = {
            "profiles": {"internal": profile,
                         "external": {"permissions": profile["permissions"],
                                      "tools": profile["tools"]}},
            "bindings": [{"id": "1", "provider": "openai", "runtime_owner": "builtin",
                          "trunk_name": "AgentTrunk_1", "provider_user": "u",
                          "provider_host": "sip.example", "api_key": "key",
                          "webhook_secret": SECRET}],
            "destinations": [{"id": 42, "agent_type": "builtin_internal",
                              "profile_key": "internal"}],
            "directory": [{"id": f"{kind}:100", "type": kind,
                           "target": {"context": f"from-{kind}", "exten": "100", "priority": 1},
                           "internal_allowed": True, "external_allowed": True}
                          for kind in ("extension", "queue", "ivr")],
            "calendars": {},
        }


class FakeAri:
    connected = True
    app = "satellite-agent"

    def __init__(self):
        self.vars = {"caller-1": {
            "AGENT_DESTINATION_ID": "42", "AGENT_TYPE": "builtin_internal",
            "AGENT_ROUTING_REVISION": "a" * 64, "AGENT_CALL_ORIGIN": "internal",
            "AGENT_ROLE": "caller", "AGENT_FLOW": "Internal"}}
        self.originated = []
        self.continued = []
        self.hungup = []
        self.bridges = []
        self.fail_continue = False

    async def start(self):
        pass

    async def stop(self):
        self.connected = False

    async def get_variable(self, channel_id, name):
        return self.vars.get(channel_id, {}).get(name)

    async def set_variable(self, channel_id, name, value):
        self.vars.setdefault(channel_id, {})[name] = value

    async def answer(self, channel_id):
        pass

    async def originate_local(self, session_id, channel_id, variables):
        assert session_id in self.owner.calls  # pending registration precedes originate
        self.originated.append((session_id, channel_id, variables))
        self.vars[channel_id] = {name.removeprefix("__"): value
                                 for name, value in variables.items()}

    async def list_channels(self):
        return ([{"id": "caller-1", "state": "Up",
                  "dialplan": {"app_name": "Stasis", "app_data": "satellite-agent,caller,42"}}]
                + [{"id": cid, "state": "Up"} for _, cid, _ in self.originated])

    async def create_bridge(self, bridge_id):
        self.bridges.append(bridge_id)

    async def add_to_bridge(self, bridge_id, channels):
        self.bridges.append((bridge_id, channels))

    async def destroy_bridge(self, bridge_id):
        pass

    async def continue_channel(self, channel_id, target=None):
        if self.fail_continue:
            raise RuntimeError("ARI HTTP 400")
        self.continued.append((channel_id, target))

    async def hangup(self, channel_id):
        self.hungup.append(channel_id)


class FakeAdapter:
    def __init__(self):
        self.accepted = []
        self.connected = []
        self.results = []
        self.responses = []
        self.hangups = []
        self.queue = asyncio.Queue()

    async def accept(self, call_id, profile, tools):
        self.accepted.append(call_id)

    async def connect(self, call_id):
        self.connected.append(call_id)

    async def send_result(self, invocation_id, result):
        self.results.append((invocation_id, result))

    async def respond(self, instructions=None, *, response_id=None):
        self.responses.append(instructions)

    async def close(self):
        await self.queue.put(None)

    async def hangup(self, call_id):
        self.hangups.append(call_id)

    async def events(self):
        while True:
            event = await self.queue.get()
            if event is None:
                return
            yield event


class FakeTools:
    def provider_tools(self, context):
        return []

    async def dispatch(self, name, arguments, invocation_id, context):
        return {"ok": True}


class FakeEvents:
    def __init__(self):
        self.items = []

    def emit(self, kind, **fields):
        self.items.append((kind, fields))


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setenv("API_TOKEN", "local-agent-token")
    runner = asyncio.Runner()
    ari = FakeAri()
    adapter = FakeAdapter()
    runtime = AgentRuntime(store=FakeStore(), tools=FakeTools(), events=FakeEvents(),
                           controller=ari, adapter_factory=lambda binding: adapter)
    ari.owner = runtime
    runner.run(runtime.start())
    yield runner, runtime, ari, adapter
    runner.run(runtime.stop())
    runner.close()


def run_async(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        setup_value = kwargs.get("setup") or args[0]
        return setup_value[0].run(fn(*args, **kwargs))
    return wrapper


async def admit(setup):
    _, runtime, ari, _ = setup
    await runtime._admit("caller-1", ["caller", "42", "builtin_internal"])
    assert len(runtime.calls) == 1
    return next(iter(runtime.calls.values()))


def signed_event(call, *, session=None, leg=None, destination=None, flow=None):
    event = {"id": "evt-1", "type": "realtime.call.incoming", "data": {"call_id": "provider-1", "sip_headers": [
        {"name": name, "value": value} for name, value in {
            "X-OS-Session-ID": session or call.session_id,
            "X-OS-Provider-Leg-ID": leg or call.leg_id,
            "X-OS-Agent-Role": "caller", "X-OS-FLOW": flow or call.flow,
            "X-OS-Agent-ID": call.destination_id,
            "X-OS-Destination-ID": destination or call.destination_id,
        }.items()]}}
    raw = json.dumps(event, separators=(",", ":")).encode()
    webhook_id = "msg-1"
    timestamp = str(int(time.time()))
    digest = hmac.new(b"test-secret-for-agent",
                      webhook_id.encode() + b"." + timestamp.encode() + b"." + raw,
                      hashlib.sha256).digest()
    return {"binding_id": "1", "raw_body": base64.b64encode(raw).decode(),
            "headers": {"webhook-id": webhook_id, "webhook-timestamp": timestamp,
                        "webhook-signature": "v1," + base64.b64encode(digest).decode()}}


async def ready(setup):
    _, runtime, ari, _ = setup
    call = await admit(setup)
    await runtime._provider_stasis(call.local_id, ["provider", call.session_id])
    assert (await runtime.provider_event("openai", signed_event(call))) == {"status": "accepted"}
    for _ in range(20):
        if call.state == CallState.CONVERSING:
            break
        await asyncio.sleep(0.01)
    assert call.state == CallState.CONVERSING
    assert len(ari.bridges) == 2
    return call


@run_async
async def test_pending_before_origination_and_exact_webhook_correlation(setup):
    _, runtime, ari, adapter = setup
    call = await admit(setup)
    assert ari.originated[0][2]["__AGENT_PROVIDER_LEG_ID"] == call.leg_id
    for changed in ({"session": "spoof"}, {"leg": "spoof"},
                    {"destination": "43"}, {"flow": "External"}):
        assert await runtime.provider_event("openai", signed_event(call, **changed)) == {"status": "ignored"}
    assert call.provider_call_id is None
    await runtime._provider_stasis(call.local_id, ["provider", call.session_id])
    assert await runtime.provider_event("openai", signed_event(call)) == {"status": "accepted"}
    assert await runtime.provider_event("openai", signed_event(call)) == {"status": "duplicate"}
    for _ in range(20):
        if call.state == CallState.CONVERSING:
            break
        await asyncio.sleep(0.01)
    assert call.state == CallState.CONVERSING
    assert adapter.accepted == ["provider-1"]


@run_async
async def test_setup_timeout_falls_back_only_once(setup):
    _, runtime, ari, _ = setup
    call = await admit(setup)
    await asyncio.gather(runtime._finish(call, "setup_timeout", True),
                         runtime._finish(call, "provider_hangup", True))
    assert ari.continued == [("caller-1", None)]
    assert ari.vars["caller-1"]["AGENT_EXIT_REASON"] == "fallback"
    assert call.local_id in ari.hungup


@pytest.mark.parametrize("kind", ["extension", "queue", "ivr"])
@run_async
async def test_handoff_commits_dispatcher_and_preserves_caller(setup, kind):
    _, runtime, ari, _ = setup
    call = await ready(setup)
    result = await runtime.handoff(call.session_id, {"destination_id": f"{kind}:100"})
    assert result["status"] == "handed_off"
    assert ari.continued == [("caller-1", {"context": "satellite-agent-handoff", "exten": "s", "priority": 1})]
    assert ari.vars["caller-1"]["AGENT_HANDOFF_TARGET_ID"] == f"{kind}:100"
    assert ari.vars["caller-1"]["AGENT_HANDOFF_ATTEMPT_ID"]
    assert 0 < ari.vars["caller-1"]["TIMEOUT(absolute)"] <= 30
    await runtime._finish(call, "late_hangup", True)
    assert "caller-1" not in ari.hungup
    assert call.session_id not in runtime.calls


@run_async
async def test_handoff_failure_keeps_conversation(setup):
    _, runtime, ari, _ = setup
    call = await ready(setup)
    ari.fail_continue = True
    with pytest.raises(Exception):
        await runtime.handoff(call.session_id, {"destination_id": "queue:100"})
    assert call.state == CallState.CONVERSING
    assert not call.terminal
    assert "caller-1" not in ari.hungup
