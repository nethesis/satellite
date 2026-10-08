import asyncio
import copy
from functools import wraps
import time

import pytest

from agent.configuration import ConfigurationStore, canonical_hash
from agent.models import InvalidConfiguration, RevisionConflict
from agent.tools import ToolRegistry


def run_async(fn):
    @wraps(fn)
    def wrapper():
        return asyncio.run(fn())
    return wrapper


def snapshot():
    permissions = {
        "directory.extensions": "allow", "directory.queues": "allow",
        "directory.ivrs": "allow", "company.public_information": "allow",
        "calendar.opening_hours": "allow", "telephony.transfer.extension": "allow",
        "telephony.transfer.queue": "allow", "telephony.transfer.ivr": "allow",
    }
    tools = {"directory.find_destinations": "enabled",
             "company.get_information": "enabled",
             "calendar.get_opening_hours": "enabled",
             "telephony.handoff": "enabled"}
    def profile(flow):
        return {"trunk_id": "7", "flow": flow, "model": "model", "voice": "voice",
                "language": "en", "greeting": "hello", "prompt": "prompt",
                "permissions": copy.deepcopy(permissions), "tools": copy.deepcopy(tools),
                "max_call_duration_seconds": 600,
                "fallback_destination": "queueexit-3,${EXTEN},1",
                "company": {"company_name": "Example"},
                "calendar_services": {"support": "4"}}
    return {
        "profiles": {"internal": profile("Internal"), "external": profile("External")},
        "bindings": [{"id": "7", "provider": "openai", "runtime_owner": "builtin",
                      "trunk_name": "AgentTrunk_7", "provider_user": "u",
                      "provider_host": "example.org", "api_key": "secret",
                      "webhook_secret": "webhook"}],
        "destinations": [{"id": 1, "agent_type": "builtin_internal", "profile_key": "internal",
                          "fallback_destination": "ext-local,203,1"},
                         {"id": 2, "agent_type": "builtin_external", "profile_key": "external",
                          "fallback_destination": None},
                         {"id": 3, "agent_type": "cleverai", "profile_key": None,
                          "fallback_destination": None}],
        "directory": [{"id": "extension:203", "type": "extension", "name": "Alice",
                       "description": "Sales", "synonyms": [], "internal_allowed": True,
                       "external_allowed": False,
                       "target": {"context": "ext-local", "exten": "203", "priority": 1}},
                      {"id": "queue:600", "type": "queue", "name": "Support",
                       "description": "Help desk", "synonyms": [], "internal_allowed": True,
                       "external_allowed": True,
                       "target": {"context": "ext-queues", "exten": "600", "priority": 1}}],
        "calendars": {"4": {"timezone": "Europe/Rome",
                            "rules": ["22:00-02:00|mon-fri|*|*"],
                            "override": "auto", "observed_at": int(time.time()),
                            "supported": True}},
    }


def envelope(payload, revision):
    return {"schema_version": 1, "revision": revision,
            "payload_hash": canonical_hash(payload), "payload": payload}


def test_snapshot_revisions_atomic_and_nonsecret_state(tmp_path):
    path = tmp_path / "state.json"
    store = ConfigurationStore(str(path))
    first = snapshot()
    ack = store.apply(envelope(first, 1))
    assert ack["revision"] == 1
    assert "secret" not in path.read_text()
    assert store.snapshot["bindings"][0]["api_key"] == "secret"
    changed = snapshot()
    changed["profiles"]["external"]["prompt"] = "different"
    with pytest.raises(RevisionConflict):
        store.apply(envelope(changed, 1))
    invalid = snapshot()
    invalid["profiles"]["internal"]["trunk_id"] = "missing"
    with pytest.raises(InvalidConfiguration):
        store.apply(envelope(invalid, 2))
    assert store.snapshot == first and store.revision == 1
    restarted = ConfigurationStore(str(path))
    assert restarted.snapshot is None
    restarted.apply(envelope(first, 1))  # equal revision rehydrates credentials
    assert restarted.snapshot == first
    restarted.reset_for_restore()
    assert restarted.revision == 0 and restarted.snapshot is None


def test_context_refresh_keeps_routing_hash(tmp_path):
    store = ConfigurationStore(str(tmp_path / "state.json"))
    payload = snapshot()
    store.apply(envelope(payload, 1))
    fresh = copy.deepcopy(payload["calendars"])
    fresh["4"]["override"] = "closed"
    store.refresh_context({"directory": payload["directory"], "calendars": fresh,
                           "payload_hash": store.payload_hash})
    assert store.live_context["calendars"]["4"]["override"] == "closed"
    assert store.snapshot["calendars"]["4"]["override"] == "auto"
    assert store.payload_hash == canonical_hash(payload)
    with pytest.raises(RevisionConflict):
        store.refresh_context({"directory": payload["directory"], "calendars": fresh,
                               "payload_hash": "0" * 64})


def _context(payload, *, origin="internal", agent_id="internal", voice=None, handoff=None):
    return {"run_id": "run-1", "agent_id": agent_id, "origin": origin,
            "profile": payload["profiles"][agent_id],
            "external_profile": payload["profiles"]["external"],
            "permissions": payload["profiles"][agent_id]["permissions"],
            "directory": payload["directory"], "calendars": payload["calendars"],
            "voice": voice, "handoff": handoff}


@run_async
async def test_external_ceiling_and_visibility_for_internal_destination():
    payload = snapshot()
    payload["profiles"]["external"]["permissions"]["directory.extensions"] = "deny"
    payload["profiles"]["external"]["permissions"]["telephony.transfer.extension"] = "deny"
    registry = ToolRegistry()
    context = _context(payload, origin="external")
    directory = await registry.dispatch("nv_find_destinations_v1", {"query": "Alice"}, "i1", context)
    assert directory == {"ok": True, "result": {"matches": []}}
    called = []
    async def handoff(destination, reason):
        called.append(destination)
    context["voice"] = {"call_session_id": "s", "participant_role": "caller", "state": "CONVERSING"}
    context["handoff"] = handoff
    denied = await registry.dispatch("nv_handoff_v1",
                                     {"destination_id": "extension:203", "reason": "asked"}, "i2", context)
    assert denied["error"]["code"] == "permission_denied" and called == []


@run_async
async def test_deduplicated_handoff_and_argument_validation():
    payload = snapshot()
    calls = []
    async def handoff(destination, reason):
        await asyncio.sleep(0.01)
        calls.append((destination, reason))
    context = _context(payload, voice={"call_session_id": "s", "participant_role": "caller",
                                       "state": "CONVERSING"}, handoff=handoff)
    registry = ToolRegistry()
    args = {"destination_id": "queue:600", "reason": "support"}
    first, second = await asyncio.gather(
        registry.dispatch("nv_handoff_v1", args, "same", context),
        registry.dispatch("nv_handoff_v1", args, "same", context))
    assert first == second and first["ok"] is True and calls == [("queue:600", "support")]
    conflict = await registry.dispatch("nv_handoff_v1",
                                       {"destination_id": "queue:600", "reason": "other"}, "same", context)
    assert conflict["error"]["code"] == "invalid_invocation"
    invalid = await registry.dispatch("nv_handoff_v1", {**args, "context": "evil"}, "new", context)
    assert invalid["error"]["code"] == "invalid_invocation"


@run_async
async def test_company_and_hours_without_voice_objects():
    payload = snapshot()
    registry = ToolRegistry()
    context = _context(payload)
    company = await registry.dispatch("nv_company_information_v1", {"fields": ["company_name"]}, "c", context)
    assert company == {"ok": True, "result": {"fields": {"company_name": "Example"}}}
    hours = await registry.dispatch("nv_opening_hours_v1",
                                    {"service_id": "support", "date": "2026-10-06"}, "h", context)
    assert hours["result"]["status"] == "known"
    assert hours["result"]["opens_at"] == "00:00" or hours["result"]["opens_at"] == "22:00"
    payload["calendars"]["4"]["observed_at"] = 1
    stale = await registry.dispatch("nv_opening_hours_v1",
                                    {"service_id": "support", "date": "2026-10-06"}, "h2", context)
    assert stale["result"]["status"] == "unknown"
