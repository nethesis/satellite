import asyncio
import base64
import hashlib
import hmac
import importlib.util
import json
import sys
import time
import types

import pytest

if importlib.util.find_spec("aiohttp") is None:
    # The repository declares aiohttp for deployment; local unit tests use a
    # fake transport and need only the import-time names.
    aiohttp_stub = types.ModuleType("aiohttp")
    aiohttp_stub.ClientSession = type("ClientSession", (), {})
    aiohttp_stub.ClientError = type("ClientError", (Exception,), {})
    aiohttp_stub.ClientTimeout = lambda **_: None
    aiohttp_stub.WSMsgType = types.SimpleNamespace(TEXT=1, CLOSED=2, ERROR=3)
    sys.modules["aiohttp"] = aiohttp_stub

from agent.providers import ProviderError, create_adapter, verify_webhook


def signed(body, *, timestamp=None, signatures=None):
    key = b"test signing key"
    secret = "whsec_" + base64.b64encode(key).decode()
    timestamp = str(int(time.time()) if timestamp is None else timestamp)
    signed_body = b"wh_test." + timestamp.encode() + b"." + body
    signature = base64.b64encode(hmac.new(key, signed_body, hashlib.sha256).digest()).decode()
    headers = {"Webhook-Id": "wh_test", "Webhook-Timestamp": timestamp,
               "Webhook-Signature": signatures or "v1,bad v1," + signature}
    return secret, headers


def test_standard_webhook_verifies_original_body_and_multiple_signatures():
    body = b'{"id":"evt_1","data":{"call_id":"call_1","sip_headers":[{"name":"From","value":"sip:1@example"}]}}'
    secret, headers = signed(body)
    assert verify_webhook(secret, body, headers)["data"]["sip_headers"][0]["value"] == "sip:1@example"
    for invalid_body, invalid_headers in [
        (body + b" ", headers),
        (body, {**headers, "Webhook-Timestamp": str(int(time.time()) - 301)}),
        (body, {**headers, "Webhook-Signature": "v1,bad"}),
    ]:
        with pytest.raises(ValueError):
            verify_webhook(secret, invalid_body, invalid_headers)


def test_standard_webhook_fixed_hmac_vector(monkeypatch):
    monkeypatch.setattr("agent.providers.webhook.time.time", lambda: 1700000000)
    # Independently generated with openssl dgst -sha256 -hmac 'test signing key'.
    secret = "whsec_" + base64.b64encode(b"test signing key").decode()
    headers = {"webhook-id": "wh_test", "webhook-timestamp": "1700000000",
               "webhook-signature": "v1,A6szrNy8oqe92iqt3gUDUL7d0WXEhcnNqaCva9Bo8tI="}
    assert verify_webhook(secret, b'{"id":"evt_1"}', headers) == {"id": "evt_1"}


def test_webhook_rejects_oversize_and_unauthenticated_json():
    body = b"not-json"
    secret, headers = signed(body)
    with pytest.raises(ValueError):
        verify_webhook(secret, body, headers)
    with pytest.raises(ValueError):
        verify_webhook(secret, b"x" * (256 * 1024 + 1), headers)


class FakeResponse:
    def __init__(self, status=200):
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False


class FakeSession:
    def __init__(self, status=200):
        self.status = status
        self.posts = []
        self.closed = False

    def post(self, url, **kwargs):
        self.posts.append((url, kwargs))
        return FakeResponse(self.status)

    async def close(self):
        self.closed = True


def test_openai_accept_payload_path_and_hangup():
    asyncio.run(_openai_accept_payload_path_and_hangup())


async def _openai_accept_payload_path_and_hangup():
    session = FakeSession()
    adapter = create_adapter({"provider": "openai", "api_key": "secret"})
    adapter._session = session
    await adapter.accept("call/1", {"model": "gpt-realtime-2", "voice": "marin", "prompt": "Be concise",
                                    "language": "it-IT"},
                         [{"wire_name": "nv_find_destinations_v1", "description": "Find",
                           "input_schema": {"type": "object"}}])
    url, options = session.posts[0]
    assert url.endswith("/calls/call%2F1/accept")
    assert options["json"]["audio"] == {"output": {"voice": "marin"}}
    assert options["json"]["instructions"] == "Be concise\n\nSpeak and respond using this language code or name: it-IT."
    assert options["json"]["tools"][0]["name"] == "nv_find_destinations_v1"
    await adapter.hangup("call/1")
    assert session.posts[1][0].endswith("/calls/call%2F1/hangup")
    await adapter.close()


def test_openai_http_error_does_not_expose_api_key():
    async def check():
        adapter = create_adapter({"provider": "openai", "api_key": "sensitive-value"})
        adapter._session = FakeSession(status=403)
        with pytest.raises(ProviderError) as error:
            await adapter.accept("call_1", {}, [])
        assert "sensitive-value" not in str(error.value)
        with pytest.raises(ValueError):
            await adapter.hangup("bad\ncall")
    asyncio.run(check())


def test_blank_profile_model_and_voice_use_provider_defaults():
    async def check():
        session = FakeSession()
        adapter = create_adapter({"provider": "openai", "api_key": "secret"})
        adapter._session = session
        await adapter.accept("call_1", {"model": "", "voice": " ", "language": "auto"}, [])
        assert session.posts[0][1]["json"]["model"] == "gpt-realtime"
        assert session.posts[0][1]["json"]["audio"]["output"]["voice"] == "alloy"
        assert session.posts[0][1]["json"]["instructions"] == ""
    asyncio.run(check())


def test_grok_session_ack_before_connect_returns(monkeypatch):
    asyncio.run(_grok_session_ack_before_connect_returns(monkeypatch))


async def _grok_session_ack_before_connect_returns(monkeypatch):
    adapter = create_adapter({"provider": "grok", "api_key": "secret"})
    sent = []
    async def connect(call_id):
        assert call_id == "call_1"
        adapter._ws = type("Socket", (), {"closed": False})()
    async def send(event):
        sent.append(event)
        await adapter._handle({"type": "session.updated"})
    monkeypatch.setattr(adapter, "_connect", connect)
    monkeypatch.setattr(adapter, "_send", send)
    await adapter.accept("call_1", {"voice": "eve", "prompt": "Hello"}, [])
    await adapter.connect("call_1")
    assert sent == [{"type": "session.update", "session": {"voice": "eve",
                    "instructions": "Hello", "turn_detection": {"type": "server_vad"}, "tools": []}}]


def test_parallel_tool_batch_waits_for_response_and_playback(monkeypatch):
    asyncio.run(_parallel_tool_batch_waits_for_response_and_playback(monkeypatch))


async def _parallel_tool_batch_waits_for_response_and_playback(monkeypatch):
    adapter = create_adapter({"provider": "grok", "api_key": "secret"})
    sent = []
    async def send(event):
        sent.append(event)
    monkeypatch.setattr(adapter, "_send", send)
    await adapter._handle({"type": "response.created", "response": {"id": "r1"}})
    await adapter._handle({"type": "output_audio_buffer.started"})
    for call in ("one", "two"):
        await adapter._handle({"type": "response.function_call_arguments.done", "call_id": call,
                               "name": "lookup", "arguments": json.dumps({"id": call}), "response_id": "r1"})
    continuations = [asyncio.create_task(adapter.respond(response_id="r1")) for _ in range(2)]
    await asyncio.sleep(0)
    await adapter.send_result("one", {"ok": True})
    await adapter._handle({"type": "response.done", "response": {"id": "r1", "status": "completed",
                            "output": [{"type": "function_call", "call_id": "one", "name": "lookup", "arguments": "{}"}]}})
    await asyncio.sleep(0)
    assert not any(event["type"] == "response.create" for event in sent)
    await adapter.send_result("two", {"ok": True})
    await asyncio.sleep(0)
    assert not any(event["type"] == "response.create" for event in sent)
    await adapter._handle({"type": "output_audio_buffer.stopped"})
    await asyncio.wait_for(asyncio.gather(*continuations), timeout=1)
    assert sum(event["type"] == "response.create" for event in sent) == 1
    queued = [adapter._queue.get_nowait() for _ in range(adapter._queue.qsize())]
    assert [event["invocation_id"] for event in queued if event["type"] == "tool_call"] == ["one", "two"]
