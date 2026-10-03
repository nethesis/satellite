"""OpenAI Realtime and Grok SIP sideband transports.

Only provider protocol details live here; call admission and tool policy belong
to the Agent runtime. See the providers' SIP and speech-to-speech documentation.
"""

import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from urllib.parse import quote, urlencode

import aiohttp


class ProviderError(RuntimeError):
    """Safe provider failure suitable for a bounded readiness/error event."""


def _call_id(value: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 256 or any(ord(c) < 33 for c in value):
        raise ValueError("invalid provider call ID")
    return value


def _profile_value(profile: Mapping, key: str, default):
    value = profile.get(key)
    return value if value is not None and (not isinstance(value, str) or value.strip()) else default


def _instructions(profile: Mapping) -> str:
    prompt = _profile_value(profile, "prompt", "")
    language = _profile_value(profile, "language", "")
    if language and language.lower() not in ("auto", "automatic"):
        return f"{prompt}\n\nSpeak and respond using this language code or name: {language}.".strip()
    return prompt


def _tools(tools) -> list[dict]:
    result = []
    for tool in tools or []:
        if isinstance(tool, dict):
            if tool.get("type") == "function":
                # The runtime validates arguments itself. Provider schemas only
                # advertise the tool; xAI does not document a strict flag.
                result.append({key: tool[key] for key in ("type", "name", "description", "parameters")
                               if key in tool})
                continue
            result.append({"type": "function", "name": tool["wire_name"],
                           "description": tool.get("description", ""),
                           "parameters": tool["input_schema"]})
        else:
            result.append({"type": "function", "name": tool.wire_name,
                           "description": tool.description,
                           "parameters": tool.input_schema})
    return result


class _RealtimeAdapter:
    http_base: str
    ws_base: str
    provider: str

    def __init__(self, binding: Mapping, *, session: aiohttp.ClientSession | None = None):
        self._api_key = binding.get("api_key")
        if not isinstance(self._api_key, str) or not self._api_key:
            raise ValueError("missing provider API key")
        self._session = session
        self._owns_session = session is None
        self._ws = None
        self._reader_task = None
        self._queue: asyncio.Queue[dict | None] = asyncio.Queue(maxsize=256)
        self._condition = asyncio.Condition()
        self._pending: set[str] = set()
        self._seen_calls: set[str] = set()
        self._response_done = True
        self._audio_playing = False
        self._response_id = "response-0"
        self._response_generation = 0
        self._done_responses: set[str] = set()
        self._continuations: set[str] = set()
        self._closed = False
        self._respond_lock = asyncio.Lock()
        self._call = None
        self._profile = None
        self._tools_config = []
        self._ready = asyncio.Event()
        self._session_configured = False

    async def _client(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15))
        return self._session

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    async def _post(self, url: str, body: dict | None = None) -> None:
        try:
            client = await self._client()
            async with client.post(url, headers=self._headers, json=body, allow_redirects=False) as response:
                if not 200 <= response.status < 300:
                    raise ProviderError(f"{self.provider} call control failed ({response.status})")
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            raise ProviderError(f"{self.provider} call control unavailable") from exc

    async def _connect(self, call_id: str) -> None:
        try:
            client = await self._client()
            self._ws = await client.ws_connect(
                self.ws_base + "?" + urlencode({"call_id": call_id}),
                headers=self._headers, heartbeat=20,
            )
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            raise ProviderError(f"{self.provider} sideband unavailable") from exc
        self._reader_task = asyncio.create_task(self._read_events())

    async def _send(self, event: dict) -> None:
        if self._ws is None or self._ws.closed:
            raise ProviderError(f"{self.provider} sideband closed")
        await self._ws.send_json(event)

    async def send_result(self, invocation_id: str, result) -> None:
        if not isinstance(invocation_id, str) or not invocation_id:
            raise ValueError("invalid invocation ID")
        output = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        await self._send({"type": "conversation.item.create", "item": {
            "type": "function_call_output", "call_id": invocation_id, "output": output,
        }})
        async with self._condition:
            self._pending.discard(invocation_id)
            self._condition.notify_all()

    async def respond(self, instructions: str | None = None, *,
                      response_id: str | None = None) -> None:
        # Claim a particular completed response before waiting. Parallel tool
        # results cannot issue another continuation after response.created has
        # advanced the session to the next generation.
        token = response_id if isinstance(response_id, str) and response_id else "greeting"
        async with self._condition:
            if token in self._continuations:
                return
            if len(self._continuations) >= 4096:
                raise ProviderError(f"{self.provider} response capacity exceeded")
            self._continuations.add(token)
        async with self._respond_lock:
            async with self._condition:
                await self._condition.wait_for(lambda: self._closed or (
                    not self._pending and not self._audio_playing and
                    (self._response_done if token == "greeting" else token in self._done_responses)))
                if self._closed:
                    raise ProviderError(f"{self.provider} sideband closed")
            event = {"type": "response.create"}
            if instructions is not None:
                event["response"] = {"instructions": instructions}
            await self._send(event)

    async def events(self) -> AsyncIterator[dict]:
        while True:
            event = await self._queue.get()
            if event is None:
                return
            yield event

    async def _emit(self, event: dict) -> None:
        # A stalled consumer must not force unbounded memory growth.
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            raise ProviderError(f"{self.provider} event queue full")

    async def _read_events(self) -> None:
        try:
            async for message in self._ws:
                if message.type != aiohttp.WSMsgType.TEXT:
                    if message.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        break
                    continue
                try:
                    raw = json.loads(message.data)
                except json.JSONDecodeError:
                    raise ProviderError(f"{self.provider} invalid event")
                if not isinstance(raw, dict):
                    continue
                await self._handle(raw)
        except Exception:
            try:
                await self._emit({"type": "error", "code": "provider_sideband_error"})
            except ProviderError:
                pass
        finally:
            self._ready.set()
            async with self._condition:
                self._closed = True
                self._response_done = True
                self._audio_playing = False
                self._pending.clear()
                self._condition.notify_all()
            for final_event in ({"type": "closed"}, None):
                if self._queue.full():
                    self._queue.get_nowait()
                self._queue.put_nowait(final_event)

    async def _handle(self, raw: dict) -> None:
        kind = raw.get("type")
        if kind == "session.updated":
            self._session_configured = True
            self._ready.set()
        if kind == "error":
            await self._emit({"type": "error", "code": "provider_event_error"})
        if kind == "response.created":
            async with self._condition:
                self._response_generation += 1
                response = raw.get("response") or {}
                self._response_id = response.get("id") or f"response-{self._response_generation}"
                self._response_done = False
        if kind == "output_audio_buffer.started":
            async with self._condition:
                self._audio_playing = True
            await self._emit({"type": "audio_started"})
        if kind in ("output_audio_buffer.stopped", "output_audio_buffer.cleared"):
            async with self._condition:
                self._audio_playing = False
                self._condition.notify_all()
            await self._emit({"type": "audio_stopped"})
        if kind == "response.function_call_arguments.done":
            await self._tool_call(raw, raw.get("response_id"))
        if kind == "response.done":
            response = raw.get("response") or {}
            response_id = response.get("id") or self._response_id
            for item in response.get("output") or []:
                if isinstance(item, dict) and item.get("type") == "function_call":
                    await self._tool_call(item, response_id)
            async with self._condition:
                if len(self._done_responses) >= 4096:
                    raise ProviderError(f"{self.provider} response capacity exceeded")
                self._done_responses.add(response_id)
                if response_id == self._response_id:
                    self._response_done = True
                self._condition.notify_all()
            await self._emit({"type": "response_done", "response_id": response_id,
                              "status": response.get("status")})

    async def _tool_call(self, item: dict, response_id: str | None) -> None:
        invocation_id = item.get("call_id")
        name = item.get("name")
        if not isinstance(invocation_id, str) or not invocation_id or not isinstance(name, str):
            return
        try:
            arguments = json.loads(item.get("arguments") or "{}")
        except (TypeError, ValueError):
            arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        response_id = response_id or self._response_id
        async with self._condition:
            if invocation_id in self._seen_calls:
                return
            if len(self._seen_calls) >= 4096:
                raise ProviderError(f"{self.provider} invocation capacity exceeded")
            self._seen_calls.add(invocation_id)
            self._pending.add(invocation_id)
            self._response_done = False
        await self._emit({"type": "tool_call", "invocation_id": invocation_id,
                          "name": name, "arguments": arguments, "response_id": response_id})

    async def close(self) -> None:
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
            self._reader_task = None
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None

    async def hangup(self, provider_call_id: str) -> None:
        call_id = _call_id(provider_call_id)
        await self._post(self.http_base + "/calls/" + quote(call_id, safe="") + "/hangup")


class OpenAIAdapter(_RealtimeAdapter):
    provider = "openai"
    http_base = "https://api.openai.com/v1/realtime"
    ws_base = "wss://api.openai.com/v1/realtime"

    async def accept(self, provider_call_id: str, profile: Mapping, tools) -> None:
        call_id = _call_id(provider_call_id)
        self._call, self._profile, self._tools_config = call_id, profile, _tools(tools)
        body = {"type": "realtime", "model": _profile_value(profile, "model", "gpt-realtime"),
                "instructions": _instructions(profile),
                "audio": {"output": {"voice": _profile_value(profile, "voice", "alloy")}},
                "tools": self._tools_config}
        await self._post(self.http_base + "/calls/" + quote(call_id, safe="") + "/accept", body)

    async def connect(self, provider_call_id: str) -> None:
        call_id = _call_id(provider_call_id)
        if call_id != self._call:
            raise ValueError("provider call mismatch")
        await self._connect(call_id)


class GrokAdapter(_RealtimeAdapter):
    provider = "grok"
    http_base = "https://api.x.ai/v1/realtime"
    ws_base = "wss://api.x.ai/v1/realtime"

    async def accept(self, provider_call_id: str, profile: Mapping, tools) -> None:
        self._call, self._profile, self._tools_config = _call_id(provider_call_id), profile, _tools(tools)

    async def connect(self, provider_call_id: str) -> None:
        call_id = _call_id(provider_call_id)
        if call_id != self._call:
            raise ValueError("provider call mismatch")
        await self._connect(call_id)
        await self._send({"type": "session.update", "session": {
            "voice": _profile_value(self._profile, "voice", "eve"),
            "instructions": _instructions(self._profile),
            "turn_detection": {"type": "server_vad"},
            "tools": self._tools_config,
        }})
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=10)
        except asyncio.TimeoutError as exc:
            raise ProviderError("grok session setup timed out") from exc
        if not self._session_configured or self._ws.closed:
            raise ProviderError("grok session setup failed")
