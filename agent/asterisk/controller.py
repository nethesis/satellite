"""A small, separate ARI connection for the satellite-agent Stasis app.

The controller transports events and explicit channel operations. Call policy and
ownership decisions belong to :mod:`agent.runtime`.
"""

import asyncio
import json
import logging
from urllib.parse import quote

import aiohttp


LOG = logging.getLogger(__name__)


class AriController:
    def __init__(self, url, username, password, app="satellite-agent", on_event=None,
                 on_disconnect=None):
        self.url = url.rstrip("/")
        self.auth = aiohttp.BasicAuth(username, password)
        self.app = app
        self.on_event = on_event
        self.on_disconnect = on_disconnect
        self.connected = False
        self._session = None
        self._ws = None
        self._task = None
        self._stopping = False
        self._queue = asyncio.Queue(maxsize=256)
        self._worker = None
        self._generation = 0

    async def start(self):
        if self._task is not None:
            return
        self._stopping = False
        self._session = aiohttp.ClientSession(auth=self.auth)
        self._worker = asyncio.create_task(self._events(), name="agent-ari-events")
        self._task = asyncio.create_task(self._run(), name="agent-ari-connection")

    async def stop(self):
        self._stopping = True
        if self._ws is not None:
            await self._ws.close()
        for task in (self._task, self._worker):
            if task is not None:
                task.cancel()
        await asyncio.gather(*(task for task in (self._task, self._worker) if task),
                             return_exceptions=True)
        self._task = self._worker = self._ws = None
        self.connected = False
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def _run(self):
        delay = 1
        while not self._stopping:
            try:
                url = self.url.replace("http://", "ws://", 1).replace("https://", "wss://", 1)
                async with self._session.ws_connect(
                    f"{url}/ari/events", params={"app": self.app}, heartbeat=30
                ) as ws:
                    self._ws = ws
                    self.connected = True
                    self._generation += 1
                    delay = 1
                    # A reconnect can leave Stasis channels from the previous owner.
                    # The runtime reconciles only channels with Agent ownership vars.
                    if self.on_event:
                        await self.on_event({"type": "AgentAriConnected"})
                    async for message in ws:
                        if message.type == aiohttp.WSMsgType.TEXT:
                            try:
                                event = json.loads(message.data)
                                self._queue.put_nowait((self._generation, event))
                            except (ValueError, asyncio.QueueFull):
                                LOG.warning("Agent ARI event rejected or queue full")
                                await ws.close()
                                break
                        elif message.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                            break
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                LOG.warning("Agent ARI connection failed: %s", type(exc).__name__)
            finally:
                self._ws = None
                was_connected = self.connected
                self.connected = False
                self._generation += 1
                if was_connected and not self._stopping and self.on_disconnect:
                    await self.on_disconnect()
            if not self._stopping:
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30)

    async def _events(self):
        while True:
            generation, event = await self._queue.get()
            try:
                if generation == self._generation and self.on_event:
                    await self.on_event(event)
            except asyncio.CancelledError:
                raise
            except Exception:
                LOG.exception("Agent ARI event handler failed")
            finally:
                self._queue.task_done()

    async def request(self, method, path, *, params=None, json_body=None):
        if self._session is None or not self.connected:
            raise ConnectionError("Agent ARI unavailable")
        async with self._session.request(method, f"{self.url}/ari{path}",
                                         params=params, json=json_body,
                                         timeout=aiohttp.ClientTimeout(total=10)) as response:
            if response.status >= 400:
                raise RuntimeError(f"Agent ARI {method} {path}: HTTP {response.status}")
            if response.status == 204:
                return None
            return await response.json()

    async def get_variable(self, channel_id, name):
        try:
            result = await self.request("GET", f"/channels/{quote(channel_id, safe='')}/variable",
                                        params={"variable": name})
            return result.get("value") if isinstance(result, dict) else None
        except RuntimeError as exc:
            if "HTTP 404" in str(exc):
                return None
            raise

    async def set_variable(self, channel_id, name, value):
        await self.request("POST", f"/channels/{quote(channel_id, safe='')}/variable",
                           params={"variable": name, "value": str(value)})

    async def originate_local(self, session_id, channel_id, variables):
        return await self.request("POST", "/channels", params={
            "endpoint": f"Local/{session_id}@satellite-agent-provider/n",
            "app": self.app,
            "appArgs": f"provider,{session_id}",
            "channelId": channel_id,
            "timeout": 20,
        }, json_body={"variables": variables})

    async def answer(self, channel_id):
        await self.request("POST", f"/channels/{quote(channel_id, safe='')}/answer")

    async def create_bridge(self, bridge_id):
        return await self.request("POST", "/bridges", params={"type": "mixing", "bridgeId": bridge_id})

    async def add_to_bridge(self, bridge_id, channel_ids):
        await self.request("POST", f"/bridges/{quote(bridge_id, safe='')}/addChannel",
                           params={"channel": ",".join(channel_ids)})

    async def destroy_bridge(self, bridge_id):
        try:
            await self.request("DELETE", f"/bridges/{quote(bridge_id, safe='')}")
        except RuntimeError as exc:
            if "HTTP 404" not in str(exc):
                raise

    async def continue_channel(self, channel_id, target=None):
        params = (dict(target) if target else {})
        if "exten" in params:
            params["extension"] = params.pop("exten")
        await self.request("POST", f"/channels/{quote(channel_id, safe='')}/continue", params=params)

    async def hangup(self, channel_id):
        try:
            await self.request("DELETE", f"/channels/{quote(channel_id, safe='')}")
        except RuntimeError as exc:
            if "HTTP 404" not in str(exc):
                raise

    async def list_channels(self):
        return await self.request("GET", "/channels")

    async def list_bridges(self):
        return await self.request("GET", "/bridges")
