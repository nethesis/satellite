import asyncio
from unittest.mock import AsyncMock

import pytest

from asterisk_bridge import AsteriskBridge


@pytest.mark.asyncio
async def test_start_connector_ignores_concurrent_duplicate_start():
    bridge = AsteriskBridge(
        url="http://ari.local",
        app="satellite",
        username="user",
        password="pass",
        mqtt_client=AsyncMock(),
        rtp_server=AsyncMock(),
    )

    start_gate = asyncio.Event()
    connector = AsyncMock()
    async def start():
        await start_gate.wait()

    connector.start = AsyncMock(side_effect=start)

    bridge.channels["chan-1"] = {
        "connector_started": False,
        "connector_starting": False,
        "rtp_stream_in": object(),
        "rtp_stream_out": object(),
        "call_elapsed_at_start": 0,
        "connector": connector,
    }

    first = asyncio.create_task(bridge._start_connector("chan-1"))
    await asyncio.sleep(0)
    second = asyncio.create_task(bridge._start_connector("chan-1"))
    await asyncio.sleep(0)

    start_gate.set()
    await asyncio.gather(first, second)

    connector.start.assert_awaited_once()
    assert bridge.channels["chan-1"]["connector_started"] is True


@pytest.mark.asyncio
async def test_disconnect_cancels_reconnect_and_connector_start_then_closes_owned_resources():
    bridge = AsteriskBridge('http://ari.local', 'satellite', 'user', 'pass',
                            AsyncMock(), AsyncMock())
    bridge.session, bridge.ws = AsyncMock(), AsyncMock()
    starting = asyncio.Event()
    connector = AsyncMock()

    async def start():
        starting.set()
        await asyncio.Event().wait()

    connector.start.side_effect = start
    bridge.channels['transcription-channel'] = {
        'rtp_stream_in': object(), 'rtp_stream_out': object(),
        'call_elapsed_at_start': 0, 'connector': connector}
    bridge._event_task = asyncio.create_task(bridge._reconnect())
    bridge._schedule_connector('transcription-channel')
    await asyncio.wait_for(starting.wait(), 1)
    event_task = bridge._event_task
    await bridge.disconnect()
    assert event_task.done()
    assert not bridge._connector_tasks
    assert not bridge.channels
    connector.close.assert_awaited_once()
    bridge.ws.close.assert_awaited_once()
    bridge.session.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_cleanup_removes_only_transcription_media_and_is_idempotent():
    bridge = AsteriskBridge('http://ari.local', 'satellite', 'user', 'pass',
                            AsyncMock(), AsyncMock())
    bridge._ari_request = AsyncMock()
    bridge.channels['original-caller'] = {
        'snoop_channel_in': 'owned-snoop', 'external_media_channel_out': 'owned-media',
        'bridge_in': 'owned-bridge'}
    await asyncio.gather(bridge.close_channel('original-caller'),
                         bridge.close_channel('original-caller'))
    assert not bridge.channels
    deleted = {call.args[1] for call in bridge._ari_request.await_args_list}
    assert deleted == {'/channels/owned-snoop', '/channels/owned-media', '/bridges/owned-bridge'}
