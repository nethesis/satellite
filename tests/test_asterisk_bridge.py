import asyncio
from unittest.mock import AsyncMock

import pytest

from asterisk_bridge import AsteriskBridge


def make_bridge():
    return AsteriskBridge(
        url="http://ari.local",
        app="satellite",
        username="user",
        password="pass",
        mqtt_client=AsyncMock(),
        rtp_server=AsyncMock(),
    )


@pytest.mark.asyncio
async def test_start_connector_ignores_concurrent_duplicate_start():
    bridge = make_bridge()

    start_gate = asyncio.Event()
    connector = AsyncMock()
    connector.start = AsyncMock(side_effect=lambda: start_gate.wait())

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
async def test_successful_blind_transfer_closes_tracked_channel():
    bridge = make_bridge()
    bridge.channels["chan-1"] = {}
    bridge.close_channel = AsyncMock()

    await bridge._handle_ari_event(
        {
            "type": "BridgeBlindTransfer",
            "result": "Success",
            "channel": {"id": "chan-1"},
        }
    )

    bridge.close_channel.assert_awaited_once_with("chan-1")


@pytest.mark.asyncio
async def test_failed_blind_transfer_keeps_tracked_channel_open():
    bridge = make_bridge()
    bridge.channels["chan-1"] = {}
    bridge.close_channel = AsyncMock()

    await bridge._handle_ari_event(
        {
            "type": "BridgeBlindTransfer",
            "result": "Failure",
            "channel": {"id": "chan-1"},
        }
    )

    bridge.close_channel.assert_not_awaited()


@pytest.mark.asyncio
async def test_attended_transfer_start_closes_tracked_channel():
    bridge = make_bridge()
    bridge.channels["chan-1"] = {}
    bridge.close_channel = AsyncMock()

    await bridge._handle_ari_event(
        {
            "type": "ChannelTransfer",
            "refer_to": {
                "requested_destination": {
                    "destination": "1001",
                    "protocol_id": "replaces-call-id",
                },
                "destination_channel": {"id": "transfer-target"},
            },
            "referred_by": {
                "source_channel": {"id": "chan-1"},
            },
        }
    )

    bridge.close_channel.assert_awaited_once_with("chan-1")


@pytest.mark.asyncio
async def test_blind_channel_transfer_start_does_not_close_before_bridge_transfer_event():
    bridge = make_bridge()
    bridge.channels["chan-1"] = {}
    bridge.close_channel = AsyncMock()

    await bridge._handle_ari_event(
        {
            "type": "ChannelTransfer",
            "refer_to": {
                "requested_destination": {
                    "destination": "1001",
                },
            },
            "referred_by": {
                "source_channel": {"id": "chan-1"},
            },
        }
    )

    bridge.close_channel.assert_not_awaited()


@pytest.mark.asyncio
async def test_successful_attended_transfer_completion_closes_tracked_channel_fallback():
    bridge = make_bridge()
    bridge.channels["chan-1"] = {}
    bridge.close_channel = AsyncMock()

    await bridge._handle_ari_event(
        {
            "type": "BridgeAttendedTransfer",
            "result": "Success",
            "transferer_first_leg": {"id": "transferer"},
            "transferee": {"id": "chan-1"},
        }
    )

    bridge.close_channel.assert_awaited_once_with("chan-1")
