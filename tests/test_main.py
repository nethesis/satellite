"""M0 regression: preserve transcription startup beside the independent runner."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
async def test_transcription_pipeline_still_starts_and_stops(monkeypatch):
    import main
    monkeypatch.setenv('DEEPGRAM_API_KEY', 'synthetic-test')
    monkeypatch.delenv('GOOGLE_APPLICATION_CREDENTIALS', raising=False)
    finished = asyncio.Event()
    finished.set()
    monkeypatch.setattr(main, 'shutdown_event', finished)
    monkeypatch.setattr(asyncio.get_running_loop(), 'add_signal_handler', MagicMock())
    rtp = AsyncMock()
    mqtt = AsyncMock()
    mqtt.set_callback = MagicMock()
    ari = AsyncMock()
    monkeypatch.setattr(main, 'RTPServer', MagicMock(return_value=rtp))
    monkeypatch.setattr(main, 'MQTTClient', MagicMock(return_value=mqtt))
    monkeypatch.setattr(main, 'AsteriskBridge', MagicMock(return_value=ari))
    await main.realtime_call_transcription()
    rtp.start.assert_awaited_once()
    mqtt.connect.assert_awaited_once()
    mqtt.subscribe.assert_awaited_once_with('transcription/control')
    ari.connect.assert_awaited_once()
    assert callable(mqtt.set_callback.call_args.args[0])
    ari.disconnect.assert_awaited_once()
    rtp.stop.assert_awaited_once()
    mqtt.disconnect.assert_awaited_once()


def test_transcription_api_honors_private_bind(monkeypatch):
    import main
    monkeypatch.delenv('HTTP_HOST', raising=False)
    assert main.api_server_settings()['host'] == '127.0.0.1'
    monkeypatch.setenv('HTTP_HOST', '127.0.0.2')
    monkeypatch.setenv('HTTP_PORT', '8765')
    assert main.api_server_settings()['host'] == '127.0.0.2'
    assert main.api_server_settings()['port'] == 8765
