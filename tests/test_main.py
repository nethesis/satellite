"""One application, private listeners, optional transcription and graceful signals."""
import asyncio
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

import main
import transcription


@pytest.fixture
def resources(monkeypatch):
    monkeypatch.setenv('DEEPGRAM_API_KEY', 'synthetic-test')
    monkeypatch.setenv('ARI_APP', 'satellite')
    monkeypatch.delenv('GOOGLE_APPLICATION_CREDENTIALS', raising=False)
    rtp, mqtt, ari = AsyncMock(), AsyncMock(), AsyncMock()
    mqtt.set_callback = MagicMock()
    monkeypatch.setattr(transcription, 'RTPServer', MagicMock(return_value=rtp))
    monkeypatch.setattr(transcription, 'MQTTClient', MagicMock(return_value=mqtt))
    monkeypatch.setattr(transcription, 'AsteriskBridge', MagicMock(return_value=ari))
    return rtp, mqtt, ari


@pytest.mark.asyncio
async def test_transcription_pipeline_starts_and_cleans_up(resources):
    stop = asyncio.Event()
    rtp, mqtt, ari = resources
    ari.connect.side_effect = stop.set
    await transcription.run(stop)
    rtp.start.assert_awaited_once()
    mqtt.connect.assert_awaited_once()
    mqtt.subscribe.assert_awaited_once_with('transcription/control')
    ari.connect.assert_awaited_once()
    control = mqtt.set_callback.call_args.args[0]
    await control('ignored', {'action': 'start', 'linkedid': 'call-id'})
    await control('ignored', {'action': 'stop', 'uniqueid': 'call-id'})
    ari.start_transcription.assert_awaited_once_with('call-id')
    ari.stop_transcription.assert_awaited_once_with('call-id')
    ari.disconnect.assert_awaited_once()
    rtp.stop.assert_awaited_once()
    mqtt.disconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_stop_interrupts_mqtt_startup_and_cleans_partial_resources(resources):
    stop, connecting = asyncio.Event(), asyncio.Event()
    rtp, mqtt, ari = resources

    async def unavailable():
        connecting.set()
        await asyncio.Event().wait()

    mqtt.connect.side_effect = unavailable
    task = asyncio.create_task(transcription.run(stop))
    await asyncio.wait_for(connecting.wait(), 1)
    stop.set()
    await asyncio.wait_for(task, 1)
    ari.connect.assert_not_awaited()
    ari.disconnect.assert_awaited_once()
    mqtt.disconnect.assert_awaited_once()
    rtp.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_transcription_retries_initial_ari_failure(resources, monkeypatch):
    stop = asyncio.Event()
    rtp, mqtt, ari = resources
    attempts = 0

    async def connect():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError('synthetic unavailable ARI')
        stop.set()

    ari.connect.side_effect = connect
    monkeypatch.setattr(transcription.asyncio, 'sleep', AsyncMock())
    await transcription.run(stop)
    assert attempts == 2
    assert ari.disconnect.await_count == mqtt.disconnect.await_count == rtp.stop.await_count == 2


def test_transcription_api_bind_is_private(monkeypatch):
    monkeypatch.delenv('HTTP_HOST', raising=False)
    assert main.api_server_settings()['host'] == '127.0.0.1'
    monkeypatch.setenv('HTTP_HOST', '127.0.0.2')
    monkeypatch.setenv('HTTP_PORT', '8765')
    assert main.api_server_settings()['port'] == 8765
    monkeypatch.setenv('HTTP_HOST', '0.0.0.0')
    with pytest.raises(ValueError, match='loopback_bind_required'):
        main.api_server_settings()


def free_port():
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        return listener.getsockname()[1]


@pytest.mark.parametrize('call,voicemail', [('False', 'False'), ('True', 'False'),
                                          ('False', 'True'), ('True', 'True')])
@pytest.mark.parametrize('stop_signal', [signal.SIGTERM, signal.SIGINT])
def test_single_process_endpoints_and_shutdown(tmp_path, call, voicemail, stop_signal):
    agent_port, api_port = free_port(), free_port()
    while api_port == agent_port:
        api_port = free_port()
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('PGVECTOR_', 'VOICE_AGENT_', 'SATELLITE_'))
           and key not in ('GOOGLE_APPLICATION_CREDENTIALS', 'OPENAI_API_KEY')}
    env.update(VOICE_AGENT_HTTP_PORT=str(agent_port), VOICE_AGENT_HTTP_HOST='127.0.0.1',
               VOICE_AGENT_ENABLED='false', VOICE_AGENT_RUNTIME_TOKEN='synthetic-token',
               HTTP_HOST='127.0.0.1', HTTP_PORT=str(api_port), API_TOKEN='synthetic-api',
               SATELLITE_CALL_TRANSCRIPTION_ENABLED=call,
               SATELLITE_VOICEMAIL_TRANSCRIPTION_ENABLED=voicemail,
               DEEPGRAM_API_KEY='synthetic-deepgram', MQTT_URL='mqtt://127.0.0.1:1',
               RTP_HOST='127.0.0.1', RTP_PORT='0', LOG_LEVEL='WARNING')
    enabled = 'True' in (call, voicemail)
    process = subprocess.Popen([sys.executable, str(Path(main.__file__))], cwd=tmp_path,
                               env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        with httpx.Client(timeout=0.3, trust_env=False) as client:
            agent_url = f'http://127.0.0.1:{agent_port}'
            api_url = f'http://127.0.0.1:{api_port}'
            deadline = time.monotonic() + 15
            while True:
                assert process.poll() is None, process.communicate()
                try:
                    if client.get(agent_url + '/internal/ready').status_code == 401:
                        if not enabled or client.get(api_url + '/metrics').status_code == 401:
                            break
                except httpx.TransportError:
                    pass
                assert time.monotonic() < deadline, 'HTTP startup timed out'
                time.sleep(0.05)
            assert client.get(agent_url + '/internal/ready', headers={
                'Authorization': 'Bearer synthetic-token'}).status_code == 200
            assert client.get(agent_url + '/api/get_transcription').status_code == 404
            assert client.post(agent_url + '/satellite-webhooks/unexpected').status_code == 404
            if enabled:
                assert client.get(api_url + '/metrics', headers={'X-API-Token': 'synthetic-api'}).status_code == 200
                assert client.get(api_url + '/internal/ready').status_code == 404
                assert client.post(api_url + '/satellite-webhooks/openai').status_code == 404
            else:
                with pytest.raises(httpx.TransportError):
                    client.get(api_url + '/metrics')
            assert not Path(f'/proc/{process.pid}/task/{process.pid}/children').read_text().strip()
        process.send_signal(stop_signal)
        _, errors = process.communicate(timeout=10)
        assert process.returncode == 0, errors.decode()
        assert b'Unclosed client session' not in errors
        assert b'Task was destroyed' not in errors
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()


@pytest.mark.asyncio
async def test_transcription_recovery_preserves_running_agent_calls(resources, monkeypatch):
    from types import SimpleNamespace
    from voice_agent.runner import create_app

    rtp, mqtt, ari = resources
    runtime = SimpleNamespace(start=AsyncMock(), stop=AsyncMock(),
                              controller=SimpleNamespace(connected=True),
                              monitoring=SimpleNamespace(degraded=False), calls={'active-call': object()})
    active_call = runtime.calls['active-call']
    monkeypatch.setattr(main, 'app_from_env', lambda: create_app('synthetic-token', runtime))
    agent_port, api_port = free_port(), free_port()
    monkeypatch.setenv('VOICE_AGENT_HTTP_PORT', str(agent_port))
    monkeypatch.setenv('VOICE_AGENT_HTTP_HOST', '127.0.0.1')
    monkeypatch.setenv('HTTP_PORT', str(api_port))
    monkeypatch.setenv('HTTP_HOST', '127.0.0.1')
    monkeypatch.setenv('SATELLITE_CALL_TRANSCRIPTION_ENABLED', 'True')
    monkeypatch.setenv('SATELLITE_VOICEMAIL_TRANSCRIPTION_ENABLED', 'False')
    monkeypatch.setattr(asyncio.get_running_loop(), 'add_signal_handler', MagicMock())
    monkeypatch.setattr(asyncio.get_running_loop(), 'remove_signal_handler', MagicMock())
    recovered = asyncio.Event()
    attempts = 0

    async def connect():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError('synthetic transcription ARI outage')
        recovered.set()

    ari.connect.side_effect = connect
    stop = asyncio.Event()
    application = asyncio.create_task(main.run(stop))
    try:
        async with httpx.AsyncClient(timeout=0.5, trust_env=False) as client:
            async with asyncio.timeout(3):
                while True:
                    try:
                        response = await client.get(f'http://127.0.0.1:{agent_port}/internal/ready',
                            headers={'Authorization': 'Bearer synthetic-token'})
                        if response.status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    await asyncio.sleep(0.01)
            assert response.json()['voice_calls_enabled'] is True
            await asyncio.wait_for(recovered.wait(), 7)
            assert runtime.calls['active-call'] is active_call
            runtime.start.assert_awaited_once()
            runtime.stop.assert_not_awaited()
            assert (await client.get(f'http://127.0.0.1:{agent_port}/internal/ready', headers={
                'Authorization': 'Bearer synthetic-token'})).json()['voice_calls_enabled'] is True
    finally:
        stop.set()
        await asyncio.wait_for(application, 3)
    runtime.stop.assert_awaited_once()
    assert rtp.stop.await_count == mqtt.disconnect.await_count == ari.disconnect.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize('conflict', ['http', 'ari'])
async def test_listener_and_ari_ownership_collisions_fail_before_start(monkeypatch, conflict):
    monkeypatch.setenv('VOICE_AGENT_ENABLED', 'false')
    monkeypatch.setenv('VOICE_AGENT_RUNTIME_TOKEN', 'synthetic-token')
    monkeypatch.setenv('SATELLITE_CALL_TRANSCRIPTION_ENABLED', 'True')
    monkeypatch.setenv('HTTP_HOST', '127.0.0.1')
    monkeypatch.setenv('VOICE_AGENT_HTTP_HOST', '127.0.0.1')
    monkeypatch.setenv('VOICE_AGENT_HTTP_PORT', '8701')
    monkeypatch.setenv('HTTP_PORT', '8701' if conflict == 'http' else '8702')
    monkeypatch.setenv('ARI_APP', 'voice-agent' if conflict == 'ari' else 'satellite')
    with pytest.raises(ValueError, match='http_listener_collision' if conflict == 'http'
                       else 'transcription_ari_app_must_differ'):
        await main.run()
