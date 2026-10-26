"""Satellite application: voice agents and optional transcription on one event loop."""
import asyncio
from contextlib import nullcontext
import ipaddress
import logging
import os
import signal

from dotenv import load_dotenv
import uvicorn

from voice_agent.runner import app_from_env, settings


def api_server_settings():
    host = os.getenv("HTTP_HOST", "127.0.0.1")
    if not ipaddress.ip_address(host).is_loopback:
        raise ValueError("loopback_bind_required")
    return {"host": host, "port": int(os.getenv("HTTP_PORT", "8000")),
            "log_level": os.getenv("LOG_LEVEL", "info").lower()}


async def run(shutdown_event=None):
    shutdown_event = shutdown_event or asyncio.Event()
    host, port = settings()
    servers = [uvicorn.Server(uvicorn.Config(
        app_from_env(), host=host, port=port, access_log=False,
        log_level=os.getenv("LOG_LEVEL", "info").lower(), timeout_graceful_shutdown=10))]
    transcription_task = None
    if any(os.getenv(name, "").lower() == "true" for name in (
            "SATELLITE_CALL_TRANSCRIPTION_ENABLED", "SATELLITE_VOICEMAIL_TRANSCRIPTION_ENABLED")):
        from api import app
        from transcription import run as transcribe
        api_settings = api_server_settings()
        if (api_settings["host"], api_settings["port"]) == (host, port):
            raise ValueError("http_listener_collision")
        if os.getenv("ARI_APP") == "voice-agent":
            raise ValueError("transcription_ari_app_must_differ_from_voice_agents")
        servers.append(uvicorn.Server(uvicorn.Config(
            app, **api_settings, timeout_graceful_shutdown=10)))
        transcription_task = asyncio.create_task(transcribe(shutdown_event), name="transcription")

    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, shutdown_event.set)
    # Only this application owns signals; neither HTTP server replaces its handlers.
    for server in servers:
        server.capture_signals = nullcontext
    tasks = [asyncio.create_task(server.serve()) for server in servers]
    stopping = asyncio.create_task(shutdown_event.wait())
    try:
        watched = [stopping, *tasks]
        if transcription_task:
            watched.append(transcription_task)
        done, _ = await asyncio.wait(watched, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    finally:
        shutdown_event.set()
        for server in servers:
            server.should_exit = True
        stopping.cancel()
        await asyncio.gather(*tasks, stopping,
                             *([transcription_task] if transcription_task else []),
                             return_exceptions=True)
        for signum in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(signum)


def main():
    load_dotenv(dotenv_path=".env")
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(),
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    asyncio.run(run())


if __name__ == "__main__":
    main()
