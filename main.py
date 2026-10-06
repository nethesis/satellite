import asyncio
import logging
import os
import signal
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")
import uvicorn
from api import app as api_app
from asterisk_bridge import AsteriskBridge
from mqtt_client import MQTTClient
from rtp_server import RTPServer


# Configure logging
log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger("main")

# For graceful shutdown
shutdown_event = asyncio.Event()

def signal_handler():
    logger.info("Shutdown signal received")
    shutdown_event.set()


async def supervise_transcription():
    delay = 1
    while not shutdown_event.is_set():
        try:
            await realtime_call_transcription()
            if shutdown_event.is_set():
                return
            logger.error("Transcription pipeline stopped")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("Transcription pipeline stopped: %s", type(exc).__name__)
        try:
            await asyncio.wait_for(shutdown_event.wait(), delay)
        except asyncio.TimeoutError:
            delay = min(delay * 2, 30)


async def realtime_call_transcription():
    # exit if deepgram api key is not set
    if not os.getenv("DEEPGRAM_API_KEY"):
        logger.error("DEEPGRAM_API_KEY is not set")
        return

    # Check for Google Cloud credentials
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials_path and not os.path.exists(credentials_path):
        logger.error(f"Google Cloud credentials file not found at {credentials_path}")
        return

    # Get configuration from environment variables
    asterisk_url = os.getenv("ASTERISK_URL", "http://localhost:8088")
    asterisk_app = os.getenv("ARI_APP", "asterisk_dialogflow")
    asterisk_username = os.getenv("ARI_USERNAME", "asterisk")
    asterisk_password = os.getenv("SATELLITE_ARI_PASSWORD", "asterisk")

    mqtt_url = os.getenv("MQTT_URL", "mqtt://localhost:1883")
    mqtt_topic_prefix = os.getenv("MQTT_TOPIC_PREFIX", "asterisk-dialogflow")
    mqtt_username = os.getenv("MQTT_USERNAME")
    mqtt_password = os.getenv("SATELLITE_MQTT_PASSWORD")

    # create an RTP server instance
    rtp_host = os.getenv("RTP_HOST", "0.0.0.0")
    rtp_port = int(os.getenv("RTP_PORT", "10000"))
    rtp_swap16 = os.getenv("RTP_SWAP16", "true").lower() == "true"
    rtp_header_size = int(os.getenv("RTP_HEADER_SIZE", "12"))
    rtp_server = RTPServer(host=rtp_host, port=rtp_port, swap16=rtp_swap16, rtp_header_size=rtp_header_size)
    await rtp_server.start()

    # Create instances
    mqtt_client = MQTTClient(
        url=mqtt_url,
        topic_prefix=mqtt_topic_prefix,
        username=mqtt_username,
        password=mqtt_password
    )
    asterisk_bridge = AsteriskBridge(
        url=asterisk_url,
        app=asterisk_app,
        username=asterisk_username,
        password=asterisk_password,
        mqtt_client=mqtt_client,
        rtp_server=rtp_server
    )

    async def handle_mqtt_control(topic, payload):
        """Handle transcription control commands from middleware."""
        if not isinstance(payload, dict):
            return

        action = payload.get("action")
        call_id = payload.get("linkedid") or payload.get("uniqueid")
        if not isinstance(action, str) or not isinstance(call_id, str) or not call_id:
            return

        if action == "start":
            await asterisk_bridge.start_transcription(call_id)
        elif action == "stop":
            await asterisk_bridge.stop_transcription(call_id)

    # Start services. A startup failure or cancellation still releases the RTP
    # socket and any partially opened MQTT/ARI connection.
    logger.info("Starting services...")
    try:
        mqtt_client.set_callback(handle_mqtt_control)
        await mqtt_client.connect()
        await mqtt_client.subscribe("transcription/control")
        await asterisk_bridge.connect()
        logger.info("All services started")
        await shutdown_event.wait()
    finally:
        logger.info("Shutting down transcription...")
        try:
            await asterisk_bridge.disconnect()
        finally:
            try:
                await mqtt_client.disconnect()
            finally:
                await rtp_server.stop()
        logger.info("Transcription shutdown complete")

async def main():
    # Keep the control plane alive independently of optional transcription.
    server = uvicorn.Server(uvicorn.Config(
        api_app, host=os.getenv("HTTP_HOST", "127.0.0.1").strip() or "127.0.0.1", port=int(os.getenv("HTTP_PORT", "8000")),
        log_level=os.getenv("LOG_LEVEL", "info").lower()))
    transcription = None
    enabled = os.getenv("SATELLITE_CALL_TRANSCRIPTION_ENABLED")
    if (enabled is None or enabled.lower() == "true") and os.getenv("DEEPGRAM_API_KEY"):
        transcription = asyncio.create_task(supervise_transcription())
    async def stop_transcription():
        shutdown_event.set()
        if transcription:
            try:
                await asyncio.wait_for(asyncio.shield(transcription), timeout=10)
            except asyncio.TimeoutError:
                transcription.cancel()
                await asyncio.gather(transcription, return_exceptions=True)

    # Uvicorn runs shutdown hooks before restoring and re-raising SIGTERM.
    api_app.add_event_handler("shutdown", stop_transcription)
    try:
        await server.serve()
    finally:
        await stop_transcription()

if __name__ == "__main__":
    asyncio.run(main())
