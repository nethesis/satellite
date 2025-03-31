import asyncio
import logging
import os
import signal
from dotenv import load_dotenv
from asterisk_bridge import AsteriskBridge
from mqtt_client import MQTTClient
from rtp_server import RTPServer


# Load environment variables
load_dotenv(dotenv_path=".env")

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger("main")

# For graceful shutdown
shutdown_event = asyncio.Event()

def signal_handler():
    logger.info("Shutdown signal received")
    shutdown_event.set()

async def main():
    # Set up signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    # Check for Google Cloud credentials
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials_path and not os.path.exists(credentials_path):
        logger.error(f"Google Cloud credentials file not found at {credentials_path}")
        return

    # Get configuration from environment variables
    asterisk_url = os.getenv("ASTERISK_URL", "http://localhost:8088")
    asterisk_app = os.getenv("ARI_APP", "asterisk_dialogflow")
    asterisk_username = os.getenv("ARI_USERNAME", "asterisk")
    asterisk_password = os.getenv("ARI_PASSWORD", "asterisk")

    mqtt_url = os.getenv("MQTT_URL", "mqtt://localhost:1883")
    mqtt_topic_prefix = os.getenv("MQTT_TOPIC_PREFIX", "asterisk-dialogflow")

    # create an RTP server instance
    rtp_host = os.getenv("RTP_HOST", "0.0.0.0")
    rtp_port = int(os.getenv("RTP_PORT", "10000"))
    rtp_swap16 = os.getenv("RTP_SWAP16", "true").lower() == "true"
    rtp_header_size = int(os.getenv("RTP_HEADER_SIZE", "12"))
    rtp_server = RTPServer(host=rtp_host, port=rtp_port, swap16=rtp_swap16, rtp_header_size=rtp_header_size)
    await rtp_server.start()

    # Create instances
    mqtt_client = MQTTClient(url=mqtt_url, topic_prefix=mqtt_topic_prefix)
    asterisk_bridge = AsteriskBridge(
        url=asterisk_url,
        app=asterisk_app,
        username=asterisk_username,
        password=asterisk_password,
        mqtt_client=mqtt_client,
        rtp_server=rtp_server
    )

    # Start services
    logger.info("Starting services...")
    await mqtt_client.connect()
    await asterisk_bridge.connect()
    logger.info("All services started")

    # Wait for shutdown signal
    await shutdown_event.wait()

    # Graceful shutdown
    logger.info("Shutting down...")
    # End all active Stasis sessions before disconnecting
    await asterisk_bridge.disconnect()
    await rtp_server.stop()
    await mqtt_client.disconnect()
    logger.info("Shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())

