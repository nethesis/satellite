import asyncio
import logging
import aiohttp
import json
import os
from deepgram_connector import DeepgramConnector

logger = logging.getLogger("asterisk_bridge")

class AsteriskBridge:
    """
    Manages the interface between Asterisk PBX and speech recognition services.

    This class connects to the Asterisk REST Interface (ARI) using WebSockets
    to monitor and control call channels. For each channel entering the Stasis
    application, it:

    1. Creates a snoop channel to capture audio
    2. Sets up an external media endpoint for RTP streaming
    3. Creates a bridge between the snoop channel and external media endpoint
    4. Establishes a connection to Deepgram for speech-to-text transcription
    5. Routes transcriptions to MQTT for external processing

    The class handles various Asterisk events including channel start/end,
    hangups, and bridge events, ensuring proper resource management throughout
    the call lifecycle.
    """

    def __init__(self, url, app, username, password, mqtt_client, rtp_server):
        self.url = url
        self.app = app
        self.auth = aiohttp.BasicAuth(username, password)
        self.mqtt_client = mqtt_client
        self.rtp_server = rtp_server
        self.channels = {}
        self.ws = None
        self.session = None
        self.is_shutting_down = False
        self.max_reconnect_delay = 30  # Maximum seconds between reconnection attempts

    async def connect(self):
        """Connect to Asterisk ARI and setup WebSocket for events"""
        logger.debug(f"Connect to Asterisk ARI at {self.url}")
        self.is_shutting_down = False
        self.session = aiohttp.ClientSession(auth=self.auth)

        # Connect to ARI WebSocket
        await self._connect_websocket()
        logger.info(f"Connected to Asterisk ARI at {self.url}")

    async def _connect_websocket(self):
        """Connect to Asterisk ARI WebSocket"""
        ws_url = f"{self.url.replace('http', 'ws')}/ari/events?app={self.app}&api_key={self.auth.login}:{self.auth.password}"
        self.ws = await self.session.ws_connect(ws_url)
        # Start event loop
        asyncio.create_task(self._process_ari_events())

    async def disconnect(self):
        """Disconnect from Asterisk ARI"""
        logger.debug(f"Disconnect from Asterisk ARI")
        self.is_shutting_down = True
        for channel_id in list(self.channels.keys()):
            await self.close_channel(channel_id)
        if self.ws:
            await self.ws.close()
        if self.session:
            await self.session.close()

    async def _process_ari_events(self):
        """Process events from Asterisk ARI WebSocket"""
        try:
            async for msg in self.ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    event = json.loads(msg.data)
                    await self._handle_ari_event(event)
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    logger.warning("WebSocket connection closed or error occurred")
                    break
        except Exception as e:
            logger.error(f"Error in WebSocket event loop: {e}")
        finally:
            # If we're not in a clean shutdown, attempt to reconnect
            if not self.is_shutting_down:
                logger.info("Unexpected WebSocket disconnection, will attempt to reconnect")
                await self._reconnect()

    async def _reconnect(self):
        """Attempt to reconnect to Asterisk with exponential backoff"""
        delay = 1  # Start with 1 second delay
        while not self.is_shutting_down:
            try:
                logger.info(f"Attempting to reconnect in {delay} seconds...")
                await asyncio.sleep(delay)
                await self._connect_websocket()
                logger.info("Successfully reconnected to Asterisk ARI")
                return
            except Exception as e:
                logger.error(f"Failed to reconnect: {e}")
                # Exponential backoff with maximum delay
                delay = min(delay * 2, self.max_reconnect_delay)

    async def _handle_ari_event(self, event):
        """Handle events from Asterisk ARI"""
        #logger.debug(f"ENTER: _handle_ari_event(type={event.get('type', 'unknown')})")
        event_type = event.get('type')
        if event_type == 'StasisStart':
            await self._handle_stasis_start(event)
        elif event_type == 'channelHangup':
            await self._handle_channel_hangup(event)
        elif event_type == 'StasisEnd':
            await self._handle_stasis_end(event)
        elif event_type == 'ChannelLeftBridge':
            await self._handle_channel_left_bridge(event)

    async def _handle_stasis_start(self, event):
        """
        Launched when a Stasis start event is received
        - create the bridge to the rtp server
        - create the deepgram connector
        - start the deepgram connector
        - return control to dialplan
        """
        channel = event['channel']
        channel_id = channel['id']
        #logger.debug(f"ENTER: _handle_stasis_start(channel_id={channel_id})")
        #logger.debug(f"Channel data: {channel}")

        if not channel_id.startswith("snoop-") and not channel_id.startswith("external-media-"):
            # Normal channel entered Stasis
            self.channels[channel_id] = {}
            self.channels[channel_id]['language'] = channel.get('language', 'en')

            # Create a snoop channel for it
            snoop_data = await self._ari_request(
                'POST',
                f"/channels/{channel_id}/snoop",
                params={
                'spy': 'both',
                'app': self.app,
                'subscribeAll': 'yes',
                'snoopId': 'snoop-' + channel_id,
                }
            )
            snoop_channel_id = snoop_data['id']
            self.channels[channel_id]['snoop_channel'] = snoop_channel_id
            logger.debug(f"Snoop channel {snoop_channel_id} created")

        if channel_id.startswith("snoop-"):
            # Snoop channel entered Stasis, create an external media channel for it
            snoop_channel_id = channel_id
            for id,values in self.channels.items():
                if values.get('snoop_channel') == snoop_channel_id:
                    original_channel_id = id
                    break
            ext_media_response = await self._ari_request(
                'POST',
                f"/channels/externalMedia",
                params={
                'app': self.app,
                'external_host': f"{self.rtp_server.host}:{self.rtp_server.port}",
                'format': 'slin16',
                'channelId': 'external-media-' + original_channel_id,
                }
            )
            logger.info(f"External media channel created: {ext_media_response}")
            self.channels[original_channel_id]['external_media_channel'] = ext_media_response['id']
            self.channels[original_channel_id]['rtp_source_port'] = ext_media_response['channelvars']['UNICASTRTP_LOCAL_PORT']

        if channel_id.startswith("external-media-"):
            # External media channel entered Stasis
            for id,values in self.channels.items():
                if values.get('external_media_channel') == channel_id:
                    original_channel_id = id
                    break
            snoop_channel_id = self.channels[original_channel_id]['snoop_channel']
            external_media_channel_id = channel_id
            # Create bridge
            bridge_data = await self._ari_request(
                'POST',
                "/bridges",
                params={
                'type': 'mixing',
                'bridgeId': 'bridge-' + original_channel_id,
                }
            )
            bridge_id = bridge_data['id']
            self.channels[original_channel_id]['bridge'] = bridge_id
            # Add channels to the bridge
            logger.debug(f"Adding channel {snoop_channel_id} to bridge {bridge_id}")
            await self._ari_request(
                'POST',
                f"/bridges/{bridge_id}/addChannel",
                params={'channel': snoop_channel_id}
            )
            logger.debug(f"Adding channel {external_media_channel_id} to bridge {bridge_id}")
            await self._ari_request(
                'POST',
                f"/bridges/{bridge_id}/addChannel",
                params={'channel': external_media_channel_id}
            )

            # get external media channel port and create a stream
            rtp_stream = await self.rtp_server.create_stream(self.channels[original_channel_id]['rtp_source_port'])
            # create a deepgram connector instance
            deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
            self.channels[original_channel_id]['connector'] = DeepgramConnector(
                deepgram_api_key=deepgram_api_key,
                rtp_stream=rtp_stream,
                mqtt_client=self.mqtt_client,
                uniqueid=original_channel_id,
                language=self.channels[original_channel_id]['language'],
            )
            # start the deepgram connector
            await self.channels[original_channel_id]['connector'].start()

            # Return the original channel to the dialplan
            await self._ari_request(
                'POST',
                f"/channels/{original_channel_id}/continue",
                params={}
            )
            logger.info(f"Channel {original_channel_id} returned to dialplan")

    async def _handle_stasis_end(self, event):
        """Handle channel hangup event"""
        channel = event['channel']
        channel_id = channel['id']
        if channel_id in self.channels:
            logger.debug(f"Channel {channel_id} snooped and continue to dialplan")
            return

    async def _handle_channel_left_bridge(self, event):
        """Handle channel left bridge event"""
        channel = event['channel']
        channel_id = channel['id']
        logger.debug(f"_handle_channel_left_bridge(channel_id={channel_id})")
        original_channel_id = None
        for id, values in self.channels.items():
            if values.get('snoop_channel') == channel_id:
                # Snoop channel left the bridge
                logger.debug(f"Snoop channel {channel_id} left the bridge")
                original_channel_id = id
                break
            if values.get('external_media_channel') == channel_id:
                # External media channel left the bridge
                logger.debug(f"External media channel {channel_id} left the bridge")
                original_channel_id = id
                break
        if original_channel_id is not None:
            await self.close_channel(original_channel_id)

    async def close_channel(self, channel_id):
        """Close a channel"""
        logger.debug(f"close_channel(channel_id={channel_id})")
        if channel_id in self.channels:
            # Close the deepgram connector
            if 'connector' in self.channels[channel_id]:
                try:
                    await self.channels[channel_id]['connector'].close()
                except Exception as e:
                    logger.debug(f"Failed to close connector for channel {channel_id}: {e}")
                del self.channels[channel_id]['connector']
            # Remove the bridge
            if 'bridge' in self.channels[channel_id]:
                try:
                    await self._ari_request(
                        'DELETE',
                        f"/bridges/{self.channels[channel_id]['bridge']}"
                    )
                except Exception as e:
                    logger.debug(f"Failed to delete bridge {self.channels[channel_id]['bridge']}: {e}")
                del self.channels[channel_id]['bridge']
            # Remove the external media channel
            if 'external_media_channel' in self.channels[channel_id]:
                try:
                    await self._ari_request(
                        'DELETE',
                        f"/channels/{self.channels[channel_id]['external_media_channel']}"
                    )
                except Exception as e:
                    logger.debug(f"Failed to delete external media channel {self.channels[channel_id]['external_media_channel']}: {e}")
                del self.channels[channel_id]['external_media_channel']
            # Remove the snoop channel
            if 'snoop_channel' in self.channels[channel_id]:
                try:
                    await self._ari_request(
                        'DELETE',
                        f"/channels/{self.channels[channel_id]['snoop_channel']}"
                    )
                except Exception as e:
                    logger.debug(f"Failed to delete snoop channel {self.channels[channel_id]['snoop_channel']}: {e}")
                del self.channels[channel_id]['snoop_channel']
            # Remove the RTP stream
            if 'rtp_source_port' in self.channels[channel_id]:
                self.rtp_server.end_stream(self.channels[channel_id]['rtp_source_port'])
                del self.channels[channel_id]['rtp_source_port']
            del self.channels[channel_id]

    async def _handle_channel_hangup(self, event):
        """Handle channel hangup event"""
        channel = event['channel']
        channel_id = channel['id']
        logger.debug(f"_handle_channel_hangup(channel_id={channel_id})")
        if channel_id in self.channels:
            logger.error(f"Channel {channel_id} hangup: {event}")
            await self.close_channel(channel_id)
            return
        original_channel_id = None
        for id, values in self.channels.items():
            if values.get('snoop_channel') == channel_id:
                # Snoop channel hangup
                logger.debug(f"Snoop channel {channel_id} hangup")
                original_channel_id = id
                break
            if values.get('external_media_channel') == channel_id:
                # External media channel hangup
                logger.debug(f"External media channel {channel_id} hangup")
                original_channel_id = id
                break
        if original_channel_id is not None:
            await self.close_channel(original_channel_id)

    async def _ari_request(self, method, endpoint, params=None, json_data=None):
        """Make a request to the Asterisk ARI"""
        logger.debug(f"ARI request(method={method}, endpoint={endpoint})")
        url = f"{self.url}/ari{endpoint}"

        async with self.session.request(
            method,
            url,
            params=params,
            json=json_data
        ) as response:
            if response.status >= 400:
                error_text = await response.text()
                logger.error(f"ARI request failed: {response.status} - {error_text}")
                raise Exception(f"ARI request failed: {response.status}")

            # Handle 204 No Content responses
            if response.status == 204:
                return None

            return await response.json()
