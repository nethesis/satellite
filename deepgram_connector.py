'''
connect rtp stream with deepgram retrieving transcriptions and publishing it to mqtt
'''

import asyncio
import json
import logging
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
)

logger = logging.getLogger("deepgram_connector")
logging.getLogger('websockets').setLevel(logging.INFO)

class DeepgramConnector:
    """
    Connect to Deepgram
    - send audio stream to deepgram
    - retrieve transcriptions
    - call transcription callback
    """
    def __init__(self, deepgram_api_key, rtp_stream, mqtt_client, uniqueid, language="en"):
        """
        Initialize Deepgram Connector
        :param deepgram_api_key: Deepgram API key
        :param mqtt_client: MQTT client
        :param audio_stream: Audio stream to send to Deepgram
        :param uniqueid: Asterisk Channel ID (uniqueid)
        """
        self.deepgram_api_key = deepgram_api_key
        self.mqtt_client = mqtt_client
        self.uniqueid = uniqueid
        self.transcription_callback = None
        self.read_audio_from_rtp_task = None
        self.send_audio_to_deepgram_task = None
        self.get_transcription_task = None
        self.rtp_stream = rtp_stream
        self.language = language
        self.audio_queue = asyncio.Queue(maxsize=100)
        self.connected = False
        self.dg_connection = None
        self.loop = None
        self.complete_call = []

    async def start(self):
        deepgram: DeepgramClient = DeepgramClient()
        self.dg_connection = deepgram.listen.websocket.v("1")
        self.dg_connection.on(LiveTranscriptionEvents.Transcript, self.on_message)
        self.dg_connection.on(LiveTranscriptionEvents.Metadata, self.on_metadata)
        self.dg_connection.on(LiveTranscriptionEvents.Error, self.on_error)
        # Store the current event loop for later use in callbacks
        self.loop = asyncio.get_running_loop()

        options: LiveOptions = LiveOptions(
            model="nova-2",
            punctuate=True,
            language=self.language,
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            ## To get UtteranceEnd, the following must be set:
            interim_results=True,
            utterance_end_ms="1000",
            vad_events=True,
        )
        if self.dg_connection.start(options) is False:
            logger.error("Failed to start connection")
            return

        self.connected = True
        self.read_audio_from_rtp_task = asyncio.create_task(self.read_audio_from_rtp())
        self.send_audio_to_deepgram_task = asyncio.create_task(self.send_audio_to_deepgram())

    def on_message(self, client, result, **kwargs):
        """
        Send transcription to mqtt
        """
        #logger.debug(f"Transcription received: {result}")
        transcription = result.channel.alternatives[0].transcript
        if len(transcription) == 0:
            #logger.debug("Empty transcription received")
            return

        logger.debug(f"Transcription {self.uniqueid} is_final: {result.is_final} speech_final: {result.speech_final} : {transcription}")
        # Use the stored event loop to schedule the task
        try:
            self.loop.call_soon_threadsafe(
                lambda: asyncio.create_task(self.publish_transcription(transcription, is_final=result.is_final, speech_final=result.speech_final))
            )
        except Exception as e:
                logger.error(f"Failed to schedule transcription publishing: {e}")

    async def publish_transcription(self, transcription, is_final=False, speech_final=False, **kwargs):
        """Async helper to publish transcription to MQTT"""
        payload = {
            "uniqueid": self.uniqueid,
            "transcription": transcription,
            "is_final": is_final,
            "speech_final": speech_final,
        }
        await self.mqtt_client.publish(
            topic="transcription",
            payload=json.dumps(payload),
        )
        self.complete_call.append(payload)

    def on_metadata(self, client, metadata, **kwargs):
        """
        Handle metadata events
        """
        #logger.debug(f"Metadata received: {metadata}")
        return

    def on_error(self, client, error, **kwargs):
        """
        Handle error events
        """
        logger.error(f"Error received: {error}")
        # Use the stored event loop to schedule the task
        try:
            if hasattr(self, 'loop') and self.loop.is_running():
                self.loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(self.close())
                )
            else:
                logger.warning("No running event loop available, using default scheduling")
                asyncio.run_coroutine_threadsafe(self.close(), asyncio.get_event_loop())
        except Exception as e:
                logger.error(f"Failed to schedule close operation: {e}")

    async def read_audio_from_rtp(self):
        """
        Read audio from RTP stream
        """
        logger.debug(f"Reading audio from RTP stream for {self.uniqueid}")
        try:
            while self.connected:
                audio_data = self.rtp_stream.reader.read(3200)  # Read 100ms of audio at 16kHz
                if not audio_data:
                    await asyncio.sleep(0.1)
                    continue
                await self.audio_queue.put(audio_data)
        except Exception as e:
            logger.error(f"Error reading audio from RTP stream: {e}")
            self.connected = False
            await self.close()

    async def send_audio_to_deepgram(self):
        """
        Read audio from queue and send to Deepgram
        """
        logger.debug(f"Sending audio to Deepgram for {self.uniqueid}")
        try:
            while self.connected:
                if self.audio_queue.empty():
                    await asyncio.sleep(0.1)
                    continue
                audio_data = await self.audio_queue.get()
                if audio_data is None:
                    await asyncio.sleep(0.1)
                    continue
                self.dg_connection.send(audio_data)
        except Exception as e:
            logger.error(f"Error sending audio to Deepgram: {e}")
            self.connected = False
            await self.close()

    async def close(self):
        """
        Close the connection to Deepgram
        """
        logger.debug(f"Closing Deepgram connection for {self.uniqueid}")
        # publish complete call transcription
        payload = {
            "uniqueid": self.uniqueid,
            "transcription": "",
        }
        for item in self.complete_call:
            if item["is_final"]:
                payload["transcription"] += item["transcription"] + "\n"
        await self.mqtt_client.publish(topic="final_transcription", payload=json.dumps(payload))
        self.connected = False
        self.dg_connection.finalize()
        self.dg_connection._socket.close()
        self.read_audio_from_rtp_task.cancel()
        self.send_audio_to_deepgram_task.cancel()
