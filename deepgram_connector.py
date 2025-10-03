'''
connect rtp stream with deepgram retrieving transcriptions and publishing it to mqtt
'''

import asyncio
import json
import logging
import numpy as np
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
    def __init__(self, deepgram_api_key, rtp_stream_in, rtp_stream_out, mqtt_client, uniqueid, **kwargs):
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
        self.rtp_stream_in = rtp_stream_in
        self.rtp_stream_out = rtp_stream_out
        self.language = kwargs.get("language", "en")
        self.speaker_name_in = kwargs.get("speaker_name_in", None)
        self.speaker_number_in = kwargs.get("speaker_number_in", None)
        self.speaker_name_out = kwargs.get("speaker_name_out", None)
        self.speaker_number_out = kwargs.get("speaker_number_out", None)
        self.audio_queue = asyncio.Queue(maxsize=100)
        self.connected = False
        self.dg_connection = None
        self.loop = None
        self.complete_call = []

    async def start(self):
        deepgram: DeepgramClient = DeepgramClient(self.deepgram_api_key)
        self.dg_connection = deepgram.listen.asyncwebsocket.v("1")
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
            multichannel=True,
            channels=2,
            sample_rate=16000,
            ## To get UtteranceEnd, the following must be set:
            interim_results=True,
            utterance_end_ms="1000",
            vad_events=True,
        )
        if await self.dg_connection.start(options) is False:
            logger.error("Failed to start connection")
            return

        self.connected = True
        self.read_audio_from_rtp_task = asyncio.create_task(self.read_audio_from_rtp())
        self.send_audio_to_deepgram_task = asyncio.create_task(self.send_audio_to_deepgram())

    async def on_message(self, client, result, **kwargs):
        """
        Send transcription to mqtt
        """
        #logger.debug(f"Transcription received: {result}")
        transcription = result.channel.alternatives[0].transcript
        if len(transcription) == 0:
            #logger.debug("Empty transcription received")
            return
        #logger.debug(f"Transcription received: {result}")
        timestamp = result.start
        #logger.debug(f"Transcription {self.uniqueid} is_final: {result.is_final} speech_final: {result.speech_final} : {transcription}")
        if result.channel_index[0] == 0:
            speaker_name = self.speaker_name_in
            speaker_number = self.speaker_number_in
            speaker_counterpart_name = self.speaker_name_out
            speaker_counterpart_number = self.speaker_number_out
        else:
            speaker_name = self.speaker_name_out
            speaker_number = self.speaker_number_out
            speaker_counterpart_name = self.speaker_name_in
            speaker_counterpart_number = self.speaker_number_in
        try:
            await self.mqtt_client.publish(
                    topic='transcription',
                    payload=json.dumps({
                        "uniqueid": self.uniqueid,
                        "transcription": transcription,
                        "timestamp": timestamp,
                        "speaker_name": speaker_name,
                        "speaker_number": speaker_number,
                        "speaker_counterpart_name": speaker_counterpart_name,
                        "speaker_counterpart_number": speaker_counterpart_number,
                        "is_final": result.is_final,
                    })
                )
            # save the transcription to the complete_call if it is final
            if result.is_final:
                self.complete_call.append({
                    "uniqueid": self.uniqueid,
                    "transcription": transcription,
                    "timestamp": timestamp,
                    "speaker_name": speaker_name,
                    "speaker_number": speaker_number,
                    "speaker_counterpart_name": speaker_counterpart_name,
                    "speaker_counterpart_number": speaker_counterpart_number,
                    "is_final": result.is_final,
                })
        except Exception as e:
                logger.error(f"Failed to schedule transcription publishing: {e}")

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
            target_size = 5120
            timeout = 0.25 # 250ms timeout
            while self.connected:
                # Read audio data from both streams till target size or timeout is reached
                buffer_in = bytearray()
                buffer_out = bytearray()
                start_time = asyncio.get_event_loop().time()
                while (len(buffer_in) < target_size and
                       len(buffer_out) < target_size and
                       (asyncio.get_event_loop().time() - start_time) < timeout):
                    if len(buffer_in) < target_size:
                        audio_data_in = self.rtp_stream_in.reader.read(320)
                        if audio_data_in:
                            buffer_in.extend(audio_data_in)
                    if len(buffer_out) < target_size:
                        audio_data_out = self.rtp_stream_out.reader.read(320)
                        if audio_data_out:
                            buffer_out.extend(audio_data_out)
                # If we have no data in both buffers after timeout, wait and continue
                if len(buffer_in) == 0 and len(buffer_out) == 0:
                    await asyncio.sleep(0.1)
                    continue
                # Convert buffers to numpy arrays
                arr1 = np.frombuffer(buffer_in, dtype=np.int16)
                arr2 = np.frombuffer(buffer_out, dtype=np.int16)

                # Ensure arrays are the same size by padding the smaller one
                if arr1.size != arr2.size:
                    if arr1.size < arr2.size:
                        arr1 = np.pad(arr1, (0, arr2.size - arr1.size), 'constant')
                    else:
                        arr2 = np.pad(arr2, (0, arr1.size - arr2.size), 'constant')
                # Interleave the audio data
                interleaved = np.empty((arr1.size + arr2.size,), dtype=np.int16)
                interleaved[0::2] = arr1
                interleaved[1::2] = arr2
                # Put interleaved audio data into the queue
                await self.audio_queue.put(interleaved.tobytes())
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
                await self.dg_connection.send(audio_data)
        except Exception as e:
            logger.error(f"Error sending audio to Deepgram: {e}")
            self.connected = False
            await self.close()

    async def close(self):
        """
        Close the connection to Deepgram
        """
        logger.debug(f"Closing Deepgram connection for {self.uniqueid}")
        self.connected = False
        await self.dg_connection.finalize()
        await self.dg_connection._socket.close()
        self.read_audio_from_rtp_task.cancel()
        self.send_audio_to_deepgram_task.cancel()
        # publish full conversation to mqtt
        text = ""
        last_speaker = None
        for message in self.complete_call:
            if last_speaker != message["speaker_name"]:
                text += f'\n{message["speaker_name"]}: '
            text += f'{message["transcription"]}\n'
            last_speaker = message["speaker_name"]

        # publish the full conversation to mqtt
        await self.mqtt_client.publish(
            topic='final',
            payload=json.dumps({
                "uniqueid": self.uniqueid,
                "raw_transcription": text
            })
        )
        # Process AI summary in background without blocking
        asyncio.create_task(self._process_ai_summary(text))

    async def _process_ai_summary(self, text):
        """Process AI summary and cleaning in background thread"""
        try:
            from ai import get_summary, get_clean
            # Run synchronous AI calls in a thread to avoid blocking the event loop
            clean_text = await asyncio.to_thread(get_clean, text)
            await self.mqtt_client.publish(
                topic='final',
                payload=json.dumps({
                    "uniqueid": self.uniqueid,
                    "clean_transcription": clean_text
                })
            )
            summary = await asyncio.to_thread(get_summary, text)
            await self.mqtt_client.publish(
                topic='final',
                payload=json.dumps({
                    "uniqueid": self.uniqueid,
                    "summary": summary
                })
            )
        except Exception as e:
            logger.error(f"Error processing transcription: {e}")

