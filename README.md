# Satellite

Satellite is a Python application that creates a bridge between Asterisk PBX and Deepgram speech recognition services. It connects to Asterisk ARI (Asterisk REST Interface) and waits for channels to enter stasis. When a channel enters stasis with the application name "satellite", it creates a snoop channel and sends external media to its RTP server address. The RTP server distinguishes various channels from the UDP source port, captures the audio, and forwards it to Deepgram for real-time speech-to-text transcription. Transcription results are then published to an MQTT broker for further processing.

## Features

- Connects to Asterisk ARI via WebSockets
- Creates snoop channels to capture audio from Asterisk calls
- Streams audio using RTP protocol
- Real-time speech-to-text transcription via Deepgram
- Publishes transcription results to MQTT
- Handles multiple concurrent channels

## Requirements

- Python 3.12+
- Asterisk PBX with ARI enabled
- MQTT broker
- Deepgram API key

## Installation

1. Clone this repository
2. Create a virtual environment: `python3 -m venv .venv`
3. Activate the virtual environment: `source .venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`

## Configuration

Create a `.env` file in the root directory with the following configuration parameters:

```
# Asterisk Configuration
ASTERISK_URL=http://127.0.0.1:8088
ARI_APP=satellite
ARI_USERNAME=satellite
ARI_PASSWORD=your_password
ASTERISK_FORMAT=slin16

# RTP Server Configuration
RTP_HOST=0.0.0.0
RTP_PORT=10000
RTP_SWAP16=true
RTP_HEADER_SIZE=12

# MQTT Configuration
MQTT_URL=mqtt://127.0.0.1:1883
MQTT_TOPIC_PREFIX=satellite

# Deepgram API Key
DEEPGRAM_API_KEY=your_deepgram_api_key
```

### Configuration Parameters

#### Asterisk Configuration
- `ASTERISK_URL`: URL of your Asterisk ARI server
- `ARI_APP`: Stasis application name
- `ARI_USERNAME`: ARI username
- `ARI_PASSWORD`: ARI password
- `ASTERISK_FORMAT`: Audio format (slin16 for 16-bit signed linear PCM)

#### RTP Server Configuration
- `RTP_HOST`: IP address to bind the RTP server to (0.0.0.0 for all interfaces)
- `RTP_PORT`: UDP port for the RTP server
- `RTP_SWAP16`: Set to "true" if byte-swapping is needed for audio (depends on Asterisk configuration)
- `RTP_HEADER_SIZE`: Size of RTP header in bytes (typically 12)

#### MQTT Configuration
- `MQTT_URL`: URL of the MQTT broker
- `MQTT_TOPIC_PREFIX`: Prefix for MQTT topics

#### Deepgram Configuration
- `DEEPGRAM_API_KEY`: Your Deepgram API key

## Usage

1. Ensure Asterisk is configured with the appropriate ARI settings
2. Make sure your MQTT broker is running
3. Run the application: `python main.py`
4. Configure Asterisk dialplan to direct calls to the Stasis application named "satellite"

## Architecture

Satellite consists of several key components:

1. **AsteriskBridge**: Connects to Asterisk ARI and manages call channels
2. **RTPServer**: Receives and processes RTP audio streams
3. **MQTTClient**: Publishes transcription results to MQTT
4. **DeepgramConnector**: Streams audio to Deepgram and receives transcriptions

When a call enters the Stasis application in Asterisk:
1. A snoop channel is created to capture audio
2. An external media endpoint is set up for RTP streaming
3. A bridge connects the snoop channel and external media endpoint
4. RTP audio is sent to Deepgram for transcription
5. Transcription results are published to MQTT

## MQTT Topics

The application publishes transcription results to the following MQTT topic:
- `transcription`: Contains JSON with transcript text, channel ID, and flags for final/interim results

