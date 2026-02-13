# Satellite

Satellite is a Python application that creates a bridge between Asterisk PBX and Deepgram speech recognition services. It connects to Asterisk ARI (Asterisk REST Interface) and waits for channels to enter stasis. When a channel enters stasis with the application name "satellite", it creates a snoop channel and sends external media to its RTP server address. The RTP server distinguishes various channels from the UDP source port, captures the audio, and forwards it to Deepgram for real-time speech-to-text transcription. Transcription results are then published to an MQTT broker for further processing.
If OpenAI API key is provided, it will be used to generate a summary of the transcriptions.

## Features

- Connects to Asterisk ARI via WebSockets
- Creates snoop channels to capture audio from Asterisk calls
- Streams audio using RTP protocol
- Real-time speech-to-text transcription via Deepgram
- Publishes transcription results to MQTT
- Handles multiple concurrent channels
- (Optional) Persists transcriptions + vector embeddings to Postgres/pgvector

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
SATELLITE_ARI_PASSWORD=your_password
ASTERISK_FORMAT=slin16

# RTP Server Configuration
RTP_HOST=0.0.0.0
RTP_PORT=10000
RTP_SWAP16=true
RTP_HEADER_SIZE=12

# MQTT Configuration
MQTT_URL=mqtt://127.0.0.1:1883
MQTT_TOPIC_PREFIX=satellite

# Transcription Provider (optional, default: deepgram)
# Options: deepgram, voxtral
TRANSCRIPTION_PROVIDER=deepgram

# Deepgram API Key (required for Deepgram provider)
DEEPGRAM_API_KEY=your_deepgram_api_key

# Mistral API Key (required for VoxTral provider)
MISTRAL_API_KEY=your_mistral_api_key

# REST API (optional)
HTTP_PORT=8000

# REST API Authentication (optional)
# When set, all /api/* endpoints require an auth header.
API_TOKEN=your_static_api_token

# OpenAI API Key (optional)
OPENAI_API_KEY=your_openai_api_key

# Log level (optional)
LOG_LEVEL=DEBUG

# PGSQL Vectorstore Configuration
PGVECTOR_HOST=localhost
PGVECTOR_PORT=5432
PGVECTOR_USER=postgres
PGVECTOR_PASSWORD=your_password
PGVECTOR_DATABASE=satellite
```

### Configuration Parameters

#### Asterisk Configuration
- `ASTERISK_URL`: URL of your Asterisk ARI server
- `ARI_APP`: Stasis application name
- `ARI_USERNAME`: ARI username
- `SATELLITE_ARI_PASSWORD`: ARI password
- `ASTERISK_FORMAT`: Audio format (slin16 for 16-bit signed linear PCM)

#### RTP Server Configuration
- `RTP_HOST`: IP address to bind the RTP server to (0.0.0.0 for all interfaces)
- `RTP_PORT`: UDP port for the RTP server
- `RTP_SWAP16`: Set to "true" if byte-swapping is needed for audio (depends on Asterisk configuration)
- `RTP_HEADER_SIZE`: Size of RTP header in bytes (typically 12)

#### MQTT Configuration
- `MQTT_URL`: URL of the MQTT broker
- `MQTT_TOPIC_PREFIX`: Prefix for MQTT topics

#### Transcription Configuration
- `TRANSCRIPTION_PROVIDER`: Choose the transcription provider (`deepgram` or `voxtral`, default: `deepgram`)
- `DEEPGRAM_API_KEY`: Your Deepgram API key (required for Deepgram provider)
- `MISTRAL_API_KEY`: Your Mistral API key (required for VoxTral provider)

#### Rest API Configuration
- `HTTP_PORT`: Port for the HTTP server (default: 8000)
- `API_TOKEN`: Optional static token for `/api/*` endpoints. If unset/empty, auth is disabled.

#### Postgres Vectorstore Configuration
If `PGVECTOR_*` environment variables are set, `POST /api/get_transcription` can persist the raw transcription to Postgres when the request includes `persist=true` and a valid `uniqueid`.

The database schema is created automatically on first use and includes:
- `transcripts`: stores `uniqueid`, diarized raw transcription (Deepgram paragraphs transcript), `state`, optional cleaned transcription + summary, and `sentiment` (0-10)
- `transcript_chunks`: table for storing chunked `text-embedding-3-small` embeddings in a `vector(1536)` column for similarity search

`transcripts.state` is DB-only and represents the processing lifecycle:
- `progress`: request accepted and persistence row created, transcription not yet stored
- `failed`: pipeline failed (Deepgram error, parsing error, persistence error, or enrichment error)
- `summarizing`: AI enrichment running (subprocess worker)
- `done`: pipeline finished (raw transcript stored; enrichment finished if enabled)

This requires the `vector` extension (pgvector) in your Postgres instance.

## Usage

1. Ensure Asterisk is configured with the appropriate ARI settings
2. Make sure your MQTT broker is running
3. Run the application: `python main.py`
4. Configure Asterisk dialplan to direct calls to the Stasis application named "satellite"

### REST API

#### `POST /api/get_transcription`

Accepts a WAV upload and returns a transcription from the configured provider (Deepgram or VoxTral).

Request requirements:
- Content type: multipart form upload with a `file` field (`audio/wav` or `audio/x-wav`)

Optional fields (query string or multipart form fields):
- `provider`: Override the transcription provider (`deepgram` or `voxtral`). If not set, uses `TRANSCRIPTION_PROVIDER` env var (default: `deepgram`)
- `uniqueid`: Asterisk-style uniqueid like `1234567890.1234` (required only when `persist=true`)
- `persist`: `true|false` (default `false`) — persist raw transcript to Postgres (requires `PGVECTOR_*` env vars)
- `summary`: `true|false` (default `false`) — run AI enrichment (requires `OPENAI_API_KEY` and also `persist=true` so there is a DB record to update)
- `channel0_name`, `channel1_name`: rename diarization labels in the returned transcript (replaces `Channel 0:` / `Channel 1:` or `Speaker 0:` / `Speaker 1:`)

Provider-specific parameters:
- **Deepgram**: Most Deepgram `/v1/listen` parameters may be provided as query/form fields (e.g., `model`, `language`, `diarize`, `punctuate`)
- **VoxTral**: Supports `model` (default: `voxtral-mini-latest`), `language`, `diarize`, `temperature`, `context_bias`, `timestamp_granularities`

Example:
```
# Using default provider (from TRANSCRIPTION_PROVIDER env var)
curl -X POST http://127.0.0.1:8000/api/get_transcription \
    -H 'Authorization: Bearer YOUR_TOKEN' \
    -F uniqueid=1234567890.1234 \
    -F persist=true \
    -F summary=true \
    -F file=@call.wav;type=audio/wav

# Override provider to use VoxTral
curl -X POST http://127.0.0.1:8000/api/get_transcription \
    -H 'Authorization: Bearer YOUR_TOKEN' \
    -F provider=voxtral \
    -F diarize=true \
    -F file=@call.wav;type=audio/wav
```

Authentication:
- If `API_TOKEN` is set, all `/api/*` endpoints require `Authorization: Bearer <token>` (or `X-API-Token: <token>`).
- If `API_TOKEN` is unset/empty, auth is disabled (backwards compatible default).

If `persist=true` and `PGVECTOR_*` is configured, the raw transcription is saved to Postgres.
If `summary=true` and `OPENAI_API_KEY` is set, the service also generates a cleaned transcription, summary, and sentiment score (0-10) via a per-request subprocess worker (`call_processor.py`) and stores them in Postgres.
If `OPENAI_API_KEY` is missing (or `persist=false`), clean/summary/sentiment are skipped.

When `persist=true`, `POST /api/get_transcription` updates `transcripts.state` as it runs: `progress` → (`summarizing` →) `done`, or `failed` on errors.

## Architecture

Satellite consists of several key components:

1. **AsteriskBridge**: Connects to Asterisk ARI and manages call channels
2. **RTPServer**: Receives and processes RTP audio streams
3. **MQTTClient**: Publishes transcription results to MQTT
4. **DeepgramConnector**: Streams audio to Deepgram and receives transcriptions
5. **AI**: (optional) Generates summaries of transcriptions using OpenAI API

When a call enters the Stasis application in Asterisk:
1. A snoop channel is created to capture audio
2. An external media endpoint is set up for RTP streaming
3. A bridge connects the snoop channel and external media endpoint
4. RTP audio is sent to Deepgram for transcription
5. Transcription results are published to MQTT

## MQTT Topics

The application publishes transcription results to the following MQTT topic:
- `transcription`: Contains JSON with transcript text, channel ID, and flags for final/interim results

## Testing

Set variables
```
export ASTERISK_URL=http://127.0.0.1:8088
export ARI_APP=satellite
export ARI_USERNAME=satellite
export SATELLITE_ARI_PASSWORD=aripassword
export ASTERISK_FORMAT=slin16
export RTP_HOST=0.0.0.0
export RTP_PORT=10000
export RTP_SWAP16=true
export RTP_HEADER_SIZE=12
export MQTT_URL=mqtt://127.0.0.1:1883
export MQTT_TOPIC_PREFIX=satellite
export MQTT_USERNAME=mqttuser
export SATELLITE_MQTT_PASSWORD=mqttpass
export DEEPGRAM_API_KEY=XXX
export HTTP_PORT=8080

# Optional: enable Postgres persistence in tests/manual runs
export PGVECTOR_HOST=localhost
export PGVECTOR_PORT=5432
export PGVECTOR_USER=postgres
export PGVECTOR_PASSWORD=your_password
export PGVECTOR_DATABASE=satellite

# Optional: enable clean/summary/embeddings
export OPENAI_API_KEY=your_openai_api_key
```

### MQTT Broker

Create MQTT password file
```
podman run -it docker.io/library/eclipse-mosquitto sh -c 'touch /mosquitto_passwd ; chmod 0700 /mosquitto_passwd ; mosquitto_passwd -b /mosquitto_passwd '$MQTT_USERNAME' '$SATELLITE_MQTT_PASSWORD'; cat /mosquitto_passwd' > ./mosquitto_passwd
```
Create MQTT config
```
cat << EOF > mosquitto.conf
password_file /mosquitto_passwd
allow_anonymous false
listener $MQTT_PORT
EOF
```
Run MQTT broker
```
podman run -d --name mqtt --replace -v=./mosquitto_passwd:/mosquitto_passwd:Z -v=./mosquitto.conf:/mosquitto/config/mosquitto.conf:Z --network=host docker.io/library/eclipse-mosquitto
```

### Asterisk

in Asterisk dialplan, add this before the dial command
```
exten => s,n,Stasis(satellite,${EXTEN})
```
in /etc/asterisk/ari.conf
```
[satellite]
type=user
password=$SATELLITE_ARI_PASSWORD
password_format=plain
read_only=no
```
Also make sure that asterisk http server is enabled on port specified in ASTETRISK_URL

### Satellite

Run the application
```
git clone ... && cd satellite
python main.py
```

Run the docker container
```
podman run -e ASTERISK_URL -e MQTT_URL -e DEEPGRAM_API_KEY ... satellite
```

## License
This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.
