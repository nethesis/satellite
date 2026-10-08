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
- MQTT broker and Deepgram API key for optional transcription
- Provider API key and signed webhook binding for built-in voice agents

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

# Deepgram API Key
DEEPGRAM_API_KEY=your_deepgram_api_key

# REST API (listens on loopback by default)
HTTP_HOST=127.0.0.1
HTTP_PORT=8000

# REST API Authentication (required for /api/*)
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

#### Deepgram Configuration
- `DEEPGRAM_API_KEY`: Your Deepgram API key

#### Rest API Configuration
- `HTTP_PORT`: Port for the HTTP server (default: 8000)
- `HTTP_HOST`: HTTP listen address. Defaults to `127.0.0.1`.
- `API_TOKEN`: Required for `/api/*`. An unset or blank server token returns `503`; invalid request credentials return `401`.

#### Built-in Satellite Agent
- `SATELLITE_AGENT_ARI_APP`: Separate Agent Stasis application (default: `satellite-agent`). The transcription application continues to use `ARI_APP`.
- `SATELLITE_AGENT_STATE_PATH`: Optional persistent path for the accepted configuration revision and webhook receipt hashes. This file contains no provider credentials.
- `API_TOKEN`: Required for `/api/agent/v1/*` and for built-in call readiness. The legacy `/api/*` surface also requires authentication.

The built-in voice runtime runs in the same process and event loop as the HTTP API. It can run without `DEEPGRAM_API_KEY`; transcription and RTP/MQTT services remain optional. FreePBX sends complete versioned configuration to `PUT /api/agent/v1/configuration` and current directory/calendar data to `PUT /api/agent/v1/context`. Calls entering `Stasis(satellite-agent,caller,...)` are matched to a pinned destination and provider binding. The runtime originates a retained `Local/...@satellite-agent-provider/n` leg and bridges it after the signed provider event is correlated and both legs are ready.

Build the runtime image with this Satellite source checkout as the container build context so the `agent/` package is included. The NS8 module repository consumes the resulting runtime image separately.

Agent endpoints include `/readiness`, `/catalog/tools`, `/catalog/permissions`, `/provider-events/{openai|grok}`, and `/calls/{session_id}` control/status under `/api/agent/v1`. Provider events require the shared Bearer token and an independently valid provider webhook signature. The Agent API does not expose provider credentials. On setup failure the original caller returns to the generated FreePBX fallback; a committed basic handoff continues directly to a configured extension, queue, or IVR.

#### Postgres Vectorstore Configuration
If `PGVECTOR_*` environment variables are set, `POST /api/get_transcription` can persist the raw transcription to Postgres when the request includes `persist=true` and a valid `uniqueid`.

The database schema is created automatically on first use and includes:
- `transcripts`: stores `uniqueid`, optional `linkedid`, and optional `src_number` and `dst_number` participant numbers, diarized raw transcription (Deepgram paragraphs transcript), `state`, optional cleaned transcription + summary, and `sentiment` (0-10). `uniqueid` is indexed but not unique, so transferred calls can persist multiple fragments under the same Asterisk call identifier while Satellite tracks each stored fragment by its internal `id`.
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
5. Send an MQTT message to the topic `transcription/control` with payload `{"action":"start", "uniqueid":"[CALL_UNIQUEID]"}` or `{"action":"start", "linkedid":"[CALL_UNIQUEID]"}`
6. Stop the trascription with MQTT message to the topic `transcription/control` with payload `{"action":"stop", "uniqueid":"[CALL_UNIQUEID]"}` or `{"action":"stop", "linkedid":"[CALL_UNIQUEID]"}`

### REST API

#### `POST /api/get_transcription`

Accepts a WAV upload and returns a Deepgram transcription.

Request requirements:
- Content type: multipart form upload with a `file` field (`audio/wav` or `audio/x-wav`)

Optional fields (query string or multipart form fields):
- `uniqueid`: Asterisk-style uniqueid like `1234567890.1234` (required only when `persist=true`)
- `persist`: `true|false` (default `false`) — persist raw transcript to Postgres (requires `PGVECTOR_*` env vars)
- `summary`: `true|false` (default `false`) — run AI enrichment (requires `OPENAI_API_KEY` and also `persist=true` so there is a DB record to update)
- `linkedid`: optional linked Asterisk call id stored with the persisted transcript row when `persist=true`
- `src_number`, `dst_number`: optional participant numbers stored with the persisted transcript row when `persist=true`
- `channel0_name`, `channel1_name`: rename diarization labels in the returned transcript (replaces `Channel 0:` / `Channel 1:`)

Deepgram parameters:
- Most Deepgram `/v1/listen` parameters may be provided as query/form fields and are passed through to Deepgram.

Example:
```
curl -X POST http://127.0.0.1:8000/api/get_transcription \
    -H 'Authorization: Bearer YOUR_TOKEN' \
    -F uniqueid=1234567890.1234 \
    -F persist=true \
    -F summary=true \
    -F file=@call.wav;type=audio/wav
```

Authentication:
- All `/api/*` endpoints require `Authorization: Bearer <token>` (or `X-API-Token: <token>`).
- If `API_TOKEN` is unset or blank, all `/api/*` requests return `503`.

If `persist=true` and `PGVECTOR_*` is configured, the raw transcription is saved to Postgres.
Each persisted request creates or updates its own transcript row by internal `id`; repeated `uniqueid` values are allowed for multi-fragment call recordings.
If `summary=true` and `OPENAI_API_KEY` is set, the service also generates a cleaned transcription, summary, and sentiment score (0-10) via a per-request subprocess worker (`call_processor.py`) and stores them in Postgres.
If `OPENAI_API_KEY` is missing (or `persist=false`), clean/summary/sentiment are skipped.

When `persist=true`, `POST /api/get_transcription` updates `transcripts.state` as it runs: `progress` → (`summarizing` →) `done`, or `failed` on errors.
If Deepgram returns `results.channels: []` for silent or zero-duration audio, Satellite returns `200` with an empty transcript and, when persistence is enabled, marks the row as `done` so the caller can discard the source audio instead of retrying forever.

#### `POST /api/get_speech`

Accepts text input and returns a Deepgram text-to-speech (TTS) MP3 stream.

Request requirements:
- Provide `text` (or `input`) as either query string parameters or as form fields (`application/x-www-form-urlencoded` or multipart form)

Model selection:
- `model` is optional when `language` is provided.
- If `model` is missing and `language` is set, Satellite uses `GET /api/get_models?language=<language>` internally and picks the first model in the returned list.
- If no model matches the requested language, the API returns `400`.

#### `GET /api/get_models`

Returns the available Deepgram TTS models known by Satellite.

Parameters:
- `language` (optional): filter models by suffix. Only models ending with `-<language>` are returned.

Response `200`:
- `{ "models": ["aura-2-melia-it", ...] }`

Example:
```
curl -X GET 'http://127.0.0.1:8000/api/get_models?language=it' \
    -H 'Authorization: Bearer YOUR_TOKEN'
```

Deepgram TTS parameters:
- `model`: Deepgram TTS voice model. See [Deepgram TTS docs](https://developers.deepgram.com/docs/tts-models) for available models. Note that language is inferred from the model, choose a model that matches your text language. See all available languages/models [here](https://developers.deepgram.com/docs/tts-models).
- Output is MP3-only. Requests with non-MP3 `encoding` or `container` return `400`.

Italian Voices:

| Model | Name | Expressed Gender | Age | Language | Accent | Characteristics | Use Cases |
| --- | --- | --- | --- | --- | --- | --- | --- |
| aura-2-melia-it | melia | feminine | Adult | it-it | Italian | Clear, Comfortable, Engaging, Friendly, Natural | Casual Chat, Customer Service, Interview |
| aura-2-elio-it | elio | masculine | Adult | it-it | Italian | Breathy, Calm, Professional, Smooth, Trustworthy | Interview, Casual Chat, Customer Service |
| aura-2-flavio-it | flavio | masculine | Adult | it-it | Italian | Confident, Deep, Empathetic, Professional, Trustworthy | Casual Chat, Interview, Customer Service |
| aura-2-maia-it | maia | feminine | Young Adult | it-it | Italian | Caring, Energetic, Expressive, Professional, Warm | Interview, Casual Chat, Customer Service |
| aura-2-cinzia-it | cinzia | feminine | Mature | it-it | Italian | Approachable, Friendly, Smooth, Trustworthy, Warm | Customer Service, Interview, Narration |
| aura-2-cesare-it | cesare | masculine | Adult | it-it | Italian | Clear, Empathetic, Knowledgeable, Natural, Smooth | Casual Chat, Customer Service, Interview, IVR |
| aura-2-livia-it | livia | feminine | Adult | it-it | Italian | Approachable, Cheerful, Clear, Engaging, Expressive | Customer Service, Interview, Audiobook |
| aura-2-perseo-it | perseo | masculine | Young Adult | it-it | Italian | Casual, Clear, Natural, Polite, Smooth | Casual Chat, Customer Service |
| aura-2-dionisio-it | dionisio | masculine | Adult | it-it | Italian | Confident, Engaging, Friendly, Melodic, Positive | Interview, Casual Chat, Customer Service |
| aura-2-demetra-it | demetra | feminine | Adult | it-it | Italian | Calm, Comfortable, Patient | Casual Chat, Interview, Narration |


- These are passed through to Deepgram `/v1/speak` when provided: `sample_rate`, `bit_rate`, `mip_opt_out`, `tag`, `callback`, `callback_method`.
- `encoding` is forced to `mp3`; `container` must be omitted (if set to `mp3`, it is ignored).

Response:
- `Content-Type: audio/mpeg`
- `Content-Disposition: attachment; filename="speech-<uuid>.mp3"`

Example:
```
curl -X POST http://127.0.0.1:8000/api/get_speech \
    -H 'Authorization: Bearer YOUR_TOKEN' \
    -d 'text=Hello from Satellite' \
    --output speech.mp3
```

Notes:
- Text is split into 2000-character chunks (Deepgram input limit) and each chunk is synthesized sequentially; the resulting MP3 parts are concatenated.
- Errors: `400` for missing text, `401` for missing/invalid credentials, `503` when the server token is unset, `504` on Deepgram timeout, `502` if Deepgram is unreachable.

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

## Agent runtime on the `agent` branch

The image `ghcr.io/nethesis/satellite:agent` includes voice agents, encrypted
monitoring history, and configurable business connectors with an OpenAI Responses
text executor. Existing transcription and MQTT interfaces remain compatible.
See [monitoring API](docs/agents/monitoring-api-contract.md),
[application API](docs/agents/phase4-api-contract.md) and
[workflow API](docs/agents/workflow-api-contract.md) for configuration, capture
controls, permissions, scoped API clients and deployment requirements.

Enabled tools are appended to the effective provider instructions automatically.
Call transfer includes only permitted, visible destinations with native extension
display names, queue names and IVR names, plus configured descriptions and aliases.
The model selects an approved destination ID; trusted PBX routing stays on the
server. Disabling a tool or permission removes that capability from the prompt.

### OpenAI Realtime and GPT-Live

Voice agents support OpenAI Realtime and GPT-Live on the same OpenAI binding.
Set the built-in profile `model` or workflow `voice_settings.model` to
`gpt-live-1` for Live, or `gpt-realtime` for Realtime. An empty workflow override
inherits the external profile. Existing stored models keep their behavior.
Live uses the same project key and a managed `gpt-6-luna` Responses backend.
Enable Live SIP for the project, retain the `realtime.call.incoming` webhook
subscription, and add `live.transport.incoming`. The provider-facing media leg
must support SRTP; the NethVoice proxy handles the external media negotiation.

Live advances workflow steps through structured tools and semantic prompt
readiness rather than audio-drain events. Configured DTMF confirmations and
exact-argument write approvals still apply. A private consultation session is
replaced before returning to the caller. See the maintained
[workflow contract](docs/agents/workflow-api-contract.md) for these semantics.
`tests/test_agent_live.py` exercises the Live protocol through local HTTP and
WebSocket fixtures without provider credentials.

### Offline OpenAI protocol tests

Run `pytest tests/test_agent_openai_emulator.py tests/test_agent_live.py`. The local aiohttp simulator
emulates accept/hangup HTTP endpoints and the Realtime sideband WebSocket; signed
`realtime.call.incoming` webhooks enter the real FastAPI route. Model tool answers
arrive as `response.function_call_arguments.done` and `response.done` events.
The suite tests all built-in functions, named extension/queue/IVR transfers,
duplicate delivery, authentication, replay/correlation, denied capabilities,
invalid requests, external permission limits and ARI transfer failures.
It uses a fake ARI client and synthetic configuration; no SIP call, production
credentials or OpenAI account is required.

`test_application_acceptance.py` and `test_monitoring_acceptance.py` guard their
storage tests with explicit isolated database host names and acceptance flags.
Never point these destructive fixture tests at an existing database.

### Agent workflows

The runtime executes published block graphs for voice and API agents. Templates
include a call router, customer support and a payment secretary. Connectors,
data sources and reusable blocks use pinned published versions. The payment
secretary supports text, CSV, XLSX, private Google Sheets and published Google CSV.
Consultative transfer requires operator acceptance before connecting the caller.

Use the NethVoice Builder to configure and publish a graph. Test with synthetic
fixtures first. Ask for confirmation before a final OpenAI test. Mock tests make
no provider calls and do not appear as live history runs. For an approved live
test, check the run status and each block outcome in the graph trace.

Run the local workflow regressions with:

```sh
pytest tests/test_workflows.py tests/test_workflow_voice.py tests/test_workflow_provider.py
```

These tests use mock providers and a local WebSocket emulator. They do not call
OpenAI or a production PBX. Database and PBX integration tests remain in the
[NethVoice module repository](https://github.com/nethesis/ns8-nethvoice/tree/agent/satellite/tests).

## Agent deployment and review fixes

The NS8 listener uses loopback and host networking. Traefik publishes only
`/agents-api/v1` for authenticated machine clients. Keep `/api/*`, provider event
forwarding, monitoring, configuration and workflow administration private. Tokens
belong in headers. Query-string credentials are not accepted.

`SATELLITE_PBX_DATA_TOKEN` is a separate credential for the local PBX contacts
and history endpoint. NS8 creates and repairs it and rejects forwarded requests.
The transcription supervisor retries startup with a 1–30 second backoff. Its
shutdown hook closes ARI, MQTT and RTP before Uvicorn finishes shutdown.

Payment caller ID is an identification hint. Every caller must provide a matching
resident name and verification code before a payment lookup. New verification
codes must contain at least six characters. Store codes, phone numbers, resident
IDs, customer IDs and unit numbers as text. Native numeric amounts use their
numeric value; the decimal separator applies only to text cells.

Data references can select an integer version or `version: "latest"`. Latest
references, including references in reusable blocks, resolve at run start and
stay fixed for that run. An unchanged successful Sheet refresh updates freshness
without adding a version. Failed refresh attempts do not make old data fresh.
Old unreferenced versions are pruned; explicit graph references and active runs
retain their versions. Each resource is limited to 64 MiB within the 256 MiB store.

The container runs as UID/GID 1001. NS8 assigns its private state volume to that
user on startup. Runtime dependencies and the Python base version are pinned.
Build and test the corrected Satellite image before selecting its digest in
NethVoice's `SATELLITE_RUNTIME_IMAGE` build setting. API contracts and Satellite
unit/acceptance tests are maintained in this repository under `docs/agents/` and
`tests/`. NethVoice keeps its PBX integration tests and acceptance runners.
