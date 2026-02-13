# Copilot Instructions — Satellite

## Build & Test

```bash
# Install
pip install -r requirements.txt
pip install -r requirements-dev.txt   # test deps (pytest, httpx, etc.)

# Run all tests (with coverage)
pytest

# Run a single test file / single test
pytest tests/test_api.py
pytest tests/test_api.py::test_get_transcription_success -k "test_get_transcription_success"

# Run the app
python main.py
```

Python 3.12+. No linter configured in CI — only `pytest` runs in the build workflow.
Container image uses `Containerfile` (multi-stage, python:slim base).

## Architecture

Satellite bridges Asterisk PBX ↔ transcription providers (Deepgram or VoxTral), publishing results over MQTT.

### Runtime components (all in one process)

| Module | Role |
|---|---|
| `main.py` | Entrypoint — starts the asyncio event loop for the real-time pipeline and a background thread running the FastAPI/Uvicorn HTTP server |
| `asterisk_bridge.py` | ARI WebSocket client — listens for Stasis events, creates snoop channels + external media, manages per-call lifecycle |
| `rtp_server.py` | UDP server — receives RTP audio, strips headers, routes packets to per-channel async queues by source port |
| `deepgram_connector.py` | Streams audio to Deepgram via WebSocket — interleaves two RTP channels into stereo for multichannel transcription; aggregates final transcript on hangup (real-time path only, Deepgram-only for now) |
| `mqtt_client.py` | Publishes interim/final transcription JSON to MQTT topics (`{prefix}/transcription`, `{prefix}/final`) |
| `transcription/` | **Provider abstraction** — `base.py` defines interface; `deepgram.py` and `voxtral.py` implement REST API clients; `__init__.py` factory selects provider via env var or per-request override |
| `api.py` | FastAPI app — `POST /api/get_transcription` accepts WAV uploads, calls transcription provider REST API, optionally persists to Postgres |
| `call_processor.py` | **Runs as a subprocess** (invoked from api.py via `subprocess.run`) — reads JSON from stdin, calls AI enrichment, writes results to DB |
| `ai.py` | LangChain + OpenAI — cleans transcript, generates summary + sentiment score (0-10) |
| `db.py` | PostgreSQL + pgvector — schema auto-init with threading lock; stores transcripts, state machine (`progress` → `summarizing` → `done` / `failed`), and text-embedding-3-small chunks |

### Key data flows

1. **Real-time path:** Asterisk → ARI WebSocket → snoop channel → RTP → `rtp_server` → `deepgram_connector` (stereo WebSocket stream) → Deepgram → `mqtt_client` (Deepgram-only for now)
2. **REST/batch path:** WAV upload → `api.py` → `transcription/<provider>` REST API (Deepgram or VoxTral) → (optionally) `db.py` persist → (optionally) `call_processor.py` subprocess → `ai.py` → `db.py` update

### Non-obvious details

- Two RTP streams per call (one per direction) are interleaved into a single stereo buffer for Deepgram's multichannel mode (real-time path only).
- `asterisk_bridge` detects if Asterisk swapped the RTP source ports and adjusts speaker labels accordingly.
- `call_processor` is deliberately a **subprocess** (not async task) — isolates OpenAI calls with independent timeout/logging, avoids blocking the event loop.
- DB schema initialization is guarded by a **threading lock** (not asyncio lock) because `psycopg` sync connections are used alongside the async FastAPI server.
- **Multi-provider support:** REST/batch path supports Deepgram and VoxTral. Select provider via `TRANSCRIPTION_PROVIDER` env var (default: `deepgram`) or per-request `provider=` parameter. Real-time path remains Deepgram-only.

## Conventions

- **Config:** Exclusively via environment variables (loaded from `.env` by `python-dotenv`). No config files or CLI args.
- **Logging:** One logger per module (`logging.getLogger(__name__)`), level controlled by `LOG_LEVEL` env var.
- **Async:** `asyncio` throughout the real-time pipeline; `asyncio.Lock` for connector close logic, `asyncio.Queue` for RTP buffer routing. Reconnection uses exponential backoff.
- **Testing:** `pytest-asyncio` with `asyncio_mode = auto`. Tests monkeypatch env vars and mock external services (Deepgram, MQTT, psycopg). A conftest auto-fixture resets `db._schema_initialized` between tests.
- **Auth:** Optional static bearer token (`API_TOKEN` env var) for `/api/*` endpoints. Accepts `Authorization: Bearer <token>` or `X-API-Token: <token>`.
- **Validation:** `uniqueid` must match `\d+\.\d+` (Asterisk format).
