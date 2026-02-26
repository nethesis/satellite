from fastapi import APIRouter, Depends, FastAPI, HTTPException, UploadFile, File, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
import re
import uuid
import json
import httpx
import os
import logging
import subprocess
import sys
from langchain_text_splitters import RecursiveCharacterTextSplitter


import db

app = FastAPI()
logger = logging.getLogger("api")

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")  # Ensure this environment variable is set

DEEPGRAM_TTS_MODELS = [
    "aura-2-agathe-fr",
    "aura-2-agustina-es",
    "aura-2-alvaro-es",
    "aura-2-ama-ja",
    "aura-2-amalthea-en",
    "aura-2-andromeda-en",
    "aura-2-antonia-es",
    "aura-2-apollo-en",
    "aura-2-aquila-es",
    "aura-2-arcas-en",
    "aura-2-aries-en",
    "aura-2-asteria-en",
    "aura-2-athena-en",
    "aura-2-atlas-en",
    "aura-2-aurelia-de",
    "aura-2-aurora-en",
    "aura-2-beatrix-nl",
    "aura-2-callista-en",
    "aura-2-carina-es",
    "aura-2-celeste-es",
    "aura-2-cesare-it",
    "aura-2-cinzia-it",
    "aura-2-cora-en",
    "aura-2-cordelia-en",
    "aura-2-cornelia-nl",
    "aura-2-daphne-nl",
    "aura-2-delia-en",
    "aura-2-demetra-it",
    "aura-2-diana-es",
    "aura-2-dionisio-it",
    "aura-2-draco-en",
    "aura-2-ebisu-ja",
    "aura-2-elara-de",
    "aura-2-electra-en",
    "aura-2-elio-it",
    "aura-2-estrella-es",
    "aura-2-fabian-de",
    "aura-2-flavio-it",
    "aura-2-fujin-ja",
    "aura-2-gloria-es",
    "aura-2-harmonia-en",
    "aura-2-hector-fr",
    "aura-2-helena-en",
    "aura-2-hera-en",
    "aura-2-hermes-en",
    "aura-2-hestia-nl",
    "aura-2-hyperion-en",
    "aura-2-iris-en",
    "aura-2-izanami-ja",
    "aura-2-janus-en",
    "aura-2-javier-es",
    "aura-2-julius-de",
    "aura-2-juno-en",
    "aura-2-jupiter-en",
    "aura-2-kara-de",
    "aura-2-lara-de",
    "aura-2-lars-nl",
    "aura-2-leda-nl",
    "aura-2-livia-it",
    "aura-2-luciano-es",
    "aura-2-luna-en",
    "aura-2-maia-it",
    "aura-2-mars-en",
    "aura-2-melia-it",
    "aura-2-minerva-en",
    "aura-2-neptune-en",
    "aura-2-nestor-es",
    "aura-2-odysseus-en",
    "aura-2-olivia-es",
    "aura-2-ophelia-en",
    "aura-2-orion-en",
    "aura-2-orpheus-en",
    "aura-2-pandora-en",
    "aura-2-perseo-it",
    "aura-2-phoebe-en",
    "aura-2-pluto-en",
    "aura-2-rhea-nl",
    "aura-2-roman-nl",
    "aura-2-sander-nl",
    "aura-2-saturn-en",
    "aura-2-selena-es",
    "aura-2-selene-en",
    "aura-2-silvia-es",
    "aura-2-sirio-es",
    "aura-2-thalia-en",
    "aura-2-theia-en",
    "aura-2-uzume-ja",
    "aura-2-valerio-es",
    "aura-2-vesta-en",
    "aura-2-viktoria-de",
    "aura-2-zeus-en",
    "aura-angus-en",
    "aura-arcas-en",
    "aura-asteria-en",
    "aura-athena-en",
    "aura-helios-en",
    "aura-hera-en",
    "aura-luna-en",
    "aura-orion-en",
    "aura-orpheus-en",
    "aura-perseus-en",
    "aura-stella-en",
    "aura-zeus-en",
]

def _iter_bytes(data: bytes, *, chunk_size: int):
    for i in range(0, len(data), chunk_size):
        yield data[i : i + chunk_size]


def _require_api_token_if_configured(request: Request) -> None:
    configured_token = (os.getenv("API_TOKEN") or "").strip()
    if not configured_token:
        return

    provided_token = None

    auth = (request.headers.get("authorization") or "").strip()
    if auth.lower().startswith("bearer "):
        provided_token = auth[7:].strip()

    if not provided_token:
        provided_token = (request.headers.get("x-api-token") or "").strip() or None

    if not provided_token or provided_token != configured_token:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Bearer"},
        )


api_router = APIRouter(
    prefix="/api",
    dependencies=[Depends(_require_api_token_if_configured)],
)

def _run_call_processor(
    *,
    transcript_id: int,
    raw_transcription: str,
    summary: bool = False,
) -> None:
    payload = {"transcript_id": transcript_id, "raw_transcription": raw_transcription, "summary": summary}
    proc = subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(__file__), "call_processor.py")],
        input=json.dumps(payload).encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=float(os.getenv("CALL_PROCESSOR_TIMEOUT_SECONDS", "600")),
    )

    # The subprocess logs (including ai pipeline logs) go to stderr by default.
    # Since we're capturing stderr, re-log it here so it shows up in the API logs
    # even when the subprocess succeeds.
    stderr_text = (proc.stderr or b"").decode("utf-8", errors="replace")
    if stderr_text.strip():
        lines = stderr_text.splitlines()
        max_lines = int(os.getenv("CALL_PROCESSOR_LOG_MAX_LINES", "200"))
        if len(lines) > max_lines:
            lines = lines[:max_lines] + [f"... (truncated; {len(stderr_text)} bytes total)"]
        for line in lines:
            if line.strip():
                logger.info("call_processor[%s]: %s", transcript_id, line)

    if proc.returncode != 0:
        stderr_preview = (proc.stderr or b"")[:2000].decode("utf-8", errors="replace")
        stdout_preview = (proc.stdout or b"")[:2000].decode("utf-8", errors="replace")
        raise RuntimeError(f"call_processor failed rc={proc.returncode} stdout={stdout_preview!r} stderr={stderr_preview!r}")

def _get_deepgram_timeout_seconds() -> float:
    raw = os.getenv("DEEPGRAM_TIMEOUT_SECONDS", "300").strip()
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid DEEPGRAM_TIMEOUT_SECONDS=%r; defaulting to 300", raw)
        return 300.0


def get_models(language: str | None = None) -> list[str]:
    models = DEEPGRAM_TTS_MODELS
    normalized_language = (language or "").strip().lower()
    if not normalized_language:
        return models
    language_suffix = f"-{normalized_language}"
    return [model for model in models if model.lower().endswith(language_suffix)]


@api_router.get("/get_models")
async def get_models_endpoint(language: str | None = None):
    return {"models": get_models(language)}


@api_router.post("/get_speech")
async def get_speech(request: Request):
    # Collect parameters from query string and multipart/x-www-form-urlencoded form fields
    try:
        form = await request.form()
    except Exception:
        form = {}

    form_params = {}
    if hasattr(form, "items"):
        for k, v in form.items():
            form_params[k] = v if isinstance(v, str) else str(v)

    input_params = {**dict(request.query_params), **form_params}
    logger.debug("Params: %s", input_params)

    text = (input_params.get("text") or input_params.get("input") or "").strip()
    language = (input_params.get("language") or "").strip().lower()
    if not text:
        raise HTTPException(status_code=400, detail="Missing required field: text")

    requested_encoding = (input_params.get("encoding") or "").strip().lower()
    requested_container = (input_params.get("container") or "").strip().lower()
    if requested_encoding and requested_encoding != "mp3":
        raise HTTPException(status_code=400, detail="Only MP3 output is supported (encoding=mp3)")
    if requested_container and requested_container != "mp3":
        raise HTTPException(status_code=400, detail="Only MP3 output is supported")

    # use lanchain text splitter to split text into smaller chunks
    # Deepgram TTS can handle 2000 characters per request https://developers.deepgram.com/docs/text-to-speech#input-text-limit
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=0,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )
    chunks = [c.strip() for c in splitter.split_text(text or "")]
    if not chunks:
        raise HTTPException(status_code=400, detail="Text is empty")

    # https://developers.deepgram.com/reference/text-to-speech/speak-request
    deepgram_params = {
        "callback": "",
        "callback_method": "",
        "mip_opt_out": "false",
        "tag": "",
        "bit_rate": "",
        "model": "",
        "sample_rate": ""
    }

    params: dict[str, str] = {}
    for k, v in deepgram_params.items():
        if k in input_params and str(input_params[k]).strip():
            params[k] = str(input_params[k]).strip()
        elif str(v).strip():
            params[k] = str(v).strip()

    if not params.get("model") and language:
        models = get_models(language)
        if not models:
            raise HTTPException(status_code=400, detail=f"No TTS model available for language: {language}")
        params["model"] = models[0]
    params["encoding"] = "mp3"

    logger.debug("Deepgram TTS params: %s", params)

    url = "https://api.deepgram.com/v1/speak"
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }

    audio_parts: list[bytes] = []
    try:
        deepgram_timeout_seconds = _get_deepgram_timeout_seconds()
        timeout = httpx.Timeout(
            connect=10.0,
            read=deepgram_timeout_seconds,
            write=deepgram_timeout_seconds,
            pool=10.0,
        )
        async with httpx.AsyncClient(timeout=timeout) as client:
            for idx, chunk in enumerate(chunks, start=1):
                response = await client.post(
                    url,
                    headers=headers,
                    params=params,
                    json={"text": chunk},
                )
                try:
                    logger.debug(
                        "Deepgram TTS response: chunk=%s/%s status=%s content_type=%s bytes=%s",
                        idx,
                        len(chunks),
                        response.status_code,
                        response.headers.get("Content-Type"),
                        len(response.content or b""),
                    )
                except Exception:
                    logger.debug("Failed to log Deepgram TTS response meta")
                response.raise_for_status()
                audio_parts.append(response.content)
    except httpx.HTTPStatusError as e:
        try:
            status = e.response.status_code if e.response is not None else 502
            body_preview = (
                e.response.text[:500]
                if e.response is not None and hasattr(e.response, "text") and e.response.text
                else ""
            )
            logger.error("Deepgram TTS API error: status=%s body_preview=%s", status, body_preview)
        except Exception:
            logger.error("Deepgram TTS API error (logging failed)")
        raise HTTPException(
            status_code=e.response.status_code if e.response is not None else 502,
            detail=f"Deepgram API error: {e.response.text if e.response is not None else ''}",
        )
    except httpx.TimeoutException:
        logger.warning("Deepgram TTS request timed out")
        raise HTTPException(status_code=504, detail="Deepgram request timed out")
    except httpx.RequestError as e:
        logger.error("Deepgram TTS request failed: %s", str(e))
        raise HTTPException(status_code=502, detail="Failed to reach Deepgram")
    except Exception as e:
        logger.exception("Unexpected error while calling Deepgram TTS")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

    audio_bytes = b"".join(audio_parts)
    if not audio_bytes:
        raise HTTPException(status_code=500, detail="Deepgram returned empty audio")

    filename = f"speech-{uuid.uuid4().hex}.mp3"
    headers_out = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
    }
    return StreamingResponse(
        _iter_bytes(audio_bytes, chunk_size=65536),
        media_type="audio/mpeg",
        headers=headers_out,
    )

@api_router.post('/get_transcription')
async def get_transcription(
    request: Request,
    file: UploadFile = File(...)
):
    if file.content_type not in ("audio/wav", "audio/x-wav"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only WAV files are supported.")

    audio_bytes = await file.read()

    # Collect parameters from query string and multipart form fields (excluding the file)
    try:
        form = await request.form()
    except Exception:
        form = {}

    form_params = {}
    if hasattr(form, "items"):
        for k, v in form.items():
            if k == "file":
                continue
            form_params[k] = v if isinstance(v, str) else str(v)

    input_params = {**dict(request.query_params), **form_params}

    logger.debug(f"Params: {input_params}")

    uniqueid = (input_params.get("uniqueid") or "").strip()
    channel0_name = (input_params.get("channel0_name") or "").strip()
    channel1_name = (input_params.get("channel1_name") or "").strip()
    # Persist only when explicitly requested.
    persist = (input_params.get("persist") or "false").lower() in ("1", "true", "yes")
    summary = (input_params.get("summary") or "false").lower() in ("1", "true", "yes")

    # uniqueid is only required when persistence is enabled.
    if persist:
        try:
            db.validate_uniqueid(uniqueid)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    transcript_id = None
    if db.is_configured() and persist:
        # Create/mark a DB row immediately so we can track state even if Deepgram fails.
        try:
            transcript_id = await run_in_threadpool(
                db.upsert_transcript_progress,
                uniqueid=uniqueid,
            )
        except Exception:
            logger.exception("Failed to initialize transcript row for state tracking")
            raise HTTPException(status_code=500, detail="Failed to initialize transcript persistence")

    # Valid Deepgram REST API parameters for /v1/listen endpoint
    deepgram_params = {
        "callback": "",
        "callback_method": "",
        "custom_topic": "",
        "custom_topic_mode": "",
        "custom_intent": "",
        "custom_intent_mode": "",
        "detect_entities": "",
        "detect_language": "true",
        "diarize": "",
        "dictation": "",
        "encoding": "",
        "extra": "",
        "filler_words": "",
        "intents": "",
        "keyterm": "",
        "keywords": "",
        "language": "",
        "measurements": "",
        "mip_opt_out": "", # Opts out requests from the Deepgram Model Improvement Program
        "model": "nova-3",
        "multichannel": "",
        "numerals": "true",
        "paragraphs": "true",
        "profanity_filter": "",
        "punctuate": "true",
        "redact": "",
        "replace": "",
        "search": "",
        "sentiment": "false",
        "smart_format": "true",
        "summarize": "",
        "tag": "",
        "topics": "",
        "utterances": "",
        "utt_split": "",
        "version": "",
    }

    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": file.content_type
    }

    params = {}
    for k, v in deepgram_params.items():
        if k in input_params and input_params[k].strip():
            params[k] = input_params[k]
        elif v.strip():
            params[k] = v

    try:
        deepgram_timeout_seconds = _get_deepgram_timeout_seconds()
        timeout = httpx.Timeout(
            connect=10.0,
            read=deepgram_timeout_seconds,
            write=deepgram_timeout_seconds,
            pool=10.0,
        )
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                "https://api.deepgram.com/v1/listen",
                headers=headers,
                params=params,
                content=audio_bytes,
            )
            # Debug: log response meta and preview
            try:
                logger.debug(
                    "Deepgram response: status=%s content_type=%s body_preview=%s",
                    response.status_code,
                    response.headers.get("Content-Type"),
                    (response.text[:500] if response is not None and hasattr(response, "text") and response.text else ""),
                )
            except Exception:
                logger.debug("Failed to log Deepgram response preview")
            response.raise_for_status()
    except httpx.HTTPStatusError as e:
        if transcript_id is not None:
            try:
                await run_in_threadpool(db.set_transcript_state, transcript_id=transcript_id, state="failed")
            except Exception:
                logger.exception("Failed to update transcript state=failed")
        try:
            status = e.response.status_code if e.response is not None else "unknown"
            body_preview = e.response.text[:500] if e.response is not None and hasattr(e.response, "text") and e.response.text else ""
            logger.error("Deepgram API error: status=%s body_preview=%s", status, body_preview)
        except Exception:
            logger.error("Deepgram API error (logging failed)")
        raise HTTPException(status_code=e.response.status_code, detail=f"Deepgram API error: {e.response.text}")
    except httpx.TimeoutException:
        logger.warning("Deepgram request timed out (uniqueid=%s)", uniqueid)
        if transcript_id is not None:
            try:
                await run_in_threadpool(db.set_transcript_state, transcript_id=transcript_id, state="failed")
            except Exception:
                logger.exception("Failed to update transcript state=failed")
        raise HTTPException(status_code=504, detail="Deepgram request timed out")
    except httpx.RequestError as e:
        logger.error("Deepgram request failed (uniqueid=%s): %s", uniqueid, str(e))
        if transcript_id is not None:
            try:
                await run_in_threadpool(db.set_transcript_state, transcript_id=transcript_id, state="failed")
            except Exception:
                logger.exception("Failed to update transcript state=failed")
        raise HTTPException(status_code=502, detail="Failed to reach Deepgram")
    except Exception as e:
        logger.exception("Unexpected error while calling Deepgram")
        if transcript_id is not None:
            try:
                await run_in_threadpool(db.set_transcript_state, transcript_id=transcript_id, state="failed")
            except Exception:
                logger.exception("Failed to update transcript state=failed")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

    result = response.json()
    detected_language = None  # always define; mocks may omit this field
    try:
        if "paragraphs" in result["results"] and "transcript" in result["results"]["paragraphs"]:
            raw_transcription = result["results"]["paragraphs"]["transcript"].strip()
        elif (
            "channels" in result["results"]
            and result["results"]["channels"]
            and "alternatives" in result["results"]["channels"][0]
            and result["results"]["channels"][0]["alternatives"]
            and "paragraphs" in result["results"]["channels"][0]["alternatives"][0]
            and "transcript" in result["results"]["channels"][0]["alternatives"][0]["paragraphs"]
        ):
            raw_transcription = (
                result["results"]["channels"][0]["alternatives"][0]["paragraphs"]["transcript"].strip()
            )
        else:
            logger.debug("failed to get paragraphs transcript")
            logger.debug(result)
            raise KeyError("paragraphs transcript not found")
        if "channels" in result["results"] and "detected_language" in result["results"]["channels"][0]:
            detected_language = result["results"]["channels"][0]["detected_language"]
        else:
            logger.debug("failed to get detected_language")
            logger.debug(result)
        if channel0_name:
            raw_transcription = raw_transcription.replace("Channel 0:", f"{channel0_name}:")
        if channel1_name:
            raw_transcription = raw_transcription.replace("Channel 1:", f"{channel1_name}:")
    except (KeyError, IndexError):
        logger.error("Failed to parse Deepgram transcription response: %s", response.text)
        if transcript_id is not None:
            try:
                await run_in_threadpool(db.set_transcript_state, transcript_id=transcript_id, state="failed")
            except Exception:
                logger.exception("Failed to update transcript state=failed")
        raise HTTPException(status_code=500, detail="Failed to parse transcription response.")

    # Persist raw transcript when Postgres config is present (default) unless disabled per request.
    if transcript_id is not None:
        try:
            transcript_id = await run_in_threadpool(
                db.upsert_transcript_raw,
                uniqueid=uniqueid,
                raw_transcription=raw_transcription,
            )
        except ValueError as e:
            logger.exception("Invalid uniqueid for Postgres persistence")
            try:
                await run_in_threadpool(db.set_transcript_state, transcript_id=transcript_id, state="failed")
            except Exception:
                logger.exception("Failed to update transcript state=failed")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception:
            logger.exception("Failed to persist raw transcript to Postgres")
            try:
                await run_in_threadpool(db.set_transcript_state, transcript_id=transcript_id, state="failed")
            except Exception:
                logger.exception("Failed to update transcript state=failed")
            raise HTTPException(status_code=500, detail="Failed to persist transcription")
    else:
        if not db.is_configured():
            logger.warning("PGVECTOR_* env vars not set; skipping Postgres persistence")
        else:
            logger.debug("Postgres persistence disabled by request")

    # Optional AI enrichment (clean/summary/sentiment) via per-request subprocess
    did_enrichment = False
    if os.getenv("OPENAI_API_KEY") and transcript_id is not None and raw_transcription:
        try:
            did_enrichment = True
            await run_in_threadpool(db.set_transcript_state, transcript_id=transcript_id, state="summarizing")
            await run_in_threadpool(
                _run_call_processor,
                transcript_id=transcript_id,
                raw_transcription=raw_transcription,
                summary=summary,
            )
            await run_in_threadpool(db.set_transcript_state, transcript_id=transcript_id, state="done")
        except Exception:
            logger.exception("Failed to process call transcript")
            try:
                await run_in_threadpool(db.set_transcript_state, transcript_id=transcript_id, state="failed")
            except Exception:
                logger.exception("Failed to update transcript state=failed")

    # If we persisted but didn't run enrichment, the pipeline is complete after raw transcript is stored.
    if transcript_id is not None and not did_enrichment:
        try:
            await run_in_threadpool(db.set_transcript_state, transcript_id=transcript_id, state="done")
        except Exception:
            logger.exception("Failed to update transcript state=done")

    return {"transcript": raw_transcription, "detected_language": detected_language}


app.include_router(api_router)
