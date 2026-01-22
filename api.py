from fastapi import APIRouter, Depends, FastAPI, HTTPException, UploadFile, File, Request
from fastapi.concurrency import run_in_threadpool
import json
import httpx
import os
import logging
import subprocess
import sys

import db

app = FastAPI()
logger = logging.getLogger("api")

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")  # Ensure this environment variable is set


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