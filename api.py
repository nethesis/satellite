from fastapi import APIRouter, Depends, FastAPI, HTTPException, UploadFile, File, Request
from fastapi.concurrency import run_in_threadpool
import json
import httpx
import os
import logging
import subprocess
import sys

import db
from transcription import get_provider

app = FastAPI()
logger = logging.getLogger("api")


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
    provider_name = (input_params.get("provider") or "").strip().lower() or None
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
        # Create/mark a DB row immediately so we can track state even if transcription fails.
        try:
            transcript_id = await run_in_threadpool(
                db.upsert_transcript_progress,
                uniqueid=uniqueid,
            )
        except Exception:
            logger.exception("Failed to initialize transcript row for state tracking")
            raise HTTPException(status_code=500, detail="Failed to initialize transcript persistence")

    # Get transcription provider
    try:
        provider = get_provider(provider_name)
    except ValueError as e:
        logger.error("Failed to get transcription provider: %s", str(e))
        if transcript_id is not None:
            try:
                await run_in_threadpool(db.set_transcript_state, transcript_id=transcript_id, state="failed")
            except Exception:
                logger.exception("Failed to update transcript state=failed")
        raise HTTPException(status_code=400, detail=str(e))

    # Call transcription provider
    try:
        result = await provider.transcribe(
            audio_bytes=audio_bytes,
            content_type=file.content_type,
            params=input_params,
        )
        raw_transcription = result.raw_transcription
        detected_language = result.detected_language
    except httpx.HTTPStatusError as e:
        if transcript_id is not None:
            try:
                await run_in_threadpool(db.set_transcript_state, transcript_id=transcript_id, state="failed")
            except Exception:
                logger.exception("Failed to update transcript state=failed")
        try:
            status = e.response.status_code if e.response is not None else "unknown"
            body_preview = e.response.text[:500] if e.response is not None and hasattr(e.response, "text") and e.response.text else ""
            logger.error("Transcription API error: status=%s body_preview=%s", status, body_preview)
        except Exception:
            logger.error("Transcription API error (logging failed)")
        raise HTTPException(status_code=e.response.status_code, detail=f"Transcription API error: {e.response.text}")
    except httpx.TimeoutException:
        logger.warning("Transcription request timed out (uniqueid=%s)", uniqueid)
        if transcript_id is not None:
            try:
                await run_in_threadpool(db.set_transcript_state, transcript_id=transcript_id, state="failed")
            except Exception:
                logger.exception("Failed to update transcript state=failed")
        raise HTTPException(status_code=504, detail="Transcription request timed out")
    except httpx.RequestError as e:
        logger.error("Transcription request failed (uniqueid=%s): %s", uniqueid, str(e))
        if transcript_id is not None:
            try:
                await run_in_threadpool(db.set_transcript_state, transcript_id=transcript_id, state="failed")
            except Exception:
                logger.exception("Failed to update transcript state=failed")
        raise HTTPException(status_code=502, detail="Failed to reach transcription service")
    except ValueError as e:
        logger.error("Failed to parse transcription response: %s", str(e))
        if transcript_id is not None:
            try:
                await run_in_threadpool(db.set_transcript_state, transcript_id=transcript_id, state="failed")
            except Exception:
                logger.exception("Failed to update transcript state=failed")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error while calling transcription service")
        if transcript_id is not None:
            try:
                await run_in_threadpool(db.set_transcript_state, transcript_id=transcript_id, state="failed")
            except Exception:
                logger.exception("Failed to update transcript state=failed")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

    # Apply channel name replacements (provider-agnostic post-processing)
    if channel0_name:
        raw_transcription = raw_transcription.replace("Channel 0:", f"{channel0_name}:")
        raw_transcription = raw_transcription.replace("Speaker 0:", f"{channel0_name}:")
    if channel1_name:
        raw_transcription = raw_transcription.replace("Channel 1:", f"{channel1_name}:")
        raw_transcription = raw_transcription.replace("Speaker 1:", f"{channel1_name}:")

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