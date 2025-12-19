from fastapi import FastAPI, HTTPException, UploadFile, File, Request
try:
    from fastapi.concurrency import run_in_threadpool
except Exception:  # pragma: no cover
    from starlette.concurrency import run_in_threadpool
import httpx
import os
import logging

import ai
import db

app = FastAPI()
logger = logging.getLogger("api")

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")  # Ensure this environment variable is set

def _get_deepgram_timeout_seconds() -> float:
    raw = os.getenv("DEEPGRAM_TIMEOUT_SECONDS", "300").strip()
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid DEEPGRAM_TIMEOUT_SECONDS=%r; defaulting to 300", raw)
        return 300.0

@app.post('/api/get_transcription')
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
    try:
        db.validate_uniqueid(uniqueid)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Valid Deepgram REST API parameters for /v1/listen endpoint
    deepgram_params = {
        "callback" : "",
        "callback_method" : "",
        "custom_topic" : "",
        "custom_topic_mode" : "",
        "custom_intent" : "",
        "custom_intent_mode" : "",
        "detect_entities" : "",
        "detect_language" : "true",
        "diarize" : "",
        "dictation" : "",
        "encoding" : "",
        "extra" : "",
        "filler_words" : "",
        "intents" : "",
        "keyterm" : "",
        "keywords" : "",
        "language" : "",
        "measurements" : "",
        "mip_opt_out" : "", # Opts out requests from the Deepgram Model Improvement Program
        "model" : "nova-3",
        "multichannel" : "",
        "numerals" : "true",
        "paragraphs" : "true",
        "profanity_filter" : "",
        "punctuate" : "true",
        "redact" : "",
        "replace" : "",
        "search" : "",
        "sentiment" : "true",
        "smart_format" : "",
        "summarize" : "",
        "tag" : "",
        "topics" : "",
        "utterances" : "",
        "utt_split" : "",
        "version" : "",
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
        try:
            status = e.response.status_code if e.response is not None else "unknown"
            body_preview = e.response.text[:500] if e.response is not None and hasattr(e.response, "text") and e.response.text else ""
            logger.error("Deepgram API error: status=%s body_preview=%s", status, body_preview)
        except Exception:
            logger.error("Deepgram API error (logging failed)")
        raise HTTPException(status_code=e.response.status_code, detail=f"Deepgram API error: {e.response.text}")
    except httpx.TimeoutException:
        logger.warning("Deepgram request timed out (uniqueid=%s)", uniqueid)
        raise HTTPException(status_code=504, detail="Deepgram request timed out")
    except httpx.RequestError as e:
        logger.error("Deepgram request failed (uniqueid=%s): %s", uniqueid, str(e))
        raise HTTPException(status_code=502, detail="Failed to reach Deepgram")
    except Exception as e:
        logger.exception("Unexpected error while calling Deepgram")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

    result = response.json()
    logger.debug(f"Deepgram result JSON: {result}")
    try:
        transcript = result["results"]["channels"][0]["alternatives"][0]["transcript"]
        if "detected_language" in result["results"]["channels"][0]:
            detected_language = result["results"]["channels"][0]["detected_language"]
        else:
            detected_language = None
        # get speaker diarization
        if "paragraphs" in result["results"] and 'transcript' in result["results"]["paragraphs"]:
            diarized_transcript = result["results"]["paragraphs"]["transcript"]
        else:
            diarized_transcript = None
    except (KeyError, IndexError):
        raise HTTPException(status_code=500, detail="Failed to parse transcription response.")

    # Persist raw transcript always (when Postgres config is present)
    transcript_id = None
    if db.is_configured():
        try:
            transcript_id = await run_in_threadpool(
                db.upsert_transcript_raw,
                uniqueid=uniqueid,
                raw_transcription=transcript,
                detected_language=detected_language,
                diarized_transcript=diarized_transcript,
            )
        except Exception:
            logger.exception("Failed to persist raw transcript to Postgres")
            raise HTTPException(status_code=500, detail="Failed to persist transcription")
    else:
        logger.warning("PGVECTOR_* env vars not set; skipping Postgres persistence")

    # Optional AI enrichment + embeddings
    if os.getenv("OPENAI_API_KEY") and transcript_id is not None and transcript.strip():
        try:
            cleaned = await run_in_threadpool(ai.get_clean, transcript)
            summary = await run_in_threadpool(ai.get_summary, cleaned)
            await run_in_threadpool(
                db.update_transcript_ai_fields,
                transcript_id=transcript_id,
                cleaned_transcription=cleaned,
                summary=summary,
            )
            await run_in_threadpool(
                db.replace_transcript_embeddings,
                transcript_id=transcript_id,
                uniqueid=uniqueid,
                raw_transcription=transcript,
            )
        except Exception:
            logger.exception("Failed to generate/store AI fields or embeddings")

    return {"transcript": transcript, "detected_language": detected_language, "diarized_transcript": diarized_transcript}