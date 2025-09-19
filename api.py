from fastapi import FastAPI, HTTPException, UploadFile, File, Request
import httpx
import os
import logging

app = FastAPI()
logger = logging.getLogger("api")

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")  # Ensure this environment variable is set

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

    # Debug: log outgoing request details (with masked Authorization)
    try:
        auth_value = headers.get("Authorization")
        if isinstance(auth_value, str) and auth_value.startswith("Token ") and len(auth_value) > 10:
            masked_auth = f"Token ****{auth_value[-4:]}"
        else:
            masked_auth = "***"
        safe_headers = {**headers, "Authorization": masked_auth}
        logger.debug(
            "Deepgram request: url=%s headers=%s params=%s content_length=%s content_type=%s",
            "https://api.deepgram.com/v1/listen",
            {k: safe_headers[k] for k in safe_headers},
            params,
            len(audio_bytes),
            file.content_type,
        )
    except Exception:
        # Never block the request on logging
        logger.debug("Failed to prepare debug log for Deepgram request")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.deepgram.com/v1/listen",
                headers=headers,
                params=params,
                content=audio_bytes
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
    except Exception as e:
        logger.exception("Unexpected error while calling Deepgram")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

    result = response.json()
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

    return {"transcript": transcript, "detected_language": detected_language, "diarized_transcript": diarized_transcript}