from fastapi import FastAPI, HTTPException, UploadFile, File, Request
import httpx
import os

app = FastAPI()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")  # Ensure this environment variable is set

@app.post('/api/get_transcription')
async def get_transcription(
    request: Request,
    file: UploadFile = File(...)
):
    if file.content_type not in ("audio/wav", "audio/x-wav"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only WAV files are supported.")

    audio_bytes = await file.read()

    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": file.content_type
    }

    # Get all query parameters from the request
    input_params = dict(request.query_params)

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

    params = {}
    for k, v in deepgram_params.items():
        if k in input_params and input_params[k].strip():
            params[k] = input_params[k]
        elif v.strip():
            params[k] = v

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.deepgram.com/v1/listen",
                headers=headers,
                params=params,
                content=audio_bytes
            )
            response.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Deepgram API error: {e.response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

    result = response.json()
    try:
        transcript = result["results"]["channels"][0]["alternatives"][0]["transcript"]
        if "detected_language" in result["results"]["channels"][0]:
            detected_language = result["results"]["channels"][0]["detected_language"]
        else:
            detected_language = None
        # get speaker diarization
        if "paragraphs" in result["results"] and "paragraphs" in result["results"]["paragraphs"] and 'transcript' in result["results"]["paragraphs"]:
            diarized_transcript = result["results"]["paragraphs"]["transcript"]
        else:
            diarized_transcript = None
    except (KeyError, IndexError):
        raise HTTPException(status_code=500, detail="Failed to parse transcription response.")

    return {"transcript": transcript, "detected_language": detected_language, "diarized_transcript": diarized_transcript}