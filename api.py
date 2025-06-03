from fastapi import FastAPI, HTTPException, UploadFile, File
import httpx
import os

app = FastAPI()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")  # Ensure this environment variable is set

@app.post('/api/get_transcription')
async def get_transcription(file: UploadFile = File(...)):
    if file.content_type not in ("audio/wav", "audio/x-wav"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only WAV files are supported.")

    audio_bytes = await file.read()

    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": file.content_type
    }

    params = {
        "model": "nova-3",
        "detect_language": "true"
    }

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
    except (KeyError, IndexError):
        raise HTTPException(status_code=500, detail="Failed to parse transcription response.")

    return {"transcript": transcript}