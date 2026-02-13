"""Deepgram transcription provider."""

import httpx
import logging
import os
from .base import TranscriptionProvider, TranscriptionResult

logger = logging.getLogger(__name__)


class DeepgramProvider(TranscriptionProvider):
    """Deepgram transcription provider using REST API."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY", "")
        if not self.api_key:
            raise ValueError("DEEPGRAM_API_KEY is required for Deepgram provider")

    async def transcribe(
        self,
        audio_bytes: bytes,
        content_type: str,
        params: dict[str, str],
    ) -> TranscriptionResult:
        """Transcribe audio using Deepgram REST API."""
        
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
            "mip_opt_out": "",
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
            "Authorization": f"Token {self.api_key}",
            "Content-Type": content_type
        }

        # Build request params from defaults + user overrides
        request_params = {}
        for k, v in deepgram_params.items():
            if k in params and params[k].strip():
                request_params[k] = params[k]
            elif v.strip():
                request_params[k] = v

        # Get timeout from env var
        timeout_seconds = self._get_timeout_seconds()
        timeout = httpx.Timeout(
            connect=10.0,
            read=timeout_seconds,
            write=timeout_seconds,
            pool=10.0,
        )

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                "https://api.deepgram.com/v1/listen",
                headers=headers,
                params=request_params,
                content=audio_bytes,
            )

            # Debug logging
            try:
                logger.debug(
                    "Deepgram response: status=%s content_type=%s body_preview=%s",
                    response.status_code,
                    response.headers.get("Content-Type"),
                    (response.text[:500] if response.text else ""),
                )
            except Exception:
                logger.debug("Failed to log Deepgram response preview")

            response.raise_for_status()

        result = response.json()
        detected_language = None

        # Parse transcription from response
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

        except (KeyError, IndexError) as e:
            logger.error("Failed to parse Deepgram transcription response: %s", response.text)
            raise ValueError(f"Failed to parse transcription response: {e}")

        return TranscriptionResult(
            raw_transcription=raw_transcription,
            detected_language=detected_language
        )

    def _get_timeout_seconds(self) -> float:
        """Get timeout from environment variable."""
        raw = os.getenv("DEEPGRAM_TIMEOUT_SECONDS", "300").strip()
        try:
            return float(raw)
        except ValueError:
            logger.warning("Invalid DEEPGRAM_TIMEOUT_SECONDS=%r; defaulting to 300", raw)
            return 300.0
