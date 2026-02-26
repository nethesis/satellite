"""VoxTral (Mistral) transcription provider."""

import httpx
import logging
import os
from .base import TranscriptionProvider, TranscriptionResult

logger = logging.getLogger(__name__)


class VoxtralProvider(TranscriptionProvider):
    """VoxTral (Mistral) transcription provider using REST API."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY", "")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY is required for VoxTral provider")

    async def transcribe(
        self,
        audio_bytes: bytes,
        content_type: str,
        params: dict[str, str],
    ) -> TranscriptionResult:
        """Transcribe audio using Mistral VoxTral REST API."""

        # Build multipart form data
        files = {
            "file": ("audio.wav", audio_bytes, content_type),
        }

        # VoxTral parameters
        data = {
            "model": params.get("model", "voxtral-mini-latest"),
        }

        # Optional parameters
        if "language" in params and params["language"].strip():
            data["language"] = params["language"]

        # Enable diarization by default (for speaker labels), unless explicitly disabled
        diarize_disabled = "diarize" in params and params["diarize"].strip().lower() in ("false", "0", "no")
        if not diarize_disabled:
            data["diarize"] = True  # Boolean, not string
            # VoxTral requires timestamp_granularities when diarize is enabled
            if "timestamp_granularities" not in params or not params.get("timestamp_granularities", "").strip():
                data["timestamp_granularities"] = ["segment"]

        if "temperature" in params and params["temperature"].strip():
            try:
                data["temperature"] = float(params["temperature"])
            except ValueError:
                pass  # Skip invalid temperature values

        # Context biasing (up to 100 words/phrases)
        if "context_bias" in params and params["context_bias"].strip():
            # Split comma-separated list if provided
            context_items = [item.strip() for item in params["context_bias"].split(",") if item.strip()]
            if context_items:
                # VoxTral expects multiple "context_bias" fields in the form data
                for item in context_items[:100]:  # limit to 100
                    data.setdefault("context_bias", [])
                    if isinstance(data["context_bias"], list):
                        data["context_bias"].append(item)

        # Timestamp granularities (user-provided or set by diarize logic above)
        if "timestamp_granularities" in params and params["timestamp_granularities"].strip():
            granularities = [g.strip() for g in params["timestamp_granularities"].split(",") if g.strip()]
            valid_granularities = [g for g in granularities if g in ("segment", "word")]
            if valid_granularities:
                data["timestamp_granularities"] = valid_granularities

        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }

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
                "https://api.mistral.ai/v1/audio/transcriptions",
                headers=headers,
                files=files,
                data=data,
            )

            # Debug logging
            try:
                logger.debug(
                    "VoxTral response: status=%s content_type=%s body_preview=%s",
                    response.status_code,
                    response.headers.get("Content-Type"),
                    (response.text[:500] if response.text else ""),
                )
            except Exception:
                logger.debug("Failed to log VoxTral response preview")

            response.raise_for_status()

        result = response.json()

        # Parse VoxTral response
        # Response format: { "text": "...", "language": "...", "segments": [...], "model": "..." }
        raw_transcription = result.get("text", "").strip()
        detected_language = result.get("language")

        # If diarization is enabled and we have segments with speaker info,
        # reconstruct a speaker-labeled transcript
        segments = result.get("segments", [])
        if segments and any("speaker_id" in seg or "speaker" in seg for seg in segments):
            raw_transcription = self._format_diarized_transcript(segments)

        if not raw_transcription:
            # Empty transcription is valid for silence/no speech
            logger.debug("VoxTral returned empty transcription (no speech detected)")

        return TranscriptionResult(
            raw_transcription=raw_transcription or "",  # Return empty string instead of raising
            detected_language=detected_language
        )

    def _format_diarized_transcript(self, segments: list[dict]) -> str:
        """Format segments with speaker diarization into a readable transcript."""
        lines = []
        last_speaker = None

        for seg in segments:
            # VoXtral uses "speaker_id" field (e.g., "speaker_1", "speaker_2")
            # Fall back to "speaker" for backward compatibility with test mocks
            speaker = seg.get("speaker_id") or seg.get("speaker")
            text = seg.get("text", "").strip()

            if not text:
                continue

            # Add speaker label when speaker changes
            if speaker is not None and speaker != last_speaker:
                # Format as "Speaker N:" to match common convention
                lines.append(f"\n{speaker}: {text}")
                last_speaker = speaker
            else:
                # Continue current speaker's text
                if lines:
                    lines.append(text)
                else:
                    lines.append(text)

        return "\n".join(lines).strip()

    def _get_timeout_seconds(self) -> float:
        """Get timeout from environment variable."""
        raw = os.getenv("VOXTRAL_TIMEOUT_SECONDS", os.getenv("DEEPGRAM_TIMEOUT_SECONDS", "300")).strip()
        try:
            return float(raw)
        except ValueError:
            logger.warning("Invalid timeout value=%r; defaulting to 300", raw)
            return 300.0
