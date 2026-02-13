"""Base class for transcription providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class TranscriptionResult:
    """Result from a transcription provider."""
    raw_transcription: str
    detected_language: str | None = None


class TranscriptionProvider(ABC):
    """Base class for transcription providers."""

    @abstractmethod
    async def transcribe(
        self,
        audio_bytes: bytes,
        content_type: str,
        params: dict[str, str],
    ) -> TranscriptionResult:
        """
        Transcribe audio bytes.

        Args:
            audio_bytes: Raw audio data
            content_type: MIME type (e.g., "audio/wav")
            params: Provider-specific parameters

        Returns:
            TranscriptionResult with raw_transcription and detected_language
        """
        pass
