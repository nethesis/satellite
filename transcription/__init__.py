"""Transcription provider abstraction."""

import logging
import os
from .base import TranscriptionProvider, TranscriptionResult
from .deepgram import DeepgramProvider
from .voxtral import VoxtralProvider

logger = logging.getLogger(__name__)

__all__ = ["TranscriptionProvider", "TranscriptionResult", "get_provider"]


def get_provider(name: str | None = None) -> TranscriptionProvider:
    """
    Get a transcription provider instance.

    Args:
        name: Provider name ("deepgram" or "voxtral"). If None, uses TRANSCRIPTION_PROVIDER env var.

    Returns:
        TranscriptionProvider instance

    Raises:
        ValueError: If provider name is unknown or required API key is missing
    """
    if name is None:
        name = os.getenv("TRANSCRIPTION_PROVIDER", "deepgram").strip().lower()

    if name == "deepgram":
        api_key = os.getenv("DEEPGRAM_API_KEY", "").strip()
        if not api_key:
            raise ValueError("DEEPGRAM_API_KEY is required for Deepgram provider")
        return DeepgramProvider(api_key=api_key)
    elif name == "voxtral":
        api_key = os.getenv("MISTRAL_API_KEY", "").strip()
        if not api_key:
            raise ValueError("MISTRAL_API_KEY is required for VoxTral provider")
        return VoxtralProvider(api_key=api_key)
    else:
        raise ValueError(f"Unknown transcription provider: {name}. Valid options: deepgram, voxtral")
