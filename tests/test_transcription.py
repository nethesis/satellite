"""Tests for transcription providers."""

import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock, patch
from transcription import get_provider, TranscriptionResult
from transcription.deepgram import DeepgramProvider
from transcription.voxtral import VoxtralProvider


class TestGetProvider:
    """Tests for provider factory."""

    def test_get_provider_deepgram_default(self, monkeypatch):
        """Test default provider is Deepgram."""
        monkeypatch.setenv("DEEPGRAM_API_KEY", "test_key")
        monkeypatch.delenv("TRANSCRIPTION_PROVIDER", raising=False)
        provider = get_provider()
        assert isinstance(provider, DeepgramProvider)

    def test_get_provider_deepgram_explicit(self, monkeypatch):
        """Test explicit Deepgram provider."""
        monkeypatch.setenv("DEEPGRAM_API_KEY", "test_key")
        provider = get_provider("deepgram")
        assert isinstance(provider, DeepgramProvider)

    def test_get_provider_voxtral(self, monkeypatch):
        """Test VoxTral provider."""
        monkeypatch.setenv("MISTRAL_API_KEY", "test_key")
        provider = get_provider("voxtral")
        assert isinstance(provider, VoxtralProvider)

    def test_get_provider_from_env(self, monkeypatch):
        """Test provider selection from env var."""
        monkeypatch.setenv("TRANSCRIPTION_PROVIDER", "voxtral")
        monkeypatch.setenv("MISTRAL_API_KEY", "test_key")
        provider = get_provider()
        assert isinstance(provider, VoxtralProvider)

    def test_get_provider_missing_api_key(self, monkeypatch):
        """Test error when API key is missing."""
        monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
        with pytest.raises(ValueError, match="DEEPGRAM_API_KEY is required"):
            get_provider("deepgram")

    def test_get_provider_unknown(self, monkeypatch):
        """Test error with unknown provider."""
        with pytest.raises(ValueError, match="Unknown transcription provider"):
            get_provider("unknown")


class TestDeepgramProvider:
    """Tests for Deepgram provider."""

    @pytest.mark.asyncio
    async def test_transcribe_success(self, monkeypatch):
        """Test successful Deepgram transcription."""
        monkeypatch.setenv("DEEPGRAM_API_KEY", "test_key")

        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "mock response"
        mock_response.json.return_value = {
            "results": {
                "channels": [{
                    "alternatives": [{
                        "paragraphs": {
                            "transcript": "Test transcription"
                        }
                    }],
                    "detected_language": "en"
                }]
            }
        }
        mock_response.headers.get.return_value = "application/json"

        # Mock httpx client
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.post.return_value = mock_response

        with patch("transcription.deepgram.httpx.AsyncClient", return_value=mock_client):
            provider = DeepgramProvider()
            result = await provider.transcribe(
                audio_bytes=b"fake audio",
                content_type="audio/wav",
                params={}
            )

        assert isinstance(result, TranscriptionResult)
        assert result.raw_transcription == "Test transcription"
        assert result.detected_language == "en"

    @pytest.mark.asyncio
    async def test_transcribe_paragraphs_format(self, monkeypatch):
        """Test Deepgram transcription with paragraphs at top level."""
        monkeypatch.setenv("DEEPGRAM_API_KEY", "test_key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "mock"
        mock_response.json.return_value = {
            "results": {
                "paragraphs": {
                    "transcript": "Top level transcript"
                }
            }
        }
        mock_response.headers.get.return_value = "application/json"

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.post.return_value = mock_response

        with patch("transcription.deepgram.httpx.AsyncClient", return_value=mock_client):
            provider = DeepgramProvider()
            result = await provider.transcribe(
                audio_bytes=b"fake audio",
                content_type="audio/wav",
                params={}
            )

        assert result.raw_transcription == "Top level transcript"

    @pytest.mark.asyncio
    async def test_transcribe_missing_transcript(self, monkeypatch):
        """Test error when transcript is missing from response."""
        monkeypatch.setenv("DEEPGRAM_API_KEY", "test_key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "invalid response"
        mock_response.json.return_value = {"results": {}}
        mock_response.headers.get.return_value = "application/json"

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.post.return_value = mock_response

        with patch("transcription.deepgram.httpx.AsyncClient", return_value=mock_client):
            provider = DeepgramProvider()
            with pytest.raises(ValueError, match="Failed to parse transcription response"):
                await provider.transcribe(
                    audio_bytes=b"fake audio",
                    content_type="audio/wav",
                    params={}
                )


class TestVoxtralProvider:
    """Tests for VoxTral provider."""

    @pytest.mark.asyncio
    async def test_transcribe_success(self, monkeypatch):
        """Test successful VoxTral transcription."""
        monkeypatch.setenv("MISTRAL_API_KEY", "test_key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "mock response"
        mock_response.json.return_value = {
            "text": "VoxTral transcription",
            "language": "en",
            "model": "voxtral-mini-latest"
        }
        mock_response.headers.get.return_value = "application/json"

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.post.return_value = mock_response

        with patch("transcription.voxtral.httpx.AsyncClient", return_value=mock_client):
            provider = VoxtralProvider()
            result = await provider.transcribe(
                audio_bytes=b"fake audio",
                content_type="audio/wav",
                params={}
            )

        assert isinstance(result, TranscriptionResult)
        assert result.raw_transcription == "VoxTral transcription"
        assert result.detected_language == "en"

    @pytest.mark.asyncio
    async def test_transcribe_with_diarization(self, monkeypatch):
        """Test VoxTral transcription with speaker diarization."""
        monkeypatch.setenv("MISTRAL_API_KEY", "test_key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "mock"
        mock_response.json.return_value = {
            "text": "ignored",
            "language": "en",
            "segments": [
                {"speaker": 0, "text": "Hello", "start": 0.0, "end": 1.0},
                {"speaker": 0, "text": "world", "start": 1.0, "end": 2.0},
                {"speaker": 1, "text": "Hi there", "start": 2.0, "end": 3.0},
            ]
        }
        mock_response.headers.get.return_value = "application/json"

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.post.return_value = mock_response

        with patch("transcription.voxtral.httpx.AsyncClient", return_value=mock_client):
            provider = VoxtralProvider()
            result = await provider.transcribe(
                audio_bytes=b"fake audio",
                content_type="audio/wav",
                params={"diarize": "true"}
            )

        # Should format with speaker labels
        assert "Speaker 0:" in result.raw_transcription
        assert "Speaker 1:" in result.raw_transcription
        assert "Hello" in result.raw_transcription
        assert "Hi there" in result.raw_transcription

    @pytest.mark.asyncio
    async def test_transcribe_empty_response(self, monkeypatch):
        """Test that VoxTral handles empty transcription (silence) gracefully."""
        monkeypatch.setenv("MISTRAL_API_KEY", "test_key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "empty"
        mock_response.json.return_value = {"text": "", "language": "en"}
        mock_response.headers.get.return_value = "application/json"

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.post.return_value = mock_response

        with patch("transcription.voxtral.httpx.AsyncClient", return_value=mock_client):
            provider = VoxtralProvider()
            # Empty transcription is valid (no speech detected)
            result = await provider.transcribe(
                audio_bytes=b"fake audio",
                content_type="audio/wav",
                params={}
            )
            assert result.raw_transcription == ""
            assert result.detected_language == "en"
