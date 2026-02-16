"""
Unit tests for the FastAPI application endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, Mock
from io import BytesIO
import httpx
import os


@pytest.fixture(autouse=True)
def _unset_api_token(monkeypatch):
    """Ensure local env doesn't accidentally enable auth during tests."""
    monkeypatch.delenv("API_TOKEN", raising=False)


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    from api import app
    return TestClient(app)


@pytest.fixture
def valid_wav_content():
    """Create valid WAV file content for testing."""
    # Minimal WAV file header
    wav_header = b'RIFF'
    wav_header += b'\x24\x00\x00\x00'  # File size - 8
    wav_header += b'WAVE'
    wav_header += b'fmt '
    wav_header += b'\x10\x00\x00\x00'  # fmt chunk size
    wav_header += b'\x01\x00'  # Audio format (PCM)
    wav_header += b'\x01\x00'  # Number of channels
    wav_header += b'\x44\xac\x00\x00'  # Sample rate (44100)
    wav_header += b'\x88\x58\x01\x00'  # Byte rate
    wav_header += b'\x02\x00'  # Block align
    wav_header += b'\x10\x00'  # Bits per sample
    wav_header += b'data'
    wav_header += b'\x00\x00\x00\x00'  # Data chunk size
    return wav_header


class TestGetTranscription:
    """Tests for the /api/get_transcription endpoint."""

    def test_auth_enabled_missing_token_returns_401(self, client, valid_wav_content):
        with patch.dict(os.environ, {"API_TOKEN": "secret"}):
            response = client.post(
                "/api/get_transcription",
                files={"file": ("test.wav", valid_wav_content, "audio/wav")},
                data={"uniqueid": "1234567890.1234"},
            )

        assert response.status_code == 401

    def test_auth_enabled_wrong_token_returns_401(self, client, valid_wav_content):
        with patch.dict(os.environ, {"API_TOKEN": "secret"}):
            response = client.post(
                "/api/get_transcription",
                headers={"Authorization": "Bearer wrong"},
                files={"file": ("test.wav", valid_wav_content, "audio/wav")},
                data={"uniqueid": "1234567890.1234"},
            )

        assert response.status_code == 401

    @patch('transcription.deepgram.httpx.AsyncClient')
    @patch('api.get_provider')
    def test_auth_enabled_valid_token_allows_request(self, mock_get_provider, mock_client_class, client, valid_wav_content, monkeypatch):
        """When API_TOKEN is set, /api endpoints require a matching token."""
        monkeypatch.setenv("DEEPGRAM_API_KEY", "test_key")
        
        # Mock the Deepgram API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "results": {
                "paragraphs": {"transcript": "SPEAKER 1: Hello world"},
                "channels": [
                    {
                        "alternatives": [
                            {"transcript": "Hello world"}
                        ],
                        "detected_language": "en"
                    }
                ]
            }
        }
        mock_response.raise_for_status = Mock()
        mock_response.status_code = 200
        mock_response.text = "mock"
        mock_response.headers.get.return_value = "application/json"

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()
        mock_client_class.return_value = mock_client

        # Use actual provider
        from transcription import get_provider as real_get_provider
        mock_get_provider.side_effect = real_get_provider

        with patch.dict(os.environ, {"API_TOKEN": "secret", "DEEPGRAM_API_KEY": "test_key"}):
            response = client.post(
                "/api/get_transcription",
                headers={"Authorization": "Bearer secret"},
                files={"file": ("test.wav", valid_wav_content, "audio/wav")},
                data={"uniqueid": "1234567890.1234", "multichannel": "true"},
            )

        assert response.status_code == 200

    def test_docs_not_protected_by_api_token(self, client):
        with patch.dict(os.environ, {"API_TOKEN": "secret"}):
            response = client.get("/docs")

        assert response.status_code == 200

    @patch("api.db.is_configured", return_value=False)
    def test_missing_uniqueid(self, mock_db_configured, client, valid_wav_content):
        """Test that missing uniqueid is rejected when persistence is requested."""
        response = client.post(
            "/api/get_transcription",
            files={"file": ("test.wav", valid_wav_content, "audio/wav")},
            data={"persist": "true"},
        )

        assert response.status_code == 400
        assert "uniqueid" in response.json()["detail"]

    @patch('transcription.deepgram.httpx.AsyncClient')
    @patch('api.get_provider')
    def test_valid_wav_file(self, mock_get_provider, mock_client_class, client, valid_wav_content, monkeypatch):
        """Test transcription with a valid WAV file."""
        monkeypatch.setenv("DEEPGRAM_API_KEY", "test_key")
        
        # Mock the Deepgram API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "results": {
                "paragraphs": {"transcript": "SPEAKER 1: Hello world"},
                "channels": [
                    {
                        "alternatives": [
                            {"transcript": "Hello world"}
                        ],
                        "detected_language": "en"
                    }
                ]
            }
        }
        mock_response.raise_for_status = Mock()
        mock_response.status_code = 200
        mock_response.text = "mock"
        mock_response.headers.get.return_value = "application/json"
        
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()
        mock_client_class.return_value = mock_client

        # Use actual provider
        from transcription import get_provider as real_get_provider
        mock_get_provider.side_effect = real_get_provider

        # Make the request
        response = client.post(
            "/api/get_transcription",
            files={"file": ("test.wav", valid_wav_content, "audio/wav")},
            data={"uniqueid": "1234567890.1234", "multichannel": "true"},
        )

        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert "transcript" in data
        assert data["transcript"] == "SPEAKER 1: Hello world"

    @patch('transcription.deepgram.httpx.AsyncClient')
    @patch('api.get_provider')
    def test_persists_raw_transcript_via_threadpool(self, mock_get_provider, mock_client_class, client, valid_wav_content, monkeypatch):
        """Ensure persistence path uses threadpool helper and forwards kwargs to db layer."""
        monkeypatch.setenv("DEEPGRAM_API_KEY", "test_key")

        # Mock the Deepgram API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "results": {
                "paragraphs": {"transcript": "SPEAKER 1: Hello world"},
                "channels": [
                    {
                        "alternatives": [
                            {"transcript": "Hello world"}
                        ],
                        "detected_language": "en"
                    }
                ]
            }
        }
        mock_response.raise_for_status = Mock()
        mock_response.status_code = 200
        mock_response.text = "mock"
        mock_response.headers.get.return_value = "application/json"

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()
        mock_client_class.return_value = mock_client

        # Use actual provider
        from transcription import get_provider as real_get_provider
        mock_get_provider.side_effect = real_get_provider

        async def fake_run_in_threadpool(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch.dict(os.environ, {"OPENAI_API_KEY": "", "DEEPGRAM_API_KEY": "test_key"}), \
             patch("api.db.is_configured", return_value=True), \
             patch("api.db.upsert_transcript_progress", return_value=123) as progress_mock, \
             patch("api.db.upsert_transcript_raw", return_value=123) as upsert_mock, \
             patch("api.db.set_transcript_state") as state_mock, \
             patch("api.run_in_threadpool", new=fake_run_in_threadpool):
            response = client.post(
                "/api/get_transcription",
                files={"file": ("test.wav", valid_wav_content, "audio/wav")},
                data={"uniqueid": "1234567890.1234", "persist": "true", "multichannel": "true"},
            )

        assert response.status_code == 200

        progress_mock.assert_called_once_with(uniqueid="1234567890.1234")
        upsert_mock.assert_called_once_with(
            uniqueid="1234567890.1234",
            raw_transcription="SPEAKER 1: Hello world",
        )
        state_mock.assert_any_call(transcript_id=123, state="done")

    def test_invalid_file_type(self, client):
        """Test that non-WAV files are rejected."""
        response = client.post(
            "/api/get_transcription",
            files={"file": ("test.mp3", b"fake audio data", "audio/mp3")}
        )

        assert response.status_code == 400
        assert "Invalid file type" in response.json()["detail"]

    @patch('transcription.deepgram.httpx.AsyncClient')
    @patch('api.get_provider')
    def test_deepgram_api_error(self, mock_get_provider, mock_client_class, client, valid_wav_content, monkeypatch):
        """Test handling of Deepgram API errors."""
        monkeypatch.setenv("DEEPGRAM_API_KEY", "test_key")
        
        # Mock an HTTP error from Deepgram
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Invalid API key"
        
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "Unauthorized",
                request=Mock(),
                response=mock_response
            )
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_class.return_value = mock_client

        # Use actual provider
        from transcription import get_provider as real_get_provider
        mock_get_provider.side_effect = real_get_provider

        response = client.post(
            "/api/get_transcription",
            files={"file": ("test.wav", valid_wav_content, "audio/wav")},
            data={"uniqueid": "1234567890.1234"},
        )

        assert response.status_code == 401
        assert "API error" in response.json()["detail"]

    @patch('transcription.deepgram.httpx.AsyncClient')
    @patch('api.get_provider')
    def test_deepgram_timeout_returns_504(self, mock_get_provider, mock_client_class, client, valid_wav_content, monkeypatch):
        """Test that Deepgram timeouts are mapped to 504 Gateway Timeout."""
        monkeypatch.setenv("DEEPGRAM_API_KEY", "test_key")
        
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=httpx.ReadTimeout("Timed out", request=Mock())
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_class.return_value = mock_client

        # Use actual provider
        from transcription import get_provider as real_get_provider
        mock_get_provider.side_effect = real_get_provider

        response = client.post(
            "/api/get_transcription",
            files={"file": ("test.wav", valid_wav_content, "audio/wav")},
            data={"uniqueid": "1234567890.1234"},
        )

        assert response.status_code == 504
        assert "timed out" in response.json()["detail"].lower()

    @patch('transcription.deepgram.httpx.AsyncClient')
    @patch('api.get_provider')
    def test_malformed_deepgram_response(self, mock_get_provider, mock_client_class, client, valid_wav_content, monkeypatch):
        """Test handling of malformed responses from Deepgram."""
        monkeypatch.setenv("DEEPGRAM_API_KEY", "test_key")
        
        # Mock a response with missing fields
        mock_response = Mock()
        mock_response.json.return_value = {"results": {}}
        mock_response.raise_for_status = Mock()
        mock_response.status_code = 200
        mock_response.text = "bad response"
        mock_response.headers.get.return_value = "application/json"
        
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()
        mock_client_class.return_value = mock_client

        # Use actual provider
        from transcription import get_provider as real_get_provider
        mock_get_provider.side_effect = real_get_provider

        response = client.post(
            "/api/get_transcription",
            files={"file": ("test.wav", valid_wav_content, "audio/wav")},
            data={"uniqueid": "1234567890.1234"},
        )

        assert response.status_code == 500
        assert "Failed to parse transcription response" in response.json()["detail"]

    @patch('transcription.deepgram.httpx.AsyncClient')
    @patch('api.get_provider')
    def test_missing_paragraphs_transcript_is_error(self, mock_get_provider, mock_client_class, client, valid_wav_content, monkeypatch):
        """Diarized-only: missing paragraphs transcript returns 500."""
        monkeypatch.setenv("DEEPGRAM_API_KEY", "test_key")
        
        # Mock response without paragraphs transcript
        mock_response = Mock()
        mock_response.json.return_value = {
            "results": {
                "channels": [
                    {
                        "alternatives": [
                            {"transcript": "Test transcript"}
                        ]
                    }
                ]
            }
        }
        mock_response.raise_for_status = Mock()
        mock_response.status_code = 200
        mock_response.text = "bad"
        mock_response.headers.get.return_value = "application/json"
        
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()
        mock_client_class.return_value = mock_client

        # Use actual provider
        from transcription import get_provider as real_get_provider
        mock_get_provider.side_effect = real_get_provider

        response = client.post(
            "/api/get_transcription",
            files={"file": ("test.wav", valid_wav_content, "audio/wav")},
            data={"uniqueid": "1234567890.1234"},
        )

        assert response.status_code == 500
        assert "Failed to parse transcription response" in response.json()["detail"]

