import sys
import os
import io
import pytest
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from app.main import app
from app.settings import settings

client = TestClient(app)

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

@pytest.mark.skipif(not settings.OPENAI_API_KEY, reason="OPENAI_API_KEY не задан")
def test_audio_transcribe_endpoint():
    """Проверка эндпоинта /audio/transcribe"""
    test_file = DATA_DIR / "test.wav"

    with open(test_file, "rb") as f:
        files = {"file": ("test.wav", f, "audio/wav")}
        response = client.post("/audio/transcribe", files=files)

    assert response.status_code == 200
    data = response.json()
    assert "text" in data

@pytest.mark.skipif(not settings.OPENAI_API_KEY, reason="OPENAI_API_KEY не задан")
def test_tts_file_endpoint():
    """Проверка эндпоинта /audio/tts_file"""
    payload = {"text": "Привет, это тест TTS", "voice": "alloy", "format": "wav"}
    response = client.post("/audio/tts_file", json=payload)

    assert response.status_code == 200
    # Проверяем заголовки
    assert response.headers["content-type"].startswith("audio/")
    assert "content-disposition" in response.headers

@pytest.mark.skipif(not settings.OPENAI_API_KEY, reason="OPENAI_API_KEY не задан")
def test_tts_bytes_endpoint():
    """Проверка эндпоинта /audio/tts_bytes"""
    payload = {"text": "Привет, это тест TTS поток", "voice": "alloy", "format": "mp3"}
    response = client.post("/audio/tts_bytes", json=payload)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/")
    # Проверяем, что что-то вернулось в теле
    content = b"".join(response.iter_bytes())
    assert len(content) > 0
