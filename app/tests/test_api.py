import sys
import os
import io
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from app.main import app
from app.settings import settings

client = TestClient(app)


def test_root_page():
    """Проверка главной страницы."""
    response = client.get("/")
    assert response.status_code == 200
    text = response.text.lower()
    assert "сервис работает" in text
    assert "/docs" in text
    assert "/redoc" in text


def test_health_check():
    """Проверка health-check эндпоинта."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.skipif(not settings.OPENAI_API_KEY, reason="OPENAI_API_KEY не задан")
def test_chat():
    """Проверка обычного режима чата."""
    payload = {
        "messages": [{"role": "user", "content": "Привет, как дела?"}]
    }
    response = client.post("/chat/", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert isinstance(data["response"], str)
    assert len(data["response"].strip()) > 0
    print("\nОбычный ответ:", data["response"])


@pytest.mark.skipif(not settings.OPENAI_API_KEY, reason="OPENAI_API_KEY не задан")
def test_chat_stream():
    """Проверка потокового режима чата."""
    payload = {
        "messages": [{"role": "user", "content": "Напиши 3 слова"}]
    }
    response = client.post("/chat/stream", json=payload)
    assert response.status_code == 200
    text = response.text
    assert isinstance(text, str)
    assert len(text.strip()) > 0
    print("\nПотоковый ответ:", text)


# @pytest.mark.skipif(not settings.OPENAI_API_KEY, reason="OPENAI_API_KEY не задан")
# def test_generate_image():
#     """Проверка генерации изображения."""
#     payload = {"prompt": "Красивый закат над морем", "n": 1}
#     response = client.post("/image/generate", json=payload)
#     assert response.status_code == 200
#     data = response.json()
#     assert "images" in data
#     assert isinstance(data["images"], list)
#     assert len(data["images"]) == 1
#     assert all(isinstance(img, str) for img in data["images"])  # base64 строки
#     print("\nСгенерировано изображение (base64, первые 50 символов):", data["images"][0][:50])


@pytest.mark.skipif(not settings.OPENAI_API_KEY, reason="OPENAI_API_KEY не задан")
def test_transcribe_audio():
    """Проверка транскрипции аудио."""
    audio_path = os.path.join(os.path.dirname(__file__), "test.wav")
    assert os.path.exists(audio_path), "Файл test.wav отсутствует для теста"
    with open(audio_path, "rb") as f:
        files = {"file": ("test.wav", f, "audio/wav")}
        response = client.post("/audio/transcribe", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    assert isinstance(data["text"], str)
    assert len(data["text"].strip()) > 0
    print("\nРаспознанный текст:", data["text"])
