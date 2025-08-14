import sys
import os

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from app.main import app
from app.settings import settings

client = TestClient(app)


def test_health_check():
    """Проверка, что сервис работает."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.skipif(not settings.OPENAI_API_KEY, reason="OPENAI_API_KEY не задан")
def test_chat():
    """Проверка обычного режима чата."""
    payload = {
        "messages": [
            {"role": "user", "content": "Привет, как дела?"}
        ]
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
        "messages": [
            {"role": "user", "content": "Напиши 3 слова"}
        ]
    }
    response = client.post("/chat/stream", json=payload)
    assert response.status_code == 200
    # StreamingResponse возвращает generator, TestClient собирает его в bytes
    text = response.text
    assert isinstance(text, str)
    assert len(text.strip()) > 0
    print("\nПотоковый ответ:", text)
