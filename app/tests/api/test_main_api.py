import sys
import os
import io
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from app.main import app

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
