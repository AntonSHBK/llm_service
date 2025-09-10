import sys
import os
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from app.main import app
from app.settings import settings

client = TestClient(app)


# @pytest.mark.skipif(not settings.OPENAI_API_KEY, reason="OPENAI_API_KEY не задан")
# def test_openai_image_bytes_endpoint():
#     """Проверка эндпоинта /image/openai_bytes"""
#     payload = {"prompt": "Простая иконка кота", "n": 1, "size": "256x256", "quality": "low"}
#     response = client.post("/image/openai_bytes", json=payload)

#     assert response.status_code == 200
#     assert response.headers["content-type"].startswith("image/") or response.headers["content-type"] == "application/zip"
#     assert len(response.content) > 1000, "Ответ должен содержать данные изображения"


# @pytest.mark.skipif(not settings.OPENAI_API_KEY, reason="OPENAI_API_KEY не задан")
# def test_openai_image_file_endpoint(tmp_path):
#     """Проверка эндпоинта /image/openai_file"""
#     payload = {"prompt": "Простая иконка собаки", "n": 1, "size": "256x256", "quality": "low"}
#     response = client.post("/image/openai_file", json=payload)

#     assert response.status_code == 200
#     assert "content-disposition" in response.headers
#     assert response.headers["content-type"].startswith("image/") or response.headers["content-type"] == "application/zip"
#     assert len(response.content) > 1000


@pytest.mark.skipif(not settings.YANDEX_API_KEY, reason="YANDEX_API_KEY не задан")
def test_yandex_image_bytes_endpoint():
    """Проверка эндпоинта /image/yandex_bytes"""
    payload = {"prompt": "Логотип в стиле минимализм", "n": 1, "width_ratio": "1", "height_ratio": "1"}
    response = client.post("/image/yandex_bytes", json=payload)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/") or response.headers["content-type"] == "application/zip"
    assert len(response.content) > 1000


@pytest.mark.skipif(not settings.YANDEX_API_KEY, reason="YANDEX_API_KEY не задан")
def test_yandex_image_file_endpoint():
    """Проверка эндпоинта /image/yandex_file"""
    payload = {"prompt": "Силуэт дерева", "n": 1, "width_ratio": "1", "height_ratio": "1"}
    response = client.post("/image/yandex_file", json=payload)

    assert response.status_code == 200
    assert "content-disposition" in response.headers
    assert response.headers["content-type"].startswith("image/") or response.headers["content-type"] == "application/zip"
    assert len(response.content) > 1000
