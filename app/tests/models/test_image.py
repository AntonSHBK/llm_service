import sys
import os
import io
import uuid
import pytest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from app.models.openai import OpenAIImageModel
from app.models.yandex import YandexImageModel
from app.settings import settings


DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


# ==========================
# Fixtures
# ==========================

@pytest.fixture(scope="module")
def openai_image():
    """Фикстура для OpenAI Image API."""
    return OpenAIImageModel(
        api_key=settings.OPENAI_API_KEY,
        model_name="gpt-image-1",
        default_n=1,
    )


@pytest.fixture(scope="module")
def yandex_image():
    """Фикстура для YandexART Image API."""
    return YandexImageModel(
        folder_id=settings.YANDEX_FOLDER_ID,
        auth=settings.YANDEX_API_KEY,
        model_name="yandex-art",
    )


# ==========================
# Tests (OpenAI)
# ==========================

# @pytest.mark.skipif(not settings.OPENAI_API_KEY, reason="OPENAI_API_KEY не задан")
# def test_openai_image_generate_bytes(openai_image: OpenAIImageModel):
#     """Проверка генерации изображения (байты) через OpenAI."""
#     images = openai_image.generate(
#         prompt="Маленький котёнок играет с клубком ниток",
#         n=1,
#         size="256x256",
#         as_bytes=True,
#     )

#     assert isinstance(images, list)
#     assert isinstance(images[0], (bytes, bytearray))
#     assert len(images[0]) > 1000, "Изображение должно быть больше 1KB"

#     print(f"\nOpenAI image (bytes) size: {len(images[0])} байт")


# @pytest.mark.skipif(not settings.OPENAI_API_KEY, reason="OPENAI_API_KEY не задан")
# def test_openai_image_generate_file(openai_image: OpenAIImageModel):
#     """Проверка генерации изображения (файл) через OpenAI."""
#     output_path = DATA_DIR / f"openai_test_{uuid.uuid4().hex}.png"
#     images = openai_image.generate(
#         prompt="Пейзаж с горами и озером",
#         n=1,
#         output_path=output_path,
#         as_bytes=False,
#         size="256x256",
#     )

#     path = images[0]
#     assert path.exists(), f"Файл не создан: {path}"
#     assert path.stat().st_size > 1000, "Файл слишком маленький"

#     print(f"\nOpenAI image file: {path}, size={path.stat().st_size}")


# ==========================
# Tests (Yandex)
# ==========================

@pytest.mark.skipif(not settings.YANDEX_API_KEY, reason="YANDEX_API_KEY не задан")
def test_yandex_image_generate_bytes(yandex_image: YandexImageModel):
    """Проверка генерации изображения (байты) через YandexART."""
    img_bytes = yandex_image.generate(
        prompt="Красочный закат над морем",
        as_bytes=True,
        width_ratio=1, 
        height_ratio=1,
    )

    assert isinstance(img_bytes, (bytes, bytearray))
    assert len(img_bytes) > 1000, "Изображение должно быть больше 1KB"

    print(f"\nYandex image (bytes) size: {len(img_bytes)} байт")


@pytest.mark.skipif(not settings.YANDEX_API_KEY, reason="YANDEX_API_KEY не задан")
def test_yandex_image_generate_file(yandex_image: YandexImageModel):
    """Проверка генерации изображения (файл) через YandexART."""
    output_path = DATA_DIR / f"yandex_test_{uuid.uuid4().hex}.png"
    path = yandex_image.generate(
        prompt="Футуристический город ночью",
        output_path=output_path,
        as_bytes=False,
        width_ratio=1, 
        height_ratio=1,
    )

    assert path.exists(), f"Файл не создан: {path}"
    assert path.stat().st_size > 1000, "Файл слишком маленький"

    print(f"\nYandex image file: {path}, size={path.stat().st_size}")
