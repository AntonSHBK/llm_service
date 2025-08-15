import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from app.models.openai import OpenAIImageModel
from app.settings import settings


@pytest.fixture(scope="module")
def openai_image():
    """Фикстура для инициализации модели генерации изображений."""
    return OpenAIImageModel(
        api_key=settings.OPENAI_API_KEY,
        model_name="dall-e-2",
        default_n=1,
        default_size="1024x1024",
        default_quality="Standard"
    )


def test_generate_single_image(openai_image: OpenAIImageModel):
    """Проверка генерации одного изображения."""
    images = openai_image.generate(prompt="Красивый закат над морем", n=1, size="1024x1024", quality="low")

    assert isinstance(images, list), "Результат должен быть списком"
    assert len(images) == 1, "Должно вернуться одно изображение"
    assert isinstance(images[0], bytes), "Изображение должно быть в формате bytes"
    assert len(images[0]) > 1000, "Изображение должно быть больше 1KB"

    # Сохраняем во временный файл (для ручной проверки)
    with open("test_image.png", "wb") as f:
        f.write(images[0])

    print("\nСгенерировано изображение test_image.png")


# def test_generate_multiple_images(openai_image):
    # """Проверка генерации нескольких изображений."""
    # images = openai_image.generate(prompt="Космическая станция на орбите", n=2)

    # assert isinstance(images, list), "Результат должен быть списком"
    # assert len(images) == 2, "Должно вернуться два изображения"
    # for img in images:
    #     assert isinstance(img, bytes), "Каждое изображение должно быть в формате bytes"
    #     assert len(img) > 1000, "Размер изображения должен быть больше 1KB"

    # # Сохраняем для ручной проверки
    # for i, img in enumerate(images, start=1):
    #     with open(f"test_image_{i}.png", "wb") as f:
    #         f.write(img)

    # print("\nСгенерированы изображения test_image_1.png и test_image_2.png")
