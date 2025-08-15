import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from app.models.openai import OpenAIChatModel
from app.settings import settings


@pytest.fixture(scope="module")
def openai_chat():
    """Фикстура для инициализации Chat-модели OpenAI."""
    return OpenAIChatModel(
        api_key=settings.OPENAI_API_KEY,
        model_name="gpt-4.1-nano",
        max_tokens=1024
    )


def test_chat_simple(openai_chat: OpenAIChatModel):
    """Проверка, что обычный чат возвращает непустой ответ."""
    messages = [{"role": "user", "content": "Привет, как тебя зовут?"}]
    response = openai_chat.chat(messages=messages)

    assert isinstance(response, str), "Ответ должен быть строкой"
    assert len(response.strip()) > 0, "Ответ не должен быть пустым"
    print("\nОтвет модели:", response)


def test_chat_streaming(openai_chat: OpenAIChatModel):
    """Проверка работы потокового режима (stream=True)."""
    messages = [{"role": "user", "content": "Напиши 3 слова"}]
    stream_gen = openai_chat.chat_stream(messages=messages)

    collected = ""
    for chunk in stream_gen:
        assert isinstance(chunk, str), "Каждый фрагмент должен быть строкой"
        collected += chunk

    assert len(collected.strip()) > 0, "Суммарный ответ не должен быть пустым"
    print("\nПотоковый ответ модели:", collected)
