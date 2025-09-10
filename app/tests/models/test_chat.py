import sys
import os
import json
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from app.models.openai import OpenAITextModel
from app.settings import settings


@pytest.fixture(scope="module")
def openai_chat():
    """Фикстура для инициализации Chat-модели OpenAI."""
    return OpenAITextModel(
        api_key=settings.OPENAI_API_KEY,
        model_name="gpt-5-nano",
        max_tokens=512
    )


def test_chat_simple(openai_chat: OpenAITextModel):
    """Проверка, что обычный чат возвращает непустой ответ."""
    messages = [{"role": "user", "content": "Привет, как тебя зовут?"}]
    response = openai_chat.generate(input=messages)

    assert hasattr(response, "output_text"), "У ответа должно быть поле output_text"
    assert isinstance(response.output_text, str)
    assert len(response.output_text.strip()) > 0
    print("\nОтвет модели:", response.output_text)


def test_chat_streaming(openai_chat: OpenAITextModel):
    """Проверка работы потокового режима (stream=True)."""
    messages = [{"role": "user", "content": "Напиши 3 слова"}]
    stream = openai_chat.generate_stream(input=messages)

    collected = ""
    for event in stream:
        if event.type == "response.output_text.delta":
            collected += event.delta

    assert len(collected.strip()) > 0
    print("\nПотоковый ответ модели:", collected)


def test_chat_with_json_schema(openai_chat: OpenAITextModel):
    """Проверка Structured Outputs с ручным JSON Schema."""
    messages = [{"role": "user", "content": "Alice и Bob идут на ярмарку в пятницу, у них много яблок и груш"}]

    json_schema = {
        "format": {
            "type": "json_schema",
            "name": "event",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "date": {"type": "string"},
                    "participants": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["name", "date", "participants"],
                "additionalProperties": False,
            },
        }
    }

    response = openai_chat.generate_with_schema(
        input=messages,
        json_schema=json_schema
    )

    data = json.loads(response.output_text)
    assert "name" in data
    assert "date" in data
    assert "participants" in data
    assert isinstance(data["participants"], list)
    print("\nStructured Outputs:", data)
