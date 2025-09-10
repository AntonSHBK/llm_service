import sys
import os
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from app.main import app
from app.settings import settings

client = TestClient(app)

@pytest.mark.skipif(not settings.OPENAI_API_KEY, reason="OPENAI_API_KEY не задан")
def test_chat_text_endpoint():
    """Проверка эндпоинта /chat/text"""
    payload = {
        "input": [{"role": "user", "content": "Напиши короткий рассказ про кота"}]
    }
    response = client.post("/chat/text", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["response"], str)
    assert len(data["response"]) > 0

@pytest.mark.skipif(not settings.OPENAI_API_KEY, reason="OPENAI_API_KEY не задан")
def test_chat_text_stream_endpoint():
    """Проверка эндпоинта /chat/text_stream"""
    payload = {
        "input": [{"role": "user", "content": "Напиши короткий рассказ про собаку"}]
    }
    response = client.post("/chat/text_stream", json=payload)
    assert response.status_code == 200
    # так как стрим, проверим что хоть что-то вернулось
    content = b"".join(response.iter_bytes())
    assert len(content) > 0
    # NDJSON → первая строка должна быть JSON
    first_line = content.splitlines()[0].decode("utf-8")
    assert first_line.startswith("{") and first_line.endswith("}")

@pytest.mark.skipif(not settings.OPENAI_API_KEY, reason="OPENAI_API_KEY не задан")
def test_chat_text_schema_endpoint():
    """Проверка эндпоинта /chat/text_schema"""

    payload = {
        "input": [{"role": "user", "content": "У пользователя температура и кашель"}],
        "json_schema": {
            "format": {
                "type": "json_schema",
                "name": "diagnosis_detection",
                "schema": {
                    "type": "object",
                    "properties": {
                        "has_diagnosis": {"type": "boolean"},
                        "possible_diagnoses": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["has_diagnosis", "possible_diagnoses"],
                    "additionalProperties": False,
                },
                "strict": True
            }           
        }
    }
    
    response = client.post("/chat/text_schema", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "has_diagnosis" in data["response"]
    assert "possible_diagnoses" in data["response"]
