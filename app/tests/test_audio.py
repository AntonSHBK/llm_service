import sys
import os
import io
import pytest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from app.models.openai import OpenAIAudioModel
from app.settings import settings


@pytest.fixture(scope="module")
def openai_audio():
    """Фикстура для инициализации модели транскрипции."""
    return OpenAIAudioModel(
        api_key=settings.OPENAI_API_KEY,
        model_name="gpt-4o-mini-transcribe"
    )


def test_transcribe_audio(openai_audio: OpenAIAudioModel):
    """Проверка транскрипции аудиофайла (байтовый поток)."""
    audio_path = Path(__file__).parent / "test.wav"
    assert audio_path.exists(), f"Тестовый аудиофайл не найден: {audio_path}"

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    audio_stream = io.BytesIO(audio_bytes)
    result_text = openai_audio.transcribe(audio_stream, language="ru")

    assert isinstance(result_text, str), "Результат транскрипции должен быть строкой"
    assert len(result_text.strip()) > 0, "Результат транскрипции не должен быть пустым"

    print("\nРезультат транскрипции:", result_text)
