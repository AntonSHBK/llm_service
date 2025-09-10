import sys
import os
import io
import pytest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from app.models.openai import OpenAITranscribeModel, OpenAITTSModel
from app.settings import settings


DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


@pytest.fixture(scope="module")
def openai_audio():
    """Фикстура для инициализации модели транскрипции."""
    return OpenAITranscribeModel(
        api_key=settings.OPENAI_API_KEY,
        model_name="gpt-4o-mini-transcribe"
    )


@pytest.fixture(scope="module")
def openai_tts():
    """Фикстура для инициализации модели генерации речи (TTS)."""
    return OpenAITTSModel(
        api_key=settings.OPENAI_API_KEY,
        model_name="gpt-4o-mini-tts"
    )


def test_transcribe_audio(openai_audio: OpenAITranscribeModel):
    """Проверка транскрипции готового тестового файла."""
    audio_path = DATA_DIR / "test.wav"
    assert audio_path.exists(), f"Тестовый аудиофайл не найден: {audio_path}"

    with open(audio_path, "rb") as f:
        audio_stream = io.BytesIO(f.read())
    result_text = openai_audio.generate(
        audio_stream,
        filename=audio_path.name,
        language="ru"
    )

    assert isinstance(result_text, str), "Результат транскрипции должен быть строкой"
    assert len(result_text.strip()) > 0, "Результат транскрипции не должен быть пустым"

    print("\nРезультат транскрипции:", result_text)


def test_tts_generate_audio(openai_tts: OpenAITTSModel):
    """Проверка генерации аудио из текста (TTS)."""
    text = "Привет, это тест синтеза речи."
    output_path = DATA_DIR / "tts_test.wav"

    path = openai_tts.generate(
        text=text,
        output_path=output_path,
        voice="alloy",
        response_format="wav"
    )

    assert path.exists(), f"Аудиофайл не создан: {path}"
    assert path.stat().st_size > 1000, "Размер аудиофайла должен быть больше 1KB"

    print(f"\nСгенерировано аудио: {path}")


# def test_tts_and_transcribe_integration(openai_tts: OpenAITTSModel, openai_audio: OpenAITranscribeModel):
#     """Интеграционный тест: TTS → Transcribe."""
#     text = "Это интеграционный тест синтеза и распознавания речи."
#     tts_path = DATA_DIR / "integration_test.wav"

#     # 1. Генерируем речь
#     generated_path = openai_tts.generate(
#         text=text,
#         output_path=tts_path,
#         voice="alloy",
#         response_format="wav"
#     )

#     assert generated_path.exists(), "TTS не создал файл"
#     assert generated_path.stat().st_size > 1000, "TTS файл слишком маленький"

#     # 2. Отправляем на транскрипцию
#     result_text = openai_audio.generate(str(generated_path), language="ru")

#     assert isinstance(result_text, str)
#     assert len(result_text.strip()) > 0
#     print("\nИсходный текст:", text)
#     print("Распознанный текст:", result_text)
