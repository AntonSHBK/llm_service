import io
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from app.models.openai import OpenAITranscribeModel
from app.settings import settings

router = APIRouter(
    prefix="/audio",
    tags=["Audio"],
    responses={500: {"description": "Internal Server Error"}},
)

audio_service = OpenAITranscribeModel(
    api_key=settings.OPENAI_API_KEY,
    model_name="gpt-4o-mini-transcribe",
)
Т

class AudioResponse(BaseModel):
    text: str = Field(..., description="Результат транскрипции")

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "Добрый день! Сегодня мы поговорим о важности сна..."
            }
        }
    }


@router.post(
    "/transcribe",
    response_model=AudioResponse,
    summary="Транскрипция аудиофайла",
    description="""
    Принимает аудиофайл и возвращает его текстовую транскрипцию.

    **Поддерживаемые форматы**: mp3, wav, m4a и другие.

    **Пример запроса (cURL)**:
    ```bash
    curl -X POST "http://localhost:8000/audio/transcribe" \
    -H "accept: application/json" \
    -H "Content-Type: multipart/form-data" \
    -F "file=@sample.mp3"
    ```
    """,
)
async def transcribe_audio(file: UploadFile = File(...)) -> AudioResponse:
    try:
        audio_bytes = await file.read()
        audio_stream = io.BytesIO(audio_bytes)

        result_text = audio_service.generate(
            audio_stream,
            filename=file.filename
        )

        return AudioResponse(text=result_text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))