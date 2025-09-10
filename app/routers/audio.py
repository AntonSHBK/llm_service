import io
import uuid
from pathlib import Path
from starlette.concurrency import run_in_threadpool

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.models.openai import OpenAITranscribeModel, OpenAITTSModel
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

tts_service = OpenAITTSModel(
    api_key=settings.OPENAI_API_KEY,
    model_name="gpt-4o-mini-tts",
)

# ==========================
# Requests
# ==========================

class TTSRequest(BaseModel):
    text: str = Field(..., description="Текст для генерации речи")
    voice: str = Field("alloy", description="Голос (по умолчанию alloy)")
    format: str = Field("wav", description="Формат аудио (wav/mp3/ogg)")

# ==========================
# Requests
# ==========================

class AudioResponse(BaseModel):
    text: str = Field(..., description="Результат транскрипции")

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "Добрый день! Сегодня мы поговорим о важности сна..."
            }
        }
    }

# ==========================
# Endpoints
# ==========================

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

        result_text = await run_in_threadpool(
            audio_service.generate,
            audio_source=audio_stream,
            filename=file.filename,
        )

        return AudioResponse(text=result_text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post(
    "/tts_file",
    summary="Генерация аудио (файл)",
    description="""
    Генерирует аудиофайл на основе переданного текста и возвращает его как загружаемый файл.  
    Файл сохраняется на сервере в директории `data/audio/` с уникальным именем.  

    **Поддерживаемые форматы**: wav, mp3, ogg  

    **Когда использовать**:  
    - Если нужно получить готовый аудиофайл и сохранить его у клиента.  
    - Подходит для сценариев, где требуется дальнейшая работа с файлом (например, пересылка).  

    **Пример запроса (cURL)**:
    ```bash
    curl -X POST "http://localhost:8000/tts/tts_file" \
    -H "accept: application/json" \
    -H "Content-Type: application/json" \
    -d '{"text": "Привет, как твои дела?", "voice": "alloy", "format": "mp3"}' \
    --output result.mp3
    ```
    """,
)
async def generate_tts_file(request: TTSRequest):
    try:
        unique_name = f"{uuid.uuid4().hex}.{request.format}"
        output_path = settings.AUDIO_DIR / unique_name
        tts_service.generate(
            text=request.text,
            output_path=output_path,
            voice=request.voice,
            response_format=request.format,
            as_bytes=False,
        )
        return FileResponse(
            path=output_path,
            filename=output_path.name,
            media_type=f"audio/{request.format}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/tts_bytes",
    summary="Генерация аудио (поток)",
    description="""
    Генерирует аудио в памяти и возвращает его в бинарном виде через потоковый ответ (`StreamingResponse`).  

    **Поддерживаемые форматы**: wav, mp3, ogg  

    **Когда использовать**:  
    - Если нужно воспроизвести аудио сразу (например, в браузере или мобильном приложении).  
    - Не сохраняет файл на сервере, возвращает только поток.  

    **Пример запроса (cURL)**:
    ```bash
    curl -X POST "http://localhost:8000/tts/tts_bytes" \
    -H "accept: audio/mpeg" \
    -H "Content-Type: application/json" \
    -d '{"text": "Привет, как твои дела?", "voice": "alloy", "format": "mp3"}' \
    --output result.mp3
    ```
    """,
)
async def generate_tts_bytes(request: TTSRequest):
    try:
        audio_bytes = tts_service.generate(
            text=request.text,
            voice=request.voice,
            response_format=request.format,
            as_bytes=True,
        )
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type=f"audio/{request.format}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))