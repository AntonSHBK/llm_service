from typing import List

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.models.openai import OpenAITextModel
from app.settings import settings

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
    responses={500: {"description": "Internal Server Error"}},
)

chat_service = OpenAITextModel(
    api_key=settings.OPENAI_API_KEY,
    model_name="gpt-4.1-nano",
    max_tokens=1024,
)


class Message(BaseModel):
    role: str = Field(..., description="Роль: 'user', 'system' или 'assistant'")
    content: str = Field(..., description="Текст сообщения")


class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., description="Список сообщений для модели")

    model_config = {
        "json_schema_extra": {
            "example": {
                "messages": [
                    {"role": "user", "content": "Напиши короткий рассказ про кота"}
                ]
            }
        }
    }


class ChatResponse(BaseModel):
    response: str = Field(..., description="Сгенерированный ответ модели")

    model_config = {
        "json_schema_extra": {
            "example": {
                "response": "Жил-был кот по имени Барсик..."
            }
        }
    }


@router.post(
    "/",
    response_model=ChatResponse,
    summary="Обычный чат-запрос",
    description="""
    Отправляет список сообщений в модель и возвращает полный ответ целиком.
    
    **Пример**:
    ```json
    {
      "messages": [
        {"role": "user", "content": "Привет, как дела?"}
      ]
    }
    ```
    """,
)
def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """Обработка обычного (непотокового) чата."""
    try:
        response_text = chat_service.chat(
            messages=[msg.model_dump() for msg in request.messages]
        )
        return ChatResponse(response=response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/stream",
    summary="Потоковый чат-запрос",
    description="""
    Отправляет список сообщений в модель и возвращает ответ **частями** в реальном времени.
    Полезно для чатов, где нужно отображать текст по мере генерации.
    
    **Пример**:
    ```json
    {
      "messages": [
        {"role": "user", "content": "Напиши короткий рассказ про кота"}
      ]
    }
    ```
    """,
)
def chat_stream_endpoint(request: ChatRequest) -> StreamingResponse:
    """Обработка потокового чата."""
    try:
        stream_gen = chat_service.chat_stream(
            messages=[msg.model_dump() for msg in request.messages]
        )
        return StreamingResponse(stream_gen, media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
