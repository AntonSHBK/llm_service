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
    model_name="gpt-5-nano",
    max_tokens=1024,
)


class Message(BaseModel):
    role: str = Field(..., description="Роль: 'user', 'system', 'assistant' и т.д.")
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
    "/text",
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
    try:
        resp = chat_service.generate(
            input=[msg.model_dump() for msg in request.messages]
        )
        return ChatResponse(response=resp.output_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/text_stream",
    summary="Потоковый чат-запрос",
    description="""
    Отправляет список сообщений в модель и возвращает ответ **частями** в реальном времени.
    
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
        stream = chat_service.generate_stream(
            input=[msg.model_dump() for msg in request.messages]
        )

        def stream_gen():
            for event in stream:
                if event.type == "response.output_text.delta":
                    yield event.delta
                elif event.type == "response.refusal.delta":
                    yield f"[REFUSAL] {event.delta}"

        return StreamingResponse(stream_gen(), media_type="text/plain")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# class StructuredRequest(BaseModel):
#     query: str = Field(..., description="Пользовательский запрос для генерации")

# class StructuredResponse(BaseModel):
#     title: str
#     summary: str
#     tags: List[str]

# @router.post("/text_structured", response_model=StructuredResponse)
# def generate_structured_text(payload: StructuredRequest):
#     try:
#         llm = OpenAITextModel(api_key=settings.OPENAI_API_KEY)

#         # задаём системный промт, чтобы модель ответила строго в JSON
#         system_msg = llm.complete_input(
#             "system",
#             "Ты помощник, который отвечает строго в JSON с ключами: title, summary, tags."
#         )
#         user_msg = llm.get_message("user", payload.query)

#         messages = [system_msg, user_msg]

#         response_text = llm.generate(
#             messages=messages,
#             response_format={"type": "json_object"}
#         )

#         return StructuredResponse.parse_raw(response_text)

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
