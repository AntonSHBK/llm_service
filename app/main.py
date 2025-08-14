from fastapi import FastAPI

from app.routers import api_router
from app.settings import settings
from app.utils.logging import setup_logging

setup_logging(log_dir=settings.LOG_DIR, log_level=settings.LOG_LEVEL)

app = FastAPI(
    title="LLM Microservice",
    version="0.1.0",
    description="""
    Микросервис для взаимодействия с LLM (OpenAI, LangChain и т.д.).

    **Доступные функции:**
    - Отправка сообщений в чат-модель (полный ответ)
    - Отправка сообщений в чат-модель (потоковый ответ)
    - (В будущем) генерация изображений, транскрипция аудио и др.
    
    **Формат сообщений**:
    - `role`: "user" | "system" | "assistant"
    - `content`: текст сообщения
    """,
    contact={
        "name": "AID",
        "email": "anton42@yandex.ru"
    }
)

app.include_router(api_router)


@app.get("/health", summary="Проверка работоспособности")
def health_check():
    """
    Возвращает `status: ok`, если сервис запущен.
    """
    return {"status": "ok"}
