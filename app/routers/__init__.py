from fastapi import APIRouter
from app.routers import chat, image, audio

api_router = APIRouter()
api_router.include_router(chat.router)
# api_router.include_router(image.router)
# api_router.include_router(audio.router)
