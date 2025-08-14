from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.models.openai_service import OpenAIService
from app.settings import settings

router = APIRouter(prefix="/image", tags=["Image"])

llm_service = OpenAIService(api_key=settings.OPENAI_API_KEY)

class ImageRequest(BaseModel):
    prompt: str

@router.post("/")
def image_endpoint(request: ImageRequest):
    try:
        result = llm_service.generate_image(prompt=request.prompt)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
