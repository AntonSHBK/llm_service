from fastapi import APIRouter, UploadFile, File, HTTPException

from app.models.openai_service import OpenAIService
from app.settings import settings

router = APIRouter(prefix="/audio", tags=["Audio"])

llm_service = OpenAIService(api_key=settings.OPENAI_API_KEY)

@router.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        text = llm_service.transcribe_audio(temp_path)
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
