import base64
import io
import zipfile
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse

from app.models.openai import OpenAIImageModel
from app.settings import settings

router = APIRouter(
    prefix="/image",
    tags=["Image"],
    responses={500: {"description": "Internal Server Error"}},
)

image_service = OpenAIImageModel(
    api_key=settings.OPENAI_API_KEY,
    model_name="gpt-image-1",
    default_n=1,
)


class ImageRequest(BaseModel):
    prompt: str = Field(..., description="Описание изображения для генерации")
    n: int = Field(1, description="Количество изображений (1–10)")
    size: str = Field("1024x1024", description="Размер: 1024x1024, 1024x1536, 1536x1024, auto")
    quality: str = Field("medium", description="Качество: low, medium или high")

    model_config = {
        "json_schema_extra": {
            "example": {
                "prompt": "Футуристический город на закате в стиле киберпанк",
                "n": 2,
                "size": "1024x1024",
                "quality": "high"
            }
        }
    }


@router.post(
    "/generate",
    summary="Генерация изображения по описанию",
    description="""
    Создаёт одно или несколько изображений на основе текстового описания.
    
    **Если n=1** → возвращает PNG-файл.  
    **Если n>1** → возвращает ZIP-архив с картинками.
    """,
)
def generate_image_endpoint(request: ImageRequest):
    try:
        images = image_service.generate(
            prompt=request.prompt,
            n=request.n,
            size=request.size,
            quality=request.quality
        )

        if request.n == 1:
            return StreamingResponse(
                iter([images[0]]),
                media_type="image/png",
                headers={"Content-Disposition": "inline; filename=generated.png"}
            )

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for i, img_bytes in enumerate(images, start=1):
                zip_file.writestr(f"image_{i}.png", img_bytes)

        zip_buffer.seek(0)
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=images.zip"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
