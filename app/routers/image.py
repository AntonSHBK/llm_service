import io
import uuid
import zipfile
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field

from app.models.openai import OpenAIImageModel
from app.models.yandex import YandexImageModel
from app.settings import settings

router = APIRouter(
    prefix="/image",
    tags=["Image"],
    responses={500: {"description": "Internal Server Error"}},
)

openai_image_service = OpenAIImageModel(
    api_key=settings.OPENAI_API_KEY,
    model_name="gpt-image-1",
    default_n=1,
)

yandex_image_service = YandexImageModel(
    folder_id=settings.YANDEX_FOLDER_ID,
    auth=settings.YANDEX_API_KEY,
    model_name="yandex-art",
)

# ==========================
# Requests
# ==========================

class ImageRequest(BaseModel):
    prompt: str = Field(..., description="Описание изображения для генерации")
    n: int = Field(1, description="Количество изображений (1–10)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "prompt": "Футуристический город на закате в стиле киберпанк",
                "n": 2,
                "seed": 1234,
                "style": "anime"
            }
        }
    }


# ==========================
# Helpers
# ==========================

def _return_images(images: list[bytes], single_name="generated.png"):
    """Вспомогательная функция для возврата изображений."""
    if len(images) == 1:
        return StreamingResponse(
            iter([images[0]]),
            media_type="image/png",
            headers={"Content-Disposition": f"inline; filename={single_name}"}
        )

    # Если несколько → ZIP
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


# ==========================
# Endpoints (OpenAI)
# ==========================

@router.post("/openai_bytes", summary="OpenAI → байты")
def generate_openai_bytes(request: ImageRequest):
    try:
        images = openai_image_service.generate(
            prompt=request.prompt,
            n=request.n,
            as_bytes=True,
            **request.model_dump(exclude={"prompt", "n"}, exclude_none=True)
        )
        return _return_images(images, "openai_image.png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/openai_file", summary="OpenAI → файл")
def generate_openai_file(request: ImageRequest):
    try:
        unique_name = f"{uuid.uuid4().hex}"
        output_path = settings.IMAGE_DIR / f"{unique_name}.png"

        images = openai_image_service.generate(
            prompt=request.prompt,
            n=request.n,
            output_path=output_path,
            as_bytes=False,
            **request.model_dump(exclude={"prompt", "n"}, exclude_none=True)
        )

        if len(images) == 1:
            return FileResponse(images[0], media_type="image/png", filename=images[0].name)

        zip_path = settings.IMAGE_DIR / f"{unique_name}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for i, img_path in enumerate(images, start=1):
                zip_file.write(img_path, f"image_{i}.png")

        return FileResponse(zip_path, media_type="application/zip", filename=zip_path.name)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================
# Endpoints (Yandex)
# ==========================

@router.post("/yandex_bytes", summary="Yandex → байты")
def generate_yandex_bytes(request: ImageRequest):
    try:
        images = []
        for _ in range(request.n):
            img_bytes = yandex_image_service.generate(
                prompt=request.prompt,
                as_bytes=True,
                **request.model_dump(exclude={"prompt", "n"}, exclude_none=True)
            )
            images.append(img_bytes)
        return _return_images(images, "yandex_image.png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/yandex_file", summary="Yandex → файл")
def generate_yandex_file(request: ImageRequest):
    try:        
        unique_name = f"{uuid.uuid4().hex}"
        
        images = []

        for i in range(request.n):
            output_path = settings.IMAGE_DIR / f"{unique_name}_{i+1}.png"
            result = yandex_image_service.generate(
                prompt=request.prompt,
                output_path=output_path,
                as_bytes=False,
                **request.model_dump(exclude={"prompt", "n"}, exclude_none=True)
            )
            images.append(result)

        if len(images) == 1:
            return FileResponse(images[0], media_type="image/png", filename=images[0].name)

        zip_path = settings.IMAGE_DIR / f"{unique_name}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for i, img_path in enumerate(images, start=1):
                zip_file.write(img_path, f"image_{i}.png")

        return FileResponse(zip_path, media_type="application/zip", filename=zip_path.name)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))