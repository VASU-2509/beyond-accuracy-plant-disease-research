import io
import logging
import warnings
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool
from PIL import Image
from ..dependencies import settings, inference
from ..schemas import Prediction

router = APIRouter()

@router.post("/api/predict", response_model=Prediction)
async def predict(file: UploadFile = File(...), model_id: str = Form(...), top_k: int = Form(5, ge=1, le=10), config=Depends(settings), service=Depends(inference)):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(415, "Upload a JPEG, PNG, or WebP image.")
    limit = config.max_upload_mb * 1024**2
    try:
        payload = await file.read(limit+1)
    finally:
        await file.close()
    if len(payload) > limit:
        raise HTTPException(413, f"Image exceeds {config.max_upload_mb} MiB limit.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload)) as decoded:
                if decoded.format not in {"JPEG", "PNG", "WEBP"} or decoded.width*decoded.height > 20_000_000:
                    raise ValueError("Unsupported format or image exceeds 20 megapixels")
                decoded.load()
                image = decoded.copy()
    except (OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise HTTPException(422, "Image cannot be decoded safely. Use a valid JPEG, PNG, or WebP under 20 megapixels.")
    try:
        return await run_in_threadpool(service.predict, model_id, image, top_k)
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc))
    except (RuntimeError, ValueError, KeyError, ImportError, OSError):
        logging.exception("Inference failed for requested model")
        raise HTTPException(503, "Model or calibration could not be loaded. Check server logs and regenerate artifacts.")
