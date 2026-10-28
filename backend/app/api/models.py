from fastapi import APIRouter, Depends
from ..dependencies import settings
from ..schemas import ModelList
from ..services.model_service import list_models

router = APIRouter()

@router.get("/api/models", response_model=ModelList)
def models(config=Depends(settings)):
    return {"models": list_models(config)}
