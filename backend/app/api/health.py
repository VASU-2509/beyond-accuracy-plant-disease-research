from fastapi import APIRouter, Depends
from ..dependencies import settings
from ..schemas import Health
from ..services.model_service import list_models
from ..services.results_service import read_results

router = APIRouter()

@router.get("/health", response_model=Health)
def health(config=Depends(settings)):
    result = read_results(config)
    return {"status": "ok", "models": list_models(config), "results_available": bool(result["results"]), "dataset_reports_available": bool(result["datasets"])}
