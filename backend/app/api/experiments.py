from fastapi import APIRouter, Depends
from ..dependencies import settings
from ..schemas import Experiments
from ..services.model_service import list_models
from ..services.results_service import read_results

router = APIRouter()

@router.get("/api/experiments/status", response_model=Experiments)
def experiments(config=Depends(settings)):
    results = {r["id"]: r for r in read_results(config)["results"]}
    rows = []
    for model in list_models(config):
        result = results.get(model["id"])
        training_state = "reused" if result and result["purpose"] == "demo" else "completed" if model["checkpoint_available"] else "pending"
        state = {"id": model["id"], "training": training_state,
                 "in_domain": "completed" if result else "pending", "external": "completed" if result and "external" in result["domains"] else "pending",
                 "calibration": "completed" if result and model["calibration_available"] else "pending", "efficiency": "completed" if result else "pending"}
        state["missing_outputs"] = [k for k, v in state.items() if v == "pending"]
        rows.append(state)
    return {"experiments": rows}
