from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from ..dependencies import settings
from ..schemas import Results, Plots
from ..services.results_service import read_results, plot_paths

router = APIRouter()

@router.get("/api/results/summary", response_model=Results)
@router.get("/api/results/comparison", response_model=Results)
@router.get("/api/results/calibration", response_model=Results)
def results(config=Depends(settings)):
    return read_results(config)

@router.get("/api/results/plots", response_model=Plots)
def plots(config=Depends(settings)):
    return {"plots": [{"name": path.stem, "run_id": path.parent.name, "url": f"/api/results/plot/{key}"} for key, path in sorted(plot_paths(config).items())]}

@router.get("/api/results/plot/{run_id}/{filename}")
def plot(run_id: str, filename: str, config=Depends(settings)):
    path = plot_paths(config).get(f"{run_id}/{filename}")
    if path is None:
        raise HTTPException(404, "Figure not found")
    return FileResponse(path, media_type="image/png")
