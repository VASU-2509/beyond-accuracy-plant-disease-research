import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from .config import Settings
from .api import health, predictions, models, experiments, research_results
from .services.inference_service import InferenceService
from .middleware import BodyLimitMiddleware

def create_app(config=None):
    config = config or Settings()
    app = FastAPI(title="Beyond Accuracy", version="0.1.0")
    app.state.settings = config
    app.state.inference = InferenceService(config)
    app.add_middleware(BodyLimitMiddleware, max_bytes=(config.max_upload_mb+1)*1024**2)
    app.add_middleware(CORSMiddleware, allow_origins=config.cors_origins, allow_credentials=False,
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type"])
    @app.middleware("http")
    async def limit_request(request: Request, call_next):
        length = request.headers.get("content-length")
        if length and (not length.isdigit() or int(length) > (config.max_upload_mb+1)*1024**2):
            return JSONResponse({"detail": "Request body too large or invalid Content-Length"}, status_code=413)
        return await call_next(request)
    for router in [health.router, models.router, predictions.router, research_results.router, experiments.router]:
        app.include_router(router)
    return app

logging.basicConfig(level=logging.INFO)
app = create_app()
