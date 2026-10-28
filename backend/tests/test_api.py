import io
import json
from pathlib import Path
import pytest
import torch
from PIL import Image
from fastapi.testclient import TestClient
from backend.app.main import create_app
from backend.app.config import Settings
from research.src.model_factory import build_model

@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(Settings(artifacts_dir=tmp_path, data_dir=tmp_path, max_upload_mb=1)))

def png():
    output = io.BytesIO()
    Image.new("RGB", (32,32), "green").save(output, format="PNG")
    return output.getvalue()

def test_health_and_missing_models(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/health").json()["results_available"] is False
    models = client.get("/api/models").json()["models"]
    assert len(models) == 2 and all(not m["checkpoint_available"] for m in models)

@pytest.mark.parametrize("route", ["summary","comparison","calibration"])
def test_empty_results(client, route):
    result = client.get(f"/api/results/{route}")
    assert result.status_code == 200 and result.json()["results"] == []
    assert "run_experiments" in result.json()["next_command"]

def test_status_and_plots(client):
    assert all(e["training"] == "pending" for e in client.get("/api/experiments/status").json()["experiments"])
    assert client.get("/api/results/plots").json() == {"plots": []}
    assert client.get("/api/results/plot/nope/secret.png").status_code == 404
    assert client.get("/api/results/plot/run/%2E%2E%2Fsecret.png").status_code == 404

def test_invalid_uploads_and_missing_checkpoint(client):
    assert client.post("/api/predict", data={"model_id":"bad"}, files={"file":("x.txt", b"bad", "text/plain")}).status_code == 415
    assert client.post("/api/predict", data={"model_id":"bad"}, files={"file":("x.png", b"bad", "image/png")}).status_code == 422
    assert client.post("/api/predict", data={"model_id":"bad"}, files={"file":("x.png", png(), "image/png")}).status_code == 503
    assert client.post("/api/predict", data={"model_id":"bad"}, files={"file":("x.png", b"0"*(1024**2+1), "image/png")}).status_code == 413
    assert client.post("/api/predict", data={"model_id":"../bad", "top_k":0}, files={"file":("x.png", png(), "image/png")}).status_code == 422

def test_cors(client):
    good = client.options("/api/predict", headers={"Origin":"http://localhost:3000", "Access-Control-Request-Method":"POST"})
    assert good.headers["access-control-allow-origin"] == "http://localhost:3000"
    bad = client.get("/health", headers={"Origin":"https://untrusted.example"})
    assert "access-control-allow-origin" not in bad.headers

def test_synthetic_checkpoint_prediction_and_cache(client):
    directory = client.app.state.settings.artifacts_dir / "models" / "fixture"
    directory.mkdir(parents=True)
    model = build_model("mobilenet_v3_small", 2)
    torch.save({"state_dict":model.state_dict(), "model_name":"mobilenet_v3_small", "classes":["a","b"], "image_size":64, "run_id":"fixture", "purpose":"synthetic", "manifest_sha256":"fixture"}, directory / "mobilenet_v3_small.pt")
    payload = {"data":{"model_id":"fixture__mobilenet_v3_small"}, "files":{"file":("x.png",png(),"image/png")}}
    result = client.post("/api/predict", **payload)
    assert result.status_code == 200, result.text
    value = result.json()
    assert value["purpose"] == "synthetic" and value["predicted_class"] in ["a","b"]
    assert value["abstained"] is None and value["calibrated_confidence"] is None
    assert sum(r["probability"] for r in value["top_k"]) == pytest.approx(1)
    assert client.post("/api/predict", **payload).status_code == 200
    assert len(client.app.state.inference.cache) == 1

def test_malformed_result_is_reported(client):
    directory = client.app.state.settings.artifacts_dir / "results" / "bad"
    directory.mkdir(parents=True)
    (directory / "model_evaluation.json").write_text('{"broken":true}')
    response = client.get("/api/results/summary")
    assert response.status_code == 200 and response.json()["warnings"]

def test_chunked_body_limit(client):
    def chunks():
        for _ in range(4):
            yield b"x"*(1024**2)
    response = client.post("/api/predict", content=chunks(), headers={"Content-Type":"application/octet-stream"})
    assert response.status_code == 413
