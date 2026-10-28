from typing import Any, Literal
from pydantic import BaseModel, Field

class ModelInfo(BaseModel):
    id: str
    model_name: str
    run_id: str | None = None
    checkpoint_available: bool
    calibration_available: bool

class Health(BaseModel):
    status: Literal["ok"] = "ok"
    models: list[ModelInfo]
    results_available: bool
    dataset_reports_available: bool

class ModelList(BaseModel):
    models: list[ModelInfo]

class Probability(BaseModel):
    class_name: str
    probability: float = Field(ge=0, le=1)

class Prediction(BaseModel):
    model_id: str
    purpose: str
    predicted_class: str
    raw_confidence: float = Field(ge=0, le=1)
    calibrated_confidence: float | None
    abstained: bool | None
    threshold: float | None
    top_k: list[Probability]
    probability_type: str
    notice: str

class Evaluation(BaseModel):
    schema_version: Literal[1]
    id: str
    run_id: str
    model_name: str
    purpose: str
    classes: list[str]
    domains: dict[str, Any]
    calibration: dict[str, Any]
    domain_gap: dict[str, float] | None
    efficiency: dict[str, Any]
    external_status: str
    protocol: str
    pretrained_loaded: bool
    source_run_id: str | None = None
    source_checkpoint_sha256: str | None = None

class Results(BaseModel):
    results: list[Evaluation]
    datasets: list[dict[str, Any]] = []
    warnings: list[str] = []
    next_command: str = "python -m research.scripts.run_experiments --config research/configs/quick_test.yaml"

class Plot(BaseModel):
    name: str
    run_id: str
    url: str

class Plots(BaseModel):
    plots: list[Plot]

class ExperimentStatus(BaseModel):
    id: str
    training: str
    in_domain: str
    external: str
    calibration: str
    efficiency: str
    missing_outputs: list[str]

class Experiments(BaseModel):
    experiments: list[ExperimentStatus]
