import re
from pathlib import Path

NAMES = ("resnet18", "mobilenet_v3_small")
SAFE_RUN = re.compile(r"^[A-Za-z0-9_-]+$")

def checkpoints(settings):
    root = settings.artifacts_dir.resolve() / "models"
    found = {}
    if root.exists():
        for path in root.glob("*/*.pt"):
            if path.stem in NAMES and SAFE_RUN.fullmatch(path.parent.name) and path.resolve().is_relative_to(root.resolve()):
                found[f"{path.parent.name}__{path.stem}"] = path
    return found

def list_models(settings):
    rows = [{"id": key, "model_name": path.stem, "run_id": path.parent.name, "checkpoint_available": True,
             "calibration_available": path.with_suffix(".calibration.json").exists()} for key, path in sorted(checkpoints(settings).items())]
    present = {r["model_name"] for r in rows}
    rows += [{"id": name, "model_name": name, "run_id": None, "checkpoint_available": False, "calibration_available": False} for name in NAMES if name not in present]
    return rows
