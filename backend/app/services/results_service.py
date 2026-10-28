import json
import logging
from pydantic import ValidationError
from ..schemas import Evaluation

def read_results(settings):
    rows, warnings, datasets = [], [], []
    for path in sorted(settings.artifacts_dir.glob("results/*/*_evaluation.json")):
        try:
            rows.append(Evaluation.model_validate_json(path.read_text(encoding="utf-8")).model_dump())
        except (OSError, ValueError, ValidationError):
            logging.exception("Invalid result artifact: %s", path.name)
            warnings.append(f"Could not read result {path.name}; rerun evaluation.")
    for path in sorted(settings.artifacts_dir.glob("reports/*/dataset_summary.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            # Serve only summary data, never arbitrary local image paths.
            datasets.append({k: value[k] for k in ["run_id", "purpose", "counts", "classes", "corrupt", "external_excluded"]})
        except (OSError, ValueError, KeyError):
            warnings.append("A dataset summary is invalid; rerun prepare_data.")
    return {"results": rows, "warnings": warnings, "datasets": datasets}

def plot_paths(settings):
    root = (settings.artifacts_dir / "plots").resolve()
    return {f"{p.parent.name}/{p.name}": p for p in root.glob("*/*.png") if p.resolve().is_relative_to(root)}
