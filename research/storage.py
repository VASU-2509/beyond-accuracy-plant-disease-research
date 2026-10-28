"""One storage root for research data, outputs, downloads and runtime caches."""
import json
import os
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def storage_root():
    settings = json.loads((PROJECT_ROOT / "research/configs/storage.json").read_text())
    value = os.environ.get("RESEARCH_STORAGE_ROOT", settings["root"])
    root = Path(value).expanduser()
    if not root.is_absolute():
        raise ValueError("RESEARCH_STORAGE_ROOT must be an absolute path")
    return root.resolve()


def managed_path(path):
    path = Path(path).expanduser().resolve()
    root = storage_root()
    if not path.is_relative_to(root):
        raise ValueError(f"Research storage must stay under {root}: {path}. Change RESEARCH_STORAGE_ROOT to relocate all storage.")
    if os.name == "nt" and path.drive.lower() == "c:":
        raise ValueError("Research data, outputs and caches on C: are disabled; choose another drive")
    return path


def paths(purpose="research"):
    root = storage_root()
    group = "full" if purpose == "research" else "smoke"
    return {
        "data_dir": managed_path(os.environ.get("DATA_DIR", root / group / "data")),
        "artifacts_dir": managed_path(os.environ.get("ARTIFACTS_DIR", root / group / "artifacts")),
    }


def configure_runtime(create=False):
    root = managed_path(storage_root())
    locations = {
        "TORCH_HOME": root / "cache/torch",
        "HF_HOME": root / "cache/huggingface",
        "HF_HUB_CACHE": root / "cache/huggingface/hub",
        "HF_DATASETS_CACHE": root / "cache/huggingface/datasets",
        "HF_XET_CACHE": root / "cache/huggingface/xet",
        "XDG_CACHE_HOME": root / "cache",
        "MPLCONFIGDIR": root / "cache/matplotlib",
        "PIP_CACHE_DIR": root / "cache/pip",
        "TEMP": root / "tmp", "TMP": root / "tmp", "TMPDIR": root / "tmp",
    }
    # Override inherited per-library locations so they cannot silently target C:.
    for key, path in locations.items():
        os.environ[key] = str(path)
        if create:
            path.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    tempfile.tempdir = str(locations["TEMP"])
    return {key: str(value) for key, value in locations.items()}
