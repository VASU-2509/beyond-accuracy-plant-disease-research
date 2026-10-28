"""Configuration, deterministic execution, provenance and atomic JSON writes."""
import hashlib
import importlib.metadata
import json
import os
import platform
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from research.storage import paths, managed_path

ROOT = Path(__file__).resolve().parents[2]

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def digest(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def git_blob_digest(path):
    body = Path(path).read_bytes()
    return hashlib.sha1(f"blob {len(body)}\0".encode() + body).hexdigest()

def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {"seed", "models", "epochs", "batch_size", "image_size", "learning_rate", "pretrained", "mode", "run_id"}
    if not isinstance(cfg, dict) or not required <= cfg.keys():
        raise ValueError(f"Configuration requires {sorted(required)}")
    if cfg["mode"] not in {"feature_extraction", "fine_tuning"}:
        raise ValueError("Invalid training mode")
    if any(cfg[k] <= 0 for k in ["epochs", "batch_size", "image_size", "learning_rate"]):
        raise ValueError("Training dimensions and rates must be positive")
    if cfg["batch_size"] < 2 or cfg["image_size"] < 32:
        raise ValueError("Batch size >=2 and image size >=32 required")
    if not cfg["models"] or set(cfg["models"]) - {"resnet18", "mobilenet_v3_small"}:
        raise ValueError("Unsupported model")
    if not str(cfg["run_id"]).replace("-", "").replace("_", "").isalnum():
        raise ValueError("run_id must be alphanumeric with hyphens/underscores")
    defaults = paths(cfg.get("purpose", "research"))
    for key, env in [("data_dir", "DATA_DIR"), ("artifacts_dir", "ARTIFACTS_DIR")]:
        cfg[key] = str(managed_path(os.getenv(env, cfg.get(key, defaults[key]))))
    if cfg.get("purpose") == "research":
        if not cfg["pretrained"] or cfg.get("allow_random_fallback"):
            raise ValueError("Final research requires pretrained weights and forbids random fallback")
        if cfg.get("max_groups_per_class") is not None:
            raise ValueError("Final research cannot cap groups per class")
    return cfg

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def device_for(name="auto"):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; use device: cpu")
    return torch.device(name)

def environment():
    import psutil
    return {"python": platform.python_version(), "platform": platform.platform(),
            "cpu": platform.processor(), "ram_bytes": psutil.virtual_memory().total,
            "cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "packages": {p: importlib.metadata.version(p) for p in ["torch", "torchvision", "numpy", "scipy", "scikit-learn", "pillow", "fastapi"]}}
