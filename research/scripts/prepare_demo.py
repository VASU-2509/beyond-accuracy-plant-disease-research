"""Prepare and evaluate a small demo using an existing compatible checkpoint."""
import argparse
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
import yaml

from research.storage import managed_path, storage_root
from research.src.manifest_validation import validate_manifest
from research.src.utils import digest, read_json, save_json
from research.scripts.evaluate_all import evaluate_all

ROOT = Path(__file__).resolve().parents[2]
SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def select_rows(rows, classes, limit, seed, split, allow_short=False):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["label"]].append(row)
    selected = []
    for label in classes:
        candidates = sorted(grouped[label], key=lambda row: row["path"])
        count = min(limit, len(candidates)) if allow_short else limit
        if count == 0:
            continue
        if len(candidates) < count:
            raise ValueError(f"{split}: {label} has {len(candidates)} rows; {count} required")
        rng = random.Random(f"{seed}:{split}:{label}")
        selected.extend(rng.sample(candidates, count))
    return sorted(selected, key=lambda row: row["path"])


def create_demo_manifest(source, cfg):
    classes = source["classes"]
    sampled = {
        "train": select_rows(source["train"], classes, cfg["train_per_class"], cfg["seed"], "train"),
        "val": select_rows(source["val"], classes, cfg["validation_per_class"], cfg["seed"], "val"),
        "test": select_rows(source["test"], classes, cfg["test_per_class"], cfg["seed"], "test"),
        "external": select_rows(
            source["external"], classes, cfg["external_per_class"], cfg["seed"], "external", allow_short=True
        ),
    }
    return {
        **sampled,
        "classes": classes,
        "seed": cfg["seed"],
        "run_id": cfg["run_id"],
        "split_seed": source.get("split_seed", source["seed"]),
        "max_groups_per_class": None,
        "purpose": "demo",
        "sources": source["sources"],
        "mapping": source["mapping"],
        "excluded_training": [],
        "excluded_external": source["excluded_external"],
        "leaf_id_fallback_count": source.get("leaf_id_fallback_count", 0),
        "leaf_id_fallback_counts": source.get("leaf_id_fallback_counts", {}),
        "leaf_group_policy": source.get("leaf_group_policy", "official_with_filename_fallback"),
        "full_inventory_verified": False,
        "split_protocol": (
            "Fixed-seed balanced demo sample drawn separately from the existing full-run "
            "train/validation/test splits; PlantDoc used only for external evaluation."
        ),
        "demo_subset": {
            "source_run_id": source["run_id"],
            "sampling_seed": cfg["seed"],
            "requested_per_class": {
                "train": cfg["train_per_class"],
                "val": cfg["validation_per_class"],
                "test": cfg["test_per_class"],
                "external": cfg["external_per_class"],
            },
            "balanced_source_splits": True,
            "external_split_is_capped_per_class": True,
        },
    }


def load_demo_config(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {
        "run_id", "source_run_id", "model", "seed", "train_per_class",
        "validation_per_class", "test_per_class", "external_per_class",
        "batch_size", "image_size", "device", "target_risk", "min_coverage",
    }
    if not isinstance(cfg, dict) or not required <= cfg.keys():
        raise ValueError(f"Demo configuration requires {sorted(required)}")
    if cfg.get("purpose") != "demo" or cfg["model"] != "resnet18":
        raise ValueError("This demo runner expects purpose 'demo' and model 'resnet18'")
    if not all(isinstance(cfg[key], str) for key in ("run_id", "source_run_id")):
        raise ValueError("Demo and source run IDs must be strings")
    if any(not SAFE_ID.fullmatch(cfg[key]) for key in ("run_id", "source_run_id")):
        raise ValueError("Demo and source run IDs must contain only letters, digits, underscores, and hyphens")
    if cfg["run_id"] == cfg["source_run_id"]:
        raise ValueError("Demo output run ID must differ from its source run ID")
    if any(not isinstance(cfg[key], int) or cfg[key] <= 0 for key in (
        "seed", "train_per_class", "validation_per_class", "test_per_class",
        "external_per_class", "batch_size", "image_size"
    )):
        raise ValueError("Demo sample sizes and image settings must be positive integers")
    if cfg["batch_size"] != 8 or cfg["image_size"] != 224:
        raise ValueError("The demo configuration requires batch_size 8 and image_size 224")
    return cfg


def _write_if_missing(path, value):
    if path.exists():
        if read_json(path) != value:
            raise FileExistsError(f"Refusing to replace existing demo artifact: {path}")
        return
    save_json(path, value)


def prepare_demo(config_path):
    cfg = load_demo_config(config_path)
    root = storage_root()
    data_dir = managed_path(root / "full" / "data")
    source_artifacts = managed_path(root / "full" / "artifacts")
    artifacts_dir = managed_path(root / "demo" / "artifacts")
    source_manifest_path = data_dir / "splits" / cfg["source_run_id"] / "manifest.json"
    source_checkpoint_path = source_artifacts / "models" / cfg["source_run_id"] / f"{cfg['model']}.pt"
    source_training_path = source_artifacts / "results" / cfg["source_run_id"] / f"{cfg['model']}_training.json"
    if not source_manifest_path.is_file() or not source_checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Missing source manifest or checkpoint for {cfg['source_run_id']}; "
            "the full research run is not started by this demo command."
        )

    source = read_json(source_manifest_path)
    if source.get("run_id") != cfg["source_run_id"] or source.get("purpose") != "research":
        raise ValueError("The source manifest is not the expected full research run.")
    source_manifest_hash = digest(source_manifest_path)
    source_checkpoint_hash = digest(source_checkpoint_path)
    checkpoint = torch.load(source_checkpoint_path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("model_name") != cfg["model"]
        or checkpoint.get("run_id") != cfg["source_run_id"]
        or checkpoint.get("manifest_sha256") != source_manifest_hash
        or checkpoint.get("classes") != source["classes"]
        or checkpoint.get("image_size") != cfg["image_size"]
        or not checkpoint.get("pretrained_loaded")
    ):
        raise ValueError("The saved checkpoint is not compatible with this demo configuration.")

    manifest = create_demo_manifest(source, cfg)
    manifest["leakage_check"] = validate_manifest(manifest, data_dir)
    manifest_path = data_dir / "splits" / cfg["run_id"] / "manifest.json"
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise FileExistsError(f"Refusing to replace existing demo manifest: {manifest_path}")
    _write_if_missing(manifest_path, manifest)

    model_path = artifacts_dir / "models" / cfg["run_id"] / f"{cfg['model']}.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    demo_checkpoint = {
        **checkpoint,
        "run_id": cfg["run_id"],
        "purpose": "demo",
        "manifest_sha256": digest(manifest_path),
        "source_run_id": cfg["source_run_id"],
        "source_checkpoint_sha256": source_checkpoint_hash,
        "demo_config": cfg,
    }
    if model_path.exists():
        existing = torch.load(model_path, map_location="cpu", weights_only=True)
        if (
            existing.get("source_checkpoint_sha256") != source_checkpoint_hash
            or existing.get("manifest_sha256") != digest(manifest_path)
        ):
            raise FileExistsError(f"Refusing to replace existing demo checkpoint: {model_path}")
    else:
        temporary = model_path.with_suffix(".partial")
        torch.save(demo_checkpoint, temporary)
        temporary.replace(model_path)

    if not source_training_path.is_file():
        raise FileNotFoundError(f"Missing training history for source checkpoint: {source_training_path}")
    training = read_json(source_training_path)
    demo_training = {
        "run_id": cfg["run_id"],
        "purpose": "model_source",
        "source_run_id": cfg["source_run_id"],
        "source_checkpoint_sha256": source_checkpoint_hash,
        "history": training["history"],
        "seconds": training.get("seconds"),
    }
    training_path = artifacts_dir / "results" / cfg["run_id"] / f"{cfg['model']}_training.json"
    _write_if_missing(training_path, demo_training)

    counts = {
        split: dict(Counter(row["label"] for row in rows))
        for split, rows in manifest.items()
        if split in {"train", "val", "test", "external"}
    }
    summary = {
        "run_id": cfg["run_id"],
        "purpose": "demo",
        "counts": counts,
        "classes": manifest["classes"],
        "corrupt": 0,
        "external_excluded": len(source["excluded_external"]),
        "external_eligible_not_sampled": len(source["external"]) - len(manifest["external"]),
        "source_run_id": cfg["source_run_id"],
        "source_checkpoint_sha256": source_checkpoint_hash,
    }
    _write_if_missing(artifacts_dir / "reports" / cfg["run_id"] / "dataset_summary.json", summary)

    evaluation_path = artifacts_dir / "results" / cfg["run_id"] / f"{cfg['model']}_evaluation.json"
    if evaluation_path.exists():
        result = read_json(evaluation_path)
        if (
            result.get("manifest_sha256") != digest(manifest_path)
            or result.get("source_checkpoint_sha256") != source_checkpoint_hash
        ):
            raise FileExistsError(f"Refusing to replace existing demo evaluation: {evaluation_path}")
        print("Demo artifacts already exist; reusing the saved evaluation.")
        return result

    evaluation_cfg = {
        **cfg,
        "models": [cfg["model"]],
        "mode": "fine_tuning",
        "pretrained": True,
        "allow_random_fallback": False,
        "learning_rate": 0.0001,
        "max_groups_per_class": None,
        "split_seed": manifest["split_seed"],
        "purpose": "demo",
        "data_dir": str(data_dir),
        "artifacts_dir": str(artifacts_dir),
    }
    results = evaluate_all(evaluation_cfg)
    result = results[0]
    result["source_run_id"] = cfg["source_run_id"]
    result["source_checkpoint_sha256"] = source_checkpoint_hash
    save_json(evaluation_path, result)
    print(
        f"Demo ready: {result['model_name']} on {result['domains']['in_domain']['classification']['n']} "
        f"PlantVillage test images; PlantDoc external n="
        f"{result['domains'].get('external', {}).get('classification', {}).get('n', 0)}"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "research" / "configs" / "demo.yaml"))
    args = parser.parse_args()
    prepare_demo(args.config)


if __name__ == "__main__":
    main()
