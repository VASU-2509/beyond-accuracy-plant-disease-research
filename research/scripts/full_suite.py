"""Prespecified sequential 2-model x 3-seed study. No download or training by default."""
import argparse
import contextlib
import copy
import shutil
import statistics
import sys
from pathlib import Path
from research.storage import configure_runtime, storage_root
from research.src.utils import ROOT, load_config, save_json, read_json, digest, environment
from research.src.manifest_validation import validate_manifest, require_research_manifest

SEEDS = (42, 43, 44)


def configs(config_path):
    base = load_config(config_path)
    if base.get("purpose") != "research":
        raise ValueError("Full suite accepts research configurations only")
    prefix = base.get("study_id", "full")
    if not str(prefix).replace("-", "").replace("_", "").isalnum():
        raise ValueError("study_id must be alphanumeric with hyphens/underscores")
    return [dict(copy.deepcopy(base), seed=seed, split_seed=42, run_id=f"{prefix}_seed{seed}") for seed in SEEDS]


class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, value):
        for stream in self.streams: stream.write(value); stream.flush()
        return len(value)
    def flush(self):
        for stream in self.streams: stream.flush()


def aggregate(configurations):
    rows, membership = [], None
    for cfg in configurations:
        manifest_path = Path(cfg["data_dir"]) / "splits" / cfg["run_id"] / "manifest.json"
        manifest = read_json(manifest_path)
        current = {s: [(r["path"], r["label"], r["sha256"]) for r in manifest[s]] for s in ["train", "val", "test", "external"]}
        if membership is not None and current != membership:
            raise ValueError("Seed runs must share identical split membership")
        membership = current
        for name in cfg["models"]:
            result = read_json(Path(cfg["artifacts_dir"]) / "results" / cfg["run_id"] / f"{name}_evaluation.json")
            checkpoint = Path(cfg["artifacts_dir"]) / "models" / cfg["run_id"] / f"{name}.pt"
            if (result["purpose"] != "research" or not result["pretrained_loaded"] or result["config"] != cfg
                    or result["manifest_sha256"] != digest(manifest_path) or result["checkpoint_sha256"] != digest(checkpoint)):
                raise ValueError("Results do not match the prespecified configuration and evidence")
            rows.append(result)
    metrics = []
    for name in configurations[0]["models"]:
        selected = [r for r in rows if r["model_name"] == name]
        for domain in ["in_domain", "in_domain_common", "external"]:
            for category, keys in {"classification": ["accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1"],
                                   "raw": ["ece", "brier"], "calibrated": ["ece", "brier"],
                                   "selective": ["coverage", "risk", "accuracy"]}.items():
                for key in keys:
                    per_seed = {str(r["config"]["seed"]): r["domains"][domain][category][key] for r in selected}
                    values = [v for v in per_seed.values() if v is not None]
                    metrics.append({"model": name, "domain": domain, "metric": f"{category}.{key}", "per_seed": per_seed,
                                    "n_defined": len(values), "mean": statistics.mean(values) if values else None,
                                    "sample_sd": statistics.stdev(values) if len(values) > 1 else None})
        for key in ["parameters", "checkpoint_mb", "latency_ms_median", "latency_ms_p95"]:
            per_seed = {str(r["config"]["seed"]): r["efficiency"][key] for r in selected}
            values = list(per_seed.values())
            metrics.append({"model": name, "domain": "efficiency", "metric": key, "per_seed": per_seed,
                            "n_defined": len(values), "mean": statistics.mean(values),
                            "sample_sd": statistics.stdev(values) if len(values) > 1 else None})
    summary = {"purpose": "research", "seeds": list(SEEDS), "split_seed": 42, "metrics": metrics,
               "interpretation": "Descriptive mean and sample SD across training seeds on fixed splits; no significance test or population confidence interval; null metrics stay undefined",
               "runs": [{"id": r["id"], "manifest_sha256": r["manifest_sha256"], "checkpoint_sha256": r["checkpoint_sha256"]} for r in rows]}
    output = Path(configurations[0]["artifacts_dir"]) / "aggregate"
    if (output / "metrics.json").exists():
        raise FileExistsError("Preserving existing aggregate metrics")
    save_json(output / "metrics.json", summary)
    import csv
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader(); writer.writerows(metrics)


def clone_prepared_run(source_cfg, cfg):
    if source_cfg["split_seed"] != cfg["split_seed"]:
        raise ValueError("Cannot reuse a prepared split with a different split seed")
    source_manifest_path = Path(source_cfg["data_dir"]) / "splits" / source_cfg["run_id"] / "manifest.json"
    target_data = Path(cfg["data_dir"])
    target_output = Path(cfg["artifacts_dir"])
    target_manifest_path = target_data / "splits" / cfg["run_id"] / "manifest.json"
    target_files = [
        target_manifest_path,
        target_data / "processed" / cfg["run_id"] / "dataset_summary.json",
        target_output / "reports" / cfg["run_id"] / "dataset_summary.json",
        target_output / "reports" / cfg["run_id"] / "plantvillage_audit.json",
        target_output / "reports" / cfg["run_id"] / "plantdoc_audit.json",
        target_output / "reports" / cfg["run_id"] / "preparation_review.json",
        target_output / "plots" / cfg["run_id"] / "dataset_distribution.png",
    ]
    if any(path.exists() for path in target_files):
        raise FileExistsError(f"Preserving existing prepared outputs for {cfg['run_id']}")

    manifest = copy.deepcopy(read_json(source_manifest_path))
    if not manifest.get("full_inventory_verified") or manifest.get("split_seed") != cfg["split_seed"]:
        raise ValueError("Source preparation is not a verified match for the requested fixed split")
    manifest["run_id"] = cfg["run_id"]
    manifest["seed"] = cfg["seed"]
    manifest["split_seed"] = cfg["split_seed"]
    require_research_manifest(manifest, cfg)
    manifest["leakage_check"] = validate_manifest(manifest, cfg["data_dir"])

    source_data = Path(source_cfg["data_dir"])
    source_output = Path(source_cfg["artifacts_dir"])
    summary = read_json(source_data / "processed" / source_cfg["run_id"] / "dataset_summary.json")
    summary["run_id"] = cfg["run_id"]
    review = read_json(source_output / "reports" / source_cfg["run_id"] / "preparation_review.json")
    review["leakage_check"] = manifest["leakage_check"]
    for name in ["dataset_summary.json", "plantvillage_audit.json", "plantdoc_audit.json", "preparation_review.json"]:
        source = source_output / "reports" / source_cfg["run_id"] / name
        destination = target_output / "reports" / cfg["run_id"] / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if name == "dataset_summary.json":
            save_json(destination, summary)
        elif name == "preparation_review.json":
            save_json(destination, review)
        else:
            shutil.copy2(source, destination)
    save_json(target_data / "processed" / cfg["run_id"] / "dataset_summary.json", summary)
    source_plot = source_output / "plots" / source_cfg["run_id"] / "dataset_distribution.png"
    target_plot = target_output / "plots" / cfg["run_id"] / "dataset_distribution.png"
    target_plot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_plot, target_plot)
    save_json(target_manifest_path, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["plan", "prepare", "train", "evaluate", "aggregate"], nargs="?", default="plan")
    parser.add_argument("--config", default="research/configs/full_experiment.yaml")
    args = parser.parse_args()
    cfgs = configs(args.config)
    if args.stage == "plan":
        for cfg in cfgs: print(f"{cfg['run_id']}: {cfg['models']}, seed={cfg['seed']}, split_seed=42, up to {cfg['epochs']} epochs, {cfg['device']}, outputs={cfg['artifacts_dir']}")
        return
    configure_runtime(create=True)
    output = Path(cfgs[0]["artifacts_dir"])
    for cfg in cfgs:
        saved = output / "reports" / cfg["run_id"] / "resolved_config.json"
        if saved.exists() and read_json(saved) != cfg:
            raise ValueError("Saved configuration differs; use a separate experiment storage root")
        save_json(saved, cfg)
    if args.stage == "aggregate":
        aggregate(cfgs); return
    # Sequential stages and models bound RAM/VRAM; failures stop the suite immediately.
    for cfg in cfgs:
        log_dir = output / "logs" / cfg["run_id"]
        log_dir.mkdir(parents=True, exist_ok=True)
        with (log_dir / f"{args.stage}.log").open("a", encoding="utf-8") as log:
            with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
                try:
                    if args.stage == "prepare":
                        from research.scripts.prepare_data import prepare
                        if cfg["run_id"] == cfgs[0]["run_id"]:
                            prepare(cfg)
                        else:
                            clone_prepared_run(cfgs[0], cfg)
                    elif args.stage == "train":
                        # Require verified local pretrained weights; this stage never downloads them.
                        record = read_json(storage_root() / "preflight/pretrained_weights.json")
                        from research.scripts.preflight import inspect
                        for name, info in inspect()["weights"].items():
                            if not info["cached"] or digest(info["path"]) != record[name]["sha256"]:
                                raise ValueError("Run cache_weights successfully before training")
                        from research.src.train import train
                        for name in cfg["models"]: train(cfg, name)
                    else:
                        from research.scripts.evaluate_all import evaluate_all
                        evaluate_all(cfg)
                    save_json(log_dir / f"{args.stage}_status.json", {"status": "completed", "hardware": environment()})
                except Exception:
                    import traceback
                    traceback.print_exc()
                    save_json(log_dir / f"{args.stage}_status.json", {"status": "failed", "hardware": environment()})
                    raise


if __name__ == "__main__":
    main()
