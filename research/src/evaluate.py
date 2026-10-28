import csv
import time
from pathlib import Path
import numpy as np
import torch

from .calibration import fit_temperature, probabilities
from .data_loader import make_loader
from .metrics import classification, reliability
from .model_factory import load_checkpoint
from .selective_prediction import select_threshold, selective_metrics, risk_curve
from .utils import read_json, save_json, digest, device_for, seed_everything, environment
from .plots import plot_result
from .manifest_validation import validate_manifest, require_research_manifest

@torch.inference_mode()
def collect_logits(model, loader, device):
    logits, labels = [], []
    for images, y in loader:
        logits.append(model(images.to(device)).cpu().numpy())
        labels.append(y.numpy())
    if not logits:
        raise ValueError("Empty evaluation dataset")
    return np.concatenate(logits), np.concatenate(labels)

@torch.inference_mode()
def efficiency(model, image_size, device, checkpoint_path):
    sample = torch.zeros(1, 3, image_size, image_size, device=device)
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()
    for _ in range(5):
        model(sample)
    sync()
    latencies = []
    for _ in range(30):
        sync()
        start = time.perf_counter()
        model(sample)
        sync()
        latencies.append((time.perf_counter()-start)*1000)
    return {"parameters": sum(p.numel() for p in model.parameters()), "checkpoint_mb": checkpoint_path.stat().st_size/1024**2,
            "latency_ms_median": float(np.median(latencies)), "latency_ms_p95": float(np.percentile(latencies, 95)),
            "device": str(device), "protocol": "batch=1; 5 warmups; 30 forwards; synchronized CUDA; excludes decode, preprocessing and HTTP; 2 CPU threads", "hardware": environment()}

def evaluate(cfg, name):
    seed_everything(cfg["seed"])
    output = Path(cfg["artifacts_dir"])
    checkpoint_path = output / "models" / cfg["run_id"] / f"{name}.pt"
    manifest_path = Path(cfg["data_dir"]) / "splits" / cfg["run_id"] / "manifest.json"
    manifest = read_json(manifest_path)
    require_research_manifest(manifest, cfg)
    validate_manifest(manifest, cfg["data_dir"], verify_files=cfg.get("purpose") == "research")
    if cfg.get("purpose") == "research" and (output / "results" / cfg["run_id"] / f"{name}_evaluation.json").exists():
        raise FileExistsError("Preserving existing final evaluation; choose a new run_id")
    device = device_for(cfg.get("device", "auto"))
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    if cfg.get("purpose") == "research" and cfg != checkpoint.get("config"):
        raise ValueError("Evaluation configuration must match the frozen training configuration")
    if checkpoint["manifest_sha256"] != digest(manifest_path):
        raise ValueError("Split manifest changed since training; use the original manifest or retrain")
    if checkpoint["image_size"] != cfg["image_size"]:
        raise ValueError("Image size must match training")
    classes = checkpoint["classes"]
    result_dir = output / "results" / cfg["run_id"]
    result_dir.mkdir(parents=True, exist_ok=True)
    validation_logits, validation_labels = collect_logits(model, make_loader(manifest["val"], classes, cfg), device)
    calibration = fit_temperature(validation_logits, validation_labels)
    threshold = select_threshold(probabilities(validation_logits, calibration["temperature"]), validation_labels,
                                 cfg.get("target_risk", .1), cfg.get("min_coverage", .2))
    calibration_record = {**calibration, "selection": threshold, "checkpoint_sha256": digest(checkpoint_path), "manifest_sha256": digest(manifest_path)}
    save_json(checkpoint_path.with_suffix(".calibration.json"), calibration_record)
    np.savez_compressed(result_dir / f"{name}_validation_logits.npz", logits=validation_logits, labels=validation_labels)
    domains = {}
    datasets = {"in_domain": manifest["test"]}
    common_classes = sorted({r["label"] for r in manifest["external"]})
    if manifest["external"]:
        datasets["external"] = manifest["external"]
        datasets["in_domain_common"] = [r for r in manifest["test"] if r["label"] in common_classes]
    for domain, rows in datasets.items():
        logits, labels = collect_logits(model, make_loader(rows, classes, cfg), device)
        raw, calibrated = probabilities(logits), probabilities(logits, calibration["temperature"])
        metric_labels = [classes.index(c) for c in common_classes] if domain != "in_domain" else list(range(len(classes)))
        domains[domain] = {"classification": classification(raw, labels, classes, metric_labels),
                           "raw": reliability(raw, labels), "calibrated": reliability(calibrated, labels),
                           "selective": selective_metrics(calibrated, labels, threshold["threshold"]),
                           "risk_curve": risk_curve(calibrated, labels)}
        np.savez_compressed(result_dir / f"{name}_{domain}_logits.npz", logits=logits, labels=labels)
        with (result_dir / f"{name}_{domain}_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["path", "true_class", "predicted_class", "raw_confidence", "calibrated_confidence", "accepted"])
            for row, label, p, q in zip(rows, labels, raw, calibrated):
                writer.writerow([row["path"], classes[label], classes[p.argmax()], float(p.max()), float(q.max()), bool(q.max() >= threshold["threshold"])])
    gaps = {metric: domains["in_domain_common"]["classification"][metric]-domains["external"]["classification"][metric]
            for metric in ["accuracy", "macro_f1"]} if "external" in domains else None
    result = {"schema_version": 1, "id": f"{cfg['run_id']}__{name}", "run_id": cfg["run_id"], "model_name": name,
              "purpose": checkpoint["purpose"], "classes": classes, "config": cfg, "pretrained_loaded": checkpoint.get("pretrained_loaded", False),
              "manifest_sha256": digest(manifest_path), "checkpoint_sha256": digest(checkpoint_path),
              "domains": domains, "calibration": calibration_record, "domain_gap": gaps,
              "efficiency": efficiency(model, cfg["image_size"], device, checkpoint_path),
              "external_status": "completed" if manifest["external"] else "pending: no eligible external images",
              "protocol": "Full training-class softmax and argmax retained. Common-class ground-truth subset used for comparable gaps; no renormalization. Macro averages over stated metric_classes.",
              "mapping": manifest["mapping"], "external_excluded": len(manifest["excluded_external"])}
    training = read_json(result_dir / f"{name}_training.json")
    plot_result(result, training["history"], output / "plots" / cfg["run_id"])
    save_json(result_dir / f"{name}_evaluation.json", result)
    print(f"Evaluated {result['id']} ({result['purpose']}); external: {result['external_status']}")
    return result
