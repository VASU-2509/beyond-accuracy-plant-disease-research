import time
from pathlib import Path
import torch
from torch import nn

from .data_loader import make_loader
from .model_factory import build_model
from .utils import read_json, save_json, seed_everything, device_for, environment, digest
from .manifest_validation import validate_manifest, require_research_manifest

def train(cfg, name):
    seed_everything(cfg["seed"])
    manifest_path = Path(cfg["data_dir"]) / "splits" / cfg["run_id"] / "manifest.json"
    manifest = read_json(manifest_path)
    require_research_manifest(manifest, cfg)
    validate_manifest(manifest, cfg["data_dir"], verify_files=cfg.get("purpose") == "research")
    checkpoint_path = Path(cfg["artifacts_dir"]) / "models" / cfg["run_id"] / f"{name}.pt"
    if cfg.get("purpose") == "research" and checkpoint_path.exists():
        raise FileExistsError("Preserving existing research checkpoint; choose a new run_id")
    if cfg.get("purpose") == "research" and (not cfg["pretrained"] or cfg.get("allow_random_fallback")):
        raise ValueError("Final research requires pretrained weights without random fallback")
    classes = manifest["classes"]
    if manifest["seed"] != cfg["seed"] or manifest["max_groups_per_class"] != cfg.get("max_groups_per_class"):
        raise ValueError("Configuration and prepared split differ; prepare data again")
    if manifest.get("split_seed", manifest["seed"]) != cfg.get("split_seed", cfg["seed"]):
        raise ValueError("Configuration and prepared split seed differ")
    device = device_for(cfg.get("device", "auto"))
    model = build_model(name, len(classes), cfg["pretrained"], cfg["mode"], cfg.get("allow_random_fallback", False)).to(device)
    if cfg.get("purpose") == "research" and not model.pretrained_loaded:
        raise ValueError("Pretrained initialization was not loaded")
    loaders = {s: make_loader(manifest[s], classes, cfg, s == "train") for s in ["train", "val"]}
    if len(loaders["train"]) == 0:
        raise ValueError("Training split too small")
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=.5)
    criterion = nn.CrossEntropyLoss()
    output = Path(cfg["artifacts_dir"])
    checkpoint_path = output / "models" / cfg["run_id"] / f"{name}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history, best, stale = [], float("inf"), 0
    start = time.perf_counter()
    metadata = {"model_name": name, "classes": classes, "image_size": cfg["image_size"], "run_id": cfg["run_id"],
                "purpose": cfg.get("purpose", "research"), "pretrained_loaded": model.pretrained_loaded,
                "pretrained_weights": model.pretrained_weights,
                "manifest_sha256": digest(manifest_path), "config": cfg, "environment": environment()}
    for epoch in range(cfg["epochs"]):
        row = {"epoch": epoch + 1, "learning_rate": optimizer.param_groups[0]["lr"]}
        for split, loader in loaders.items():
            training = split == "train"
            model.train(training)
            if training and cfg["mode"] == "feature_extraction":
                # Freeze running means/variances as well as gradients in the backbone.
                model.eval()
                (model.fc if name == "resnet18" else model.classifier[-1]).train()
            total_loss = correct = count = 0
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(training):
                    logits = model(images)
                    loss = criterion(logits, labels)
                    if not torch.isfinite(loss):
                        raise ValueError("Non-finite training loss")
                    if training:
                        loss.backward()
                        optimizer.step()
                count += len(labels)
                total_loss += float(loss.detach()) * len(labels)
                correct += int((logits.argmax(1) == labels).sum())
            row[f"{split}_loss"] = total_loss / count
            row[f"{split}_accuracy"] = correct / count
        scheduler.step(row["val_loss"])
        history.append(row)
        print(f"{name} epoch {epoch+1}: train={row['train_loss']:.4f}, val={row['val_loss']:.4f}", flush=True)
        if row["val_loss"] < best:
            best, stale = row["val_loss"], 0
            temporary = checkpoint_path.with_suffix(".partial")
            torch.save({**metadata, "state_dict": model.state_dict(), "best_epoch": epoch+1}, temporary)
            temporary.replace(checkpoint_path)
        else:
            stale += 1
        save_json(output / "results" / cfg["run_id"] / f"{name}_training.json", {**metadata, "history": history, "seconds": time.perf_counter()-start})
        if stale >= cfg.get("patience", cfg["epochs"]):
            break
    return checkpoint_path
