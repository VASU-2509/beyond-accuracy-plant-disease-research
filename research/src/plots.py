"""All figures originate in saved observations; PNG and vector PDF exports."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 220})

def finish(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

def distribution(counts, path):
    fig, ax = plt.subplots(figsize=(10, max(4, len(counts)*.24)))
    ax.barh(list(counts), list(counts.values()), color="#168273")
    ax.set(xlabel="Decoded images", title="Observed dataset class distribution")
    finish(fig, path)

def plot_result(result, history, output):
    output = Path(output)
    name = result["model_name"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for split in ["train", "val"]:
        for ax, metric in zip(axes, ["loss", "accuracy"]):
            ax.plot([r["epoch"] for r in history], [r[f"{split}_{metric}"] for r in history], marker="o", label=split)
            ax.set(xlabel="Epoch", ylabel=metric)
            ax.legend()
    finish(fig, output / f"{name}_training.png")
    for domain, data in result["domains"].items():
        prefix = f"{name}_{domain}"
        fig, ax = plt.subplots(figsize=(max(7, len(result["classes"])*.3), max(6, len(result["classes"])*.3)))
        image = ax.imshow(data["classification"]["confusion_matrix"], cmap="Blues")
        ax.set(xticks=range(len(result["classes"])), yticks=range(len(result["classes"])), xticklabels=result["classes"], yticklabels=result["classes"], xlabel="Predicted", ylabel="True", title=domain)
        ax.tick_params(axis="x", labelrotation=90, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        fig.colorbar(image, ax=ax)
        finish(fig, output / f"{prefix}_confusion.png")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
        for kind in ["raw", "calibrated"]:
            points = [b for b in data[kind]["bins"] if b["count"]]
            ax.plot([b["confidence"] for b in points], [b["accuracy"] for b in points], "o-", label=kind)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence", ylabel="Empirical accuracy", title=f"{domain}: reliability")
        ax.legend()
        finish(fig, output / f"{prefix}_reliability.png")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, kind in zip(axes, ["raw", "calibrated"]):
            bins = data[kind]["confidence_histogram"]
            for outcome, color in [("correct", "#168273"), ("incorrect", "#d98745")]:
                ax.step([b["upper"] for b in bins], [b[outcome] for b in bins], label=outcome, color=color)
            ax.set(title=kind, xlabel="Confidence", ylabel="Count")
            ax.legend()
        finish(fig, output / f"{prefix}_confidence.png")
        fig, ax = plt.subplots(figsize=(5, 4))
        curve = [p for p in data["risk_curve"] if p["risk"] is not None]
        ax.plot([p["coverage"] for p in curve], [p["risk"] for p in curve], color="#168273")
        ax.set(xlabel="Coverage", ylabel="Selective risk", xlim=(0, 1), ylim=(0, 1))
        finish(fig, output / f"{prefix}_risk.png")

def comparison(results, output):
    if not results:
        return
    output = Path(output)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ["accuracy", "macro_f1"]):
        for i, result in enumerate(results):
            for j, domain in enumerate(["in_domain_common", "external"]):
                if domain in result["domains"]:
                    ax.bar(i*3+j, result["domains"][domain]["classification"][metric], label=f"{result['model_name']} {domain}")
        ax.set(ylabel=metric, ylim=(0, 1), xticks=[])
        ax.legend(fontsize=7)
    finish(fig, output / "model_comparison.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, label in zip(axes, ["checkpoint_mb", "latency_ms_median"], ["Checkpoint size (MiB)", "Forward latency (ms)"]):
        ax.bar([r["model_name"] for r in results], [r["efficiency"][key] for r in results], color="#168273")
        ax.set(ylabel=label)
    finish(fig, output / "efficiency_comparison.png")
