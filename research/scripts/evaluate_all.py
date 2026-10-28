import argparse
import csv
from pathlib import Path
from research.src.utils import load_config
from research.src.evaluate import evaluate
from research.src.plots import comparison

def evaluate_all(cfg):
    results = [evaluate(cfg, name) for name in cfg["models"]]
    output = Path(cfg["artifacts_dir"])
    comparison(results, output / "plots" / cfg["run_id"])
    with (output / "results" / cfg["run_id"] / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "purpose", "domain", "n", "accuracy", "macro_f1", "raw_ece", "calibrated_ece", "raw_brier", "calibrated_brier", "coverage", "risk", "parameters", "checkpoint_mib", "latency_ms"])
        for result in results:
            for domain, data in result["domains"].items():
                writer.writerow([result["model_name"], result["purpose"], domain, data["classification"]["n"], data["classification"]["accuracy"], data["classification"]["macro_f1"], data["raw"]["ece"], data["calibrated"]["ece"], data["raw"]["brier"], data["calibrated"]["brier"], data["selective"]["coverage"], data["selective"]["risk"], result["efficiency"]["parameters"], result["efficiency"]["checkpoint_mb"], result["efficiency"]["latency_ms_median"]])
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="research/configs/quick_test.yaml")
    evaluate_all(load_config(parser.parse_args().config))

if __name__ == "__main__":
    main()
