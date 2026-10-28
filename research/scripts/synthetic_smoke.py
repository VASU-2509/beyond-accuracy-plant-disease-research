"""Offline fallback: generated texture fixtures, explicitly not plant research data."""
from pathlib import Path
import numpy as np
from PIL import Image
from research.src.utils import ROOT, load_config, save_json
from research.scripts.prepare_data import prepare
from research.src.train import train
from research.scripts.evaluate_all import evaluate_all

def main():
    cfg = load_config(ROOT / "research/configs/quick_test.yaml")
    root = Path(cfg["data_dir"]) / "synthetic_smoke"
    rng = np.random.default_rng(42)
    for dataset, labels, n in [("plantvillage", ["Apple___Apple_scab", "Apple___healthy"], 12), ("plantdoc", ["Apple Scab Leaf", "Apple leaf"], 6)]:
        for index, label in enumerate(labels):
            folder = root / "raw" / dataset / label
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                pixels = rng.integers(0, 140, (64,64,3), dtype=np.uint8)
                pixels[:,:,index] += 100
                Image.fromarray(pixels).save(folder / f"fixture_{i}.png")
    save_json(root / "raw" / "sources.json", {"source": "Generated random texture fixtures; NOT PlantVillage or PlantDoc observations", "synthetic": True, "seed":42})
    cfg.update(data_dir=str(root), run_id="synthetic_smoke", purpose="synthetic")
    prepare(cfg)
    for name in cfg["models"]:
        train(cfg, name)
    evaluate_all(cfg)

if __name__ == "__main__":
    main()
