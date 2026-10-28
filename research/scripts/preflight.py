"""Inspect storage/hardware; optional short synthetic throughput timing, never full training."""
import argparse
import json
import shutil
import time
from pathlib import Path
import torch
from torchvision import models
from research.storage import configure_runtime, paths, storage_root
from research.src.utils import environment, save_json, seed_everything
from research.src.model_factory import build_model


def inspect(benchmark=False):
    configure_runtime(create=True)
    root = storage_root()
    result = {"hardware": environment(), "storage_root": str(root),
              "free_gib": shutil.disk_usage(root).free / 1024**3,
              "paths": {k: str(v) for k, v in paths().items()},
              "caches": configure_runtime(), "weights": {}, "timing": {}}
    for name, weights in [("resnet18", models.ResNet18_Weights.IMAGENET1K_V1),
                          ("mobilenet_v3_small", models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)]:
        file = Path(torch.hub.get_dir()) / "checkpoints" / weights.url.rsplit("/", 1)[-1]
        result["weights"][name] = {"enum": str(weights), "url": weights.url, "cached": file.exists(), "path": str(file)}
        if benchmark:
            seed_everything(42)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = build_model(name, 38).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
            x = torch.rand(8, 3, 224, 224, device=device)
            y = torch.arange(8, device=device) % 38
            def sync():
                if device.type == "cuda": torch.cuda.synchronize()
            train_times, eval_times = [], []
            for step in range(10):
                sync(); start = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                torch.nn.functional.cross_entropy(model(x), y).backward()
                optimizer.step(); sync()
                if step >= 2: train_times.append(time.perf_counter()-start)
            model.eval()
            with torch.inference_mode():
                for step in range(10):
                    sync(); start = time.perf_counter(); model(x); sync()
                    if step >= 2: eval_times.append(time.perf_counter()-start)
            train_s, eval_s = sum(train_times)/8, sum(eval_times)/8
            epoch_s = (54305*.70/8)*train_s + (54305*.15/8)*eval_s
            result["timing"][name] = {"device": str(device), "batch_size": 8, "image_size": 224,
                "train_batch_seconds": train_s, "eval_batch_seconds": eval_s,
                "estimated_epoch_hours_compute_only": epoch_s/3600,
                "estimated_30_epoch_hours_compute_only": epoch_s*30/3600,
                "protocol": "Random tensors/weights; 2 warmups, 8 timed steps; no downloads/checkpoints/accuracy measurements; excludes decode, augmentation, disk, sustained thermal throttling"}
            del optimizer, model, x, y
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    result = inspect(args.benchmark)
    save_json(storage_root() / "preflight" / ("hardware_benchmark.json" if args.benchmark else "hardware.json"), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
