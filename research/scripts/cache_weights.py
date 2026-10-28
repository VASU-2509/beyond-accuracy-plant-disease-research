"""Explicitly fetch and verify ImageNet weights before scheduling training."""
from pathlib import Path
from torchvision import models
import torch
from research.storage import configure_runtime, storage_root
from research.src.utils import save_json, digest


def main():
    configure_runtime(create=True)
    records = {}
    for name, weights in [("resnet18", models.ResNet18_Weights.IMAGENET1K_V1),
                          ("mobilenet_v3_small", models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)]:
        print(f"Fetching/verifying {name}: {weights}", flush=True)
        # Hash checked by torchvision before accepting a newly downloaded file.
        weights.get_state_dict(progress=True, check_hash=True)
        file = Path(torch.hub.get_dir()) / "checkpoints" / weights.url.rsplit("/", 1)[-1]
        checksum = digest(file)
        expected = file.stem.rsplit("-", 1)[-1]
        if not checksum.startswith(expected):
            raise ValueError(f"Cached pretrained checksum mismatch: {file}")
        records[name] = {"weights": str(weights), "url": weights.url, "sha256": checksum, "bytes": file.stat().st_size}
    save_json(storage_root() / "preflight/pretrained_weights.json", records)


if __name__ == "__main__":
    main()
