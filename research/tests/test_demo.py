from PIL import Image
import pytest
import torch

from research.scripts.prepare_demo import create_demo_manifest, select_rows
from research.src.predict import predict_image


def rows(split, labels, count):
    return [
        {
            "path": f"{split}/{label}/{index}.jpg",
            "label": label,
            "sha256": f"{split}-{label}-{index}",
            "pixel_hash": f"pixel-{split}-{label}-{index}",
            "dhash": f"dhash-{split}-{label}-{index}",
            "leaf_id": f"leaf-{split}-{label}-{index}",
        }
        for label in labels
        for index in range(count)
    ]


def test_demo_sampling_is_fixed_balanced_and_keeps_external_separate():
    classes = ["a", "b"]
    source = {
        "run_id": "full_seed42",
        "seed": 42,
        "split_seed": 42,
        "classes": classes,
        "train": rows("train", classes, 8),
        "val": rows("val", classes, 8),
        "test": rows("test", classes, 8),
        "external": rows("external", classes, 5),
        "sources": {},
        "mapping": {},
        "excluded_external": [],
    }
    cfg = {
        "run_id": "demo",
        "seed": 2026,
        "train_per_class": 4,
        "validation_per_class": 3,
        "test_per_class": 2,
        "external_per_class": 1,
    }

    manifest = create_demo_manifest(source, cfg)
    repeated = create_demo_manifest(source, cfg)

    assert manifest == repeated
    assert {split: len(manifest[split]) for split in ("train", "val", "test", "external")} == {
        "train": 8, "val": 6, "test": 4, "external": 2
    }
    assert all(
        set(row["label"] for row in manifest[split]) == set(classes)
        for split in ("train", "val", "test", "external")
    )
    assert not (
        {row["path"] for row in manifest["test"]}
        & {row["path"] for row in manifest["external"]}
    )


def test_demo_external_sampling_allows_small_classes():
    sampled = select_rows(
        rows("external", ["a", "b"], 2), ["a", "b"], 5, 2026, "external", allow_short=True
    )
    assert len(sampled) == 4
    assert [row["label"] for row in sampled] == ["a", "a", "b", "b"]


@pytest.mark.parametrize(
    ("threshold", "constraint_satisfied"),
    [(1.01, False), (0.0, True)],
)
def test_uninformative_validation_threshold_does_not_warn(threshold, constraint_satisfied):
    class ConstantModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.tensor([1.0, 0.0]))

        def forward(self, image):
            return self.logits.expand(image.shape[0], -1)

    result = predict_image(
        ConstantModel(),
        {
            "image_size": 32,
            "classes": ["a", "b"],
            "run_id": "demo",
            "model_name": "resnet18",
            "purpose": "demo",
        },
        Image.new("RGB", (16, 16), "green"),
        {"temperature": 1.0, "selection": {"threshold": threshold, "constraint_satisfied": constraint_satisfied}},
    )

    assert result["calibrated_confidence"] is not None
    assert result["threshold"] is None
    assert result["abstained"] is None
