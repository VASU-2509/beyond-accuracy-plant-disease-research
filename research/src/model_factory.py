"""Architecture construction and safe, self-describing checkpoint loading."""
import logging
import torch
from torch import nn
from torchvision import models

MODEL_NAMES = ("resnet18", "mobilenet_v3_small")

def build_model(name, num_classes, pretrained=False, mode="fine_tuning", allow_random_fallback=False):
    if name not in MODEL_NAMES or num_classes < 2:
        raise ValueError("Supported architecture and at least two classes required")
    constructor = getattr(models, name)
    weights = (models.ResNet18_Weights.IMAGENET1K_V1 if name == "resnet18" else models.MobileNet_V3_Small_Weights.IMAGENET1K_V1) if pretrained else None
    initialized = bool(weights)
    try:
        model = constructor(weights=weights)
    except (OSError, RuntimeError) as exc:
        if not allow_random_fallback:
            raise RuntimeError("Pretrained weights unavailable. Cache weights or explicitly set pretrained: false / allow_random_fallback: true.") from exc
        logging.warning("Pretrained download failed: using random initialization: %s", exc)
        model, initialized = constructor(weights=None), False
    if mode == "feature_extraction":
        for parameter in model.parameters():
            parameter.requires_grad = False
    elif mode != "fine_tuning":
        raise ValueError("Unknown training mode")
    if name == "resnet18":
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    model.pretrained_loaded = initialized
    model.pretrained_weights = str(weights) if initialized else None
    return model

def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    required = {"state_dict", "model_name", "classes", "image_size", "run_id", "purpose", "manifest_sha256"}
    if not isinstance(checkpoint, dict) or not required <= checkpoint.keys():
        raise ValueError("Invalid or incompatible checkpoint metadata")
    if len(set(checkpoint["classes"])) != len(checkpoint["classes"]):
        raise ValueError("Duplicate checkpoint classes")
    model = build_model(checkpoint["model_name"], len(checkpoint["classes"]))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), checkpoint
