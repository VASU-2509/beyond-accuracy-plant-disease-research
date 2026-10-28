"""Shared deterministic inference used by the CLI and FastAPI."""
import torch
from PIL import ImageOps
from .data_loader import image_transform

@torch.inference_mode()
def predict_image(model, metadata, image, calibration=None, top_k=5):
    image = ImageOps.exif_transpose(image).convert("RGB")
    tensor = image_transform(metadata["image_size"])(image).unsqueeze(0).to(next(model.parameters()).device)
    logits = model(tensor)[0]
    raw = torch.softmax(logits, dim=0)
    temperature = calibration["temperature"] if calibration else 1.0
    calibrated = torch.softmax(logits / temperature, dim=0)
    indices = torch.argsort(calibrated, descending=True)[:min(top_k, len(metadata["classes"]))]
    selection = calibration.get("selection") if calibration else None
    threshold = (
        selection["threshold"]
        if selection and selection.get("constraint_satisfied") and 0 < selection["threshold"] <= 1
        else None
    )
    confidence = float(calibrated.max())
    return {"model_id": f"{metadata['run_id']}__{metadata['model_name']}", "purpose": metadata["purpose"],
            "predicted_class": metadata["classes"][int(indices[0])], "raw_confidence": float(raw.max()),
            "calibrated_confidence": confidence if calibration else None,
            "abstained": confidence < threshold if threshold is not None else None, "threshold": threshold,
            "top_k": [{"class_name": metadata["classes"][int(i)], "probability": float(calibrated[i])} for i in indices],
            "probability_type": "temperature_scaled" if calibration else "raw",
            "notice": "Research prototype. Confidence is not a guarantee of correctness or a definitive agricultural diagnosis."}
