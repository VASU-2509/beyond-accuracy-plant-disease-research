from collections import OrderedDict
import json
import threading

from .model_service import checkpoints

class InferenceService:
    def __init__(self, settings):
        self.settings = settings
        self.cache = OrderedDict()
        self.lock = threading.Lock()

    def predict(self, model_id, image, top_k=5):
        # Serialize on a small host to bound memory and CPU contention.
        with self.lock:
            path = checkpoints(self.settings).get(model_id)
            if path is None:
                raise FileNotFoundError("Model checkpoint unavailable. Run training and evaluation first.")
            sidecar = path.with_suffix(".calibration.json")
            stamp = (str(path), path.stat().st_mtime_ns, sidecar.stat().st_mtime_ns if sidecar.exists() else None)
            if stamp not in self.cache:
                from research.src.model_factory import load_checkpoint
                from research.src.utils import digest
                import torch
                torch.set_num_threads(2)
                model, metadata = load_checkpoint(path)
                calibration = json.loads(sidecar.read_text()) if sidecar.exists() else None
                if calibration and calibration["checkpoint_sha256"] != digest(path):
                    raise ValueError("Calibration does not match checkpoint; rerun evaluation")
                self.cache.clear()  # One model resident at a time on 8 GB machines.
                self.cache[stamp] = model, metadata, calibration
            model, metadata, calibration = self.cache[stamp]
            from research.src.predict import predict_image
            return predict_image(model, metadata, image, calibration, top_k)
