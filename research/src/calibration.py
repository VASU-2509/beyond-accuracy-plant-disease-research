"""Scalar temperature fitted by validation negative log likelihood only."""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax

def probabilities(logits, temperature=1.0):
    logits = np.asarray(logits, dtype=float)
    if logits.ndim != 2 or not np.isfinite(logits).all() or temperature <= 0:
        raise ValueError("Finite NxC logits and positive temperature required")
    return softmax(logits / temperature, axis=1)

def fit_temperature(validation_logits, validation_labels):
    logits = np.asarray(validation_logits, dtype=float)
    labels = np.asarray(validation_labels, dtype=int)
    if len(labels) == 0 or logits.ndim != 2 or len(labels) != len(logits) or not np.isfinite(logits).all():
        raise ValueError("Nonempty finite validation logits and matching labels required")
    def nll(log_t):
        scaled = logits / np.exp(log_t)
        return float(np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(len(labels)), labels]))
    fit = minimize_scalar(nll, bounds=(-4, 4), method="bounded")
    t = float(np.exp(fit.x)) if fit.success and fit.fun <= nll(0) else 1.0
    return {"temperature": t, "validation_nll_before": nll(0), "validation_nll_after": nll(np.log(t)),
            "fit_source": "PlantVillage validation only", "log_temperature_bounds": [-4, 4]}
