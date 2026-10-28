import numpy as np

def selective_metrics(probabilities, labels, threshold):
    p, y = np.asarray(probabilities), np.asarray(labels)
    if not len(y):
        raise ValueError("Nonempty labels required")
    accepted = p.max(1) >= threshold
    n = int(accepted.sum())
    accuracy = float((p.argmax(1)[accepted] == y[accepted]).mean()) if n else None
    return {"threshold": float(threshold), "accepted": n, "total": len(y), "coverage": n/len(y),
            "accuracy": accuracy, "risk": 1-accuracy if accuracy is not None else None}

def risk_curve(probabilities, labels):
    # Unique thresholds preserve ties; no arbitrary acceptance order within ties.
    p, y = np.asarray(probabilities), np.asarray(labels)
    if not len(y) or len(p) != len(y):
        raise ValueError("Nonempty aligned probabilities and labels required")
    confidence = p.max(1)
    order = np.argsort(confidence)
    confidence = confidence[order]
    errors = (p.argmax(1) != y)[order].astype(int)
    suffix_errors = np.r_[np.cumsum(errors[::-1])[::-1], 0]
    rows = []
    for threshold in np.unique(np.r_[0., confidence, 1.000001]):
        index = int(np.searchsorted(confidence, threshold, side="left"))
        accepted = len(y) - index
        risk = float(suffix_errors[index] / accepted) if accepted else None
        rows.append({"threshold": float(threshold), "accepted": accepted, "total": len(y),
                     "coverage": accepted/len(y), "risk": risk, "accuracy": 1-risk if risk is not None else None})
    return rows

def select_threshold(validation_probabilities, validation_labels, target_risk=.1, min_coverage=.2):
    if not 0 <= target_risk <= 1 or not 0 <= min_coverage <= 1:
        raise ValueError("Risk and coverage must lie in [0,1]")
    candidates = [r for r in risk_curve(validation_probabilities, validation_labels)
                  if r["risk"] is not None and r["risk"] <= target_risk and r["coverage"] >= min_coverage]
    selected = max(candidates, key=lambda r: (r["coverage"], -r["threshold"])) if candidates else selective_metrics(validation_probabilities, validation_labels, 1.000001)
    return {**selected, "target_risk": target_risk, "min_coverage": min_coverage,
            "constraint_satisfied": bool(candidates), "selection_source": "PlantVillage validation only",
            "procedure": "Maximize validation coverage subject to empirical risk and minimum coverage; otherwise abstain on all. No risk guarantee."}
