import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

def reliability(probabilities, labels, bins=15):
    p, y = np.asarray(probabilities), np.asarray(labels)
    if len(y) == 0 or p.ndim != 2 or len(p) != len(y):
        raise ValueError("Nonempty aligned probabilities and labels required")
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1):
        raise ValueError("Invalid probabilities")
    confidence, correct = p.max(1), p.argmax(1) == y
    index = np.minimum((confidence * bins).astype(int), bins-1)
    points, ece = [], 0.0
    for i in range(bins):
        selected = index == i
        n = int(selected.sum())
        conf = float(confidence[selected].mean()) if n else None
        acc = float(correct[selected].mean()) if n else None
        if n:
            ece += n / len(y) * abs(conf-acc)
        points.append({"lower": i/bins, "upper": (i+1)/bins, "count": n, "confidence": conf, "accuracy": acc})
    one_hot = np.eye(p.shape[1])[y]
    return {"ece": float(ece), "brier": float(np.mean(np.sum((p-one_hot)**2, axis=1))),
            "bins": points, "confidence_histogram": [
                {"lower": i/bins, "upper": (i+1)/bins, "correct": int(((index == i) & correct).sum()),
                 "incorrect": int(((index == i) & ~correct).sum())} for i in range(bins)]}

def classification(probabilities, labels, classes, metric_labels=None):
    p, y = np.asarray(probabilities), np.asarray(labels)
    pred = p.argmax(1)
    eligible = np.asarray(metric_labels if metric_labels is not None else np.unique(y), dtype=int)
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, labels=eligible, average="macro", zero_division=0)
    return {"n": len(y), "accuracy": float(accuracy_score(y, pred)), "balanced_accuracy": float(recall),
            "macro_precision": float(precision), "macro_recall": float(recall), "macro_f1": float(f1),
            "metric_classes": [classes[i] for i in eligible],
            "per_class": classification_report(y, pred, labels=list(range(len(classes))), target_names=classes, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y, pred, labels=list(range(len(classes)))).tolist()}
