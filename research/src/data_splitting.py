"""Group-aware per-class split. Conflicting duplicate labels are quarantined."""
import random
from collections import defaultdict

def grouped_split(rows, seed=42, limit_per_class=None):
    parent = list(range(len(rows)))
    def find(i):
        while i != parent[i]:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, row in enumerate(rows):
        for key in ["pixel_hash", "dhash", "leaf_id"]:
            value = row.get(key)
            if value:
                token = (key, value)
                if token in seen:
                    parent[find(i)] = find(seen[token])
                seen[token] = i
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(row)
    by_class, excluded = defaultdict(list), []
    for group in groups.values():
        if len({r["label"] for r in group}) != 1:
            excluded.extend({**r, "reason": "Conflicting labels in duplicate group"} for r in group)
        else:
            by_class[group[0]["label"]].append(group)
    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for label, class_groups in sorted(by_class.items()):
        rng.shuffle(class_groups)
        if limit_per_class:
            # Cap groups, not images; never split a duplicate group.
            class_groups = class_groups[:limit_per_class]
        n = len(class_groups)
        if n < 3:
            raise ValueError(f"{label}: at least 3 independent image groups required (found {n})")
        n_test = max(1, round(n * .15))
        n_val = max(1, round(n * .15))
        for split, selected in [("test", class_groups[:n_test]), ("val", class_groups[n_test:n_test+n_val]), ("train", class_groups[n_test+n_val:])]:
            for group in selected:
                splits[split].extend(group)
    if not splits["train"]:
        raise ValueError("No eligible training examples")
    return splits, excluded
