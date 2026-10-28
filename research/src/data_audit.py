"""Decode images, inventory classes and identify conservative duplicate candidates."""
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from .utils import digest

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def inspect_image(path):
    with Image.open(path) as im:
        fmt = im.format
        rgb = ImageOps.exif_transpose(im).convert("RGB")
        rgb.load()
        pixels = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        small = np.asarray(rgb.convert("L").resize((9, 8)), dtype=np.int16)
        dhash = np.packbits(small[:, 1:] > small[:, :-1]).tobytes().hex()
        return {"width": rgb.width, "height": rgb.height, "format": fmt,
                "pixel_hash": pixels, "dhash": dhash}

def audit(root):
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")
    rows, corrupt = [], []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            try:
                row = {"path": str(path.resolve()), "label": path.parent.name, "sha256": digest(path), **inspect_image(path)}
                # PlantVillage filename suffix identifies repeated photographs of a leaf.
                row["leaf_id"] = path.stem.split("___", 1)[1] if "___" in path.stem else None
                rows.append(row)
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                corrupt.append({"path": str(path), "error": str(exc)})
    if not rows:
        raise ValueError(f"No readable images in {root}")
    counts = Counter(r["label"] for r in rows)
    duplicates = {}
    for key in ["pixel_hash", "dhash"]:
        groups = {}
        for row in rows:
            groups.setdefault(row[key], []).append(row["path"])
        duplicates[key] = [g for g in groups.values() if len(g) > 1]
    return {"root": str(root.resolve()), "count": len(rows), "counts": dict(counts),
            "corrupt": corrupt, "duplicates": duplicates, "images": rows,
            "near_duplicate_policy": "Equal 64-bit dHash grouped conservatively; other near duplicates may remain."}
