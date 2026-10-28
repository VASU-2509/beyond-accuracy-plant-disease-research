"""Fail closed on split overlap, foreign fitting images or changed source bytes."""
from pathlib import Path
from .utils import digest, ROOT, read_json


def validate_manifest(manifest, data_dir, verify_files=False):
    root = Path(data_dir).resolve()
    splits = ["train", "val", "test", "external"]
    for split in splits:
        expected = root / "raw" / ("plantdoc" if split == "external" else "plantvillage")
        for row in manifest[split]:
            path = Path(row["path"]).resolve()
            if not path.is_relative_to(expected):
                raise ValueError(f"Foreign image in {split}: {path}")
            if row["label"] not in manifest["classes"]:
                raise ValueError(f"Unknown label in {split}")
            if split != "external" and not row.get("leaf_id"):
                raise ValueError(f"Missing source leaf grouping identifier in {split}: {path}")
            if verify_files and (not row.get("sha256") or digest(path) != row["sha256"]):
                raise ValueError(f"Image changed or checksum missing: {path}")
    for key in ["path", "sha256", "pixel_hash", "dhash", "leaf_id"]:
        sets = {s: {r[key] for r in manifest[s] if r.get(key)} for s in splits}
        for i, left in enumerate(splits):
            for right in splits[i+1:]:
                # PlantDoc leaf identifiers do not have a shared provenance vocabulary.
                if key == "leaf_id" and "external" in (left, right):
                    continue
                if sets[left] & sets[right]:
                    raise ValueError(f"Data leakage: {key} shared by {left} and {right}")
    if any(not manifest[s] for s in ["train", "val", "test"]):
        raise ValueError("Empty source split")
    return {"status": "passed", "counts": {s: len(manifest[s]) for s in splits},
            "checks": ["source path containment", "class vocabulary", "source leaf grouping IDs",
                       "pairwise path/byte/pixel/dHash/leaf overlap"],
            "source_bytes_verified": verify_files}


def require_research_manifest(manifest, cfg):
    if cfg.get("purpose") != "research":
        return
    sources = manifest.get("sources", {})
    pinned = read_json(ROOT / "research/configs/sources.json")
    if any(sources.get(name, {}).get("revision") != pinned[name] for name in ["plantvillage", "plantdoc"]):
        raise ValueError("Final research requires the pinned dataset revisions and complete, audited sources")
    policy = cfg.get("leaf_group_policy", "official_only")
    fallback_count = manifest.get("leaf_id_fallback_count", 1)
    fallback_counts = manifest.get("leaf_id_fallback_counts")
    if policy not in {"official_only", "official_with_filename_fallback"}:
        raise ValueError(f"Unsupported leaf grouping policy: {policy}")
    if manifest.get("leaf_group_policy") != policy:
        raise ValueError("Prepared leaf grouping policy differs from the research configuration")
    if (not isinstance(fallback_count, int) or isinstance(fallback_count, bool) or fallback_count < 0
            or not isinstance(fallback_counts, dict)
            or any(not isinstance(value, int) or isinstance(value, bool) or value < 0
                   for value in fallback_counts.values())
            or sum(fallback_counts.values()) != fallback_count):
        raise ValueError("Invalid or inconsistent leaf-map fallback counts")
    if fallback_count and policy != "official_with_filename_fallback":
        raise ValueError("Incomplete official leaf IDs require an explicit, recorded filename-fallback policy")
    if (manifest.get("purpose") != "research" or sources.get("subset_per_class")
            or sources.get("synthetic") or not sources.get("acquisition_complete")
            or manifest.get("max_groups_per_class") is not None
            or not manifest.get("external") or not manifest.get("full_inventory_verified")):
        raise ValueError("Final research requires complete, audited full corpora and eligible external images")
