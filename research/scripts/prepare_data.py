import argparse
from pathlib import Path
from collections import Counter
from research.src.data_audit import audit
from research.src.data_splitting import grouped_split
from research.src.label_mapping import map_label, PLANTDOC_TO_PLANTVILLAGE, EXCLUSIONS
from research.src.plots import distribution
from research.src.utils import load_config, read_json, save_json, git_blob_digest, ROOT
from research.src.manifest_validation import validate_manifest, require_research_manifest

def leaf_group_id(identifier, label, leaf_map):
    suggestions = leaf_map.get(identifier, [])
    if len(suggestions) == 1:
        # A unique upstream mapping remains valid for legacy source-class aliases.
        return suggestions[0]
    matches = {suggestion for suggestion in suggestions
               if suggestion.partition(":::")[0] == label}
    if len(matches) == 1:
        return matches.pop()
    return f"fallback_{label}_{identifier}"

def prepare(cfg):
    root, output = Path(cfg["data_dir"]), Path(cfg["artifacts_dir"])
    provenance_path = root / "raw" / "sources.json"
    provenance = read_json(provenance_path) if provenance_path.exists() else {}
    manifest_path = root / "splits" / cfg["run_id"] / "manifest.json"
    if cfg.get("purpose") == "research" and manifest_path.exists():
        raise FileExistsError("Preserving existing research manifest; choose a new run_id")
    if cfg.get("purpose") == "research" and (provenance.get("subset_per_class") or provenance.get("synthetic")):
        raise ValueError("Research runs cannot use acquisition smoke subsets or synthetic fixtures. Download full data to a separate DATA_DIR.")
    if cfg.get("purpose") == "research":
        inventory = read_json(ROOT / "research/configs/source_inventory.json")
        for name in ["splits/color_train.txt", "splits/color_test.txt", "leaf_grouping/leaf-map.json"]:
            path = root / "raw/metadata" / name
            if not path.exists() or git_blob_digest(path) != inventory["plantvillage_metadata_git_oids"][name]:
                raise ValueError(f"Official metadata missing or checksum mismatch: {path}")
    report = audit(root / "raw" / "plantvillage")
    leaf_map_path = root / "raw" / "metadata" / "leaf_grouping" / "leaf-map.json"
    leaf_map = read_json(leaf_map_path) if leaf_map_path.exists() else {}
    fallback_count = 0
    fallback_counts = Counter()
    for row in report["images"]:
        identifier = Path(row["path"]).stem.replace("_final_masked", "").split("___")[-1].split("copy")[0].lower().strip()
        row["leaf_id"] = leaf_group_id(identifier, row["label"], leaf_map)
        fallback_count += row["leaf_id"].startswith("fallback_")
        if row["leaf_id"].startswith("fallback_"):
            fallback_counts[row["label"]] += 1
    splits, excluded = grouped_split(report["images"], cfg.get("split_seed", cfg["seed"]), cfg.get("max_groups_per_class"))
    classes = sorted({r["label"] for r in splits["train"]})
    external, exclusions = [], []
    pd_root = root / "raw" / "plantdoc"
    external_report = audit(pd_root) if pd_root.exists() else None
    # All PlantVillage images, including unused quick-subset images, guard contamination.
    pv_pixels = {r["pixel_hash"] for r in report["images"]}
    pv_dhash = {r["dhash"] for r in report["images"]}
    seen_pixels, seen_dhash = set(), set()
    if external_report:
        for row in external_report["images"]:
            mapped = map_label(row["label"], classes)
            reason = None
            if not mapped:
                reason = EXCLUSIONS.get(row["label"], "No compatible training class")
            elif row["pixel_hash"] in pv_pixels or row["dhash"] in pv_dhash:
                reason = "Overlap with PlantVillage"
            elif row["pixel_hash"] in seen_pixels or row["dhash"] in seen_dhash:
                reason = "Duplicate external image"
            if reason:
                exclusions.append({**row, "reason": reason})
            else:
                external.append({**row, "source_label": row["label"], "label": mapped})
                seen_pixels.add(row["pixel_hash"])
                seen_dhash.add(row["dhash"])
    source_path = root / "raw" / "sources.json"
    manifest = {**splits, "external": external, "classes": classes, "seed": cfg["seed"], "run_id": cfg["run_id"],
                "split_seed": cfg.get("split_seed", cfg["seed"]),
                "max_groups_per_class": cfg.get("max_groups_per_class"), "purpose": cfg.get("purpose", "research"),
                "sources": read_json(source_path) if source_path.exists() else {"source": "manual; version not supplied"},
                "mapping": PLANTDOC_TO_PLANTVILLAGE, "excluded_training": excluded, "excluded_external": exclusions,
                "leaf_id_fallback_count": fallback_count,
                "leaf_id_fallback_counts": dict(fallback_counts),
                "leaf_group_policy": cfg.get("leaf_group_policy", "official_only"),
                "split_protocol": "Custom approximately 70/15/15 per-class groups; not the published split; external never used for fitting"}
    manifest["full_inventory_verified"] = False
    if cfg.get("purpose") == "research":
        declared = set()
        for split in ["train", "test"]:
            path = root / "raw/metadata/splits" / f"color_{split}.txt"
            if path.exists():
                declared.update((Path(line.strip()).parent.name, Path(line.strip()).name) for line in path.read_text().splitlines() if line.strip())
        observed = {(Path(row["path"]).parent.name, Path(row["path"]).name) for row in report["images"] + report["corrupt"]}
        expected_pd = read_json(Path(__file__).resolve().parents[1] / "configs/source_inventory.json")["plantdoc_image_counts"]
        actual_pd = Counter()
        if external_report:
            for row in external_report["images"] + external_report["corrupt"]:
                parts = Path(row["path"]).parts
                for bucket in ["train", "test"]:
                    if bucket in parts:
                        actual_pd[bucket + "/" + Path(row["path"]).parent.name] += 1
                        break
        manifest["full_inventory_verified"] = (bool(declared) and declared == observed
            and len(observed) == len(report["images"]) + len(report["corrupt"]) and dict(actual_pd) == expected_pd)
    manifest["leakage_check"] = validate_manifest(manifest, root)
    counts = {s: dict(Counter(r["label"] for r in rows)) for s, rows in {**splits, "external": external}.items()}
    summary = {"run_id": cfg["run_id"], "purpose": cfg.get("purpose", "research"), "counts": counts, "classes": classes,
               "corrupt": len(report["corrupt"])+(len(external_report["corrupt"]) if external_report else 0),
               "external_excluded": len(exclusions), "sources": manifest["sources"], "leaf_id_fallback_count": fallback_count,
               "leaf_id_fallback_counts": dict(fallback_counts), "leaf_group_policy": manifest["leaf_group_policy"]}
    save_json(root / "processed" / cfg["run_id"] / "dataset_summary.json", summary)
    save_json(output / "reports" / cfg["run_id"] / "dataset_summary.json", summary)
    save_json(output / "reports" / cfg["run_id"] / "plantvillage_audit.json", report)
    if external_report:
        save_json(output / "reports" / cfg["run_id"] / "plantdoc_audit.json", external_report)
    save_json(output / "reports" / cfg["run_id"] / "preparation_review.json", {
        "full_inventory_verified": manifest["full_inventory_verified"],
        "leaf_id_fallback_count": fallback_count, "leakage_check": manifest["leakage_check"],
        "leaf_id_fallback_counts": dict(fallback_counts), "leaf_group_policy": manifest["leaf_group_policy"],
        "leaf_group_limitation": "Official IDs where available; missing entries use class-scoped filename IDs and image hashes. Undetected related leaves may remain.",
        "mapping": PLANTDOC_TO_PLANTVILLAGE, "mapping_exclusions": EXCLUSIONS,
        "eligible_external_classes": sorted({r["label"] for r in external}),
        "excluded_training": excluded, "excluded_external": exclusions})
    require_research_manifest(manifest, cfg)
    save_json(manifest_path, manifest)
    distribution(report["counts"], output / "plots" / cfg["run_id"] / "dataset_distribution.png")
    print({split: sum(c.values()) for split, c in counts.items()})
    return manifest

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="research/configs/quick_test.yaml")
    prepare(load_config(parser.parse_args().config))

if __name__ == "__main__":
    main()
