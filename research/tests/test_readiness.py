import copy
import json
from pathlib import Path
import pytest
from research.storage import paths, managed_path, configure_runtime
from research.src.manifest_validation import validate_manifest, require_research_manifest
from research.src.utils import ROOT, load_config
from research.scripts.full_suite import configs, aggregate, clone_prepared_run


def manifest(root):
    result = {"classes": ["a", "b"]}
    for split in ["train", "val", "test", "external"]:
        dataset = "plantdoc" if split == "external" else "plantvillage"
        result[split] = [{"path": str(root / "raw" / dataset / "a" / f"{split}.png"), "label": "a",
                          "pixel_hash": split, "dhash": split, "sha256": split, "leaf_id": split}]
    return result


def test_storage_relocation_and_smoke_separation(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_STORAGE_ROOT", str(tmp_path))
    monkeypatch.delenv("DATA_DIR", raising=False)
    monkeypatch.delenv("ARTIFACTS_DIR", raising=False)
    assert paths()["data_dir"] == tmp_path / "full/data"
    assert paths("smoke")["data_dir"] == tmp_path / "smoke/data"
    assert paths()["artifacts_dir"] != paths("smoke")["artifacts_dir"]
    with pytest.raises(ValueError, match="storage must stay"):
        managed_path(tmp_path.parent / "escaped")
    monkeypatch.setenv("DATA_DIR", str(tmp_path.parent / "escaped"))
    with pytest.raises(ValueError): load_config(ROOT / "research/configs/full_experiment.yaml")


@pytest.mark.parametrize("key", ["path", "sha256", "pixel_hash", "dhash", "leaf_id"])
def test_manifest_overlap_rejected(tmp_path, key):
    value = manifest(tmp_path)
    value["test"][0][key] = value["train"][0][key]
    with pytest.raises(ValueError, match="Data leakage"):
        validate_manifest(value, tmp_path)


def test_plantdoc_cannot_enter_fitting(tmp_path):
    value = manifest(tmp_path)
    value["val"] = value["external"]
    with pytest.raises(ValueError, match="Foreign image"):
        validate_manifest(value, tmp_path)


def test_changed_image_rejected(tmp_path):
    value = manifest(tmp_path)
    for split in ["train", "val", "test", "external"]:
        path = Path(value[split][0]["path"])
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Image changed"):
        validate_manifest(value, tmp_path, verify_files=True)


def test_research_refuses_unverified_corpora(tmp_path):
    value = manifest(tmp_path)
    with pytest.raises(ValueError, match="complete, audited"):
        require_research_manifest(value, {"purpose": "research"})


def test_prespecified_seed_matrix():
    values = configs(ROOT / "research/configs/full_experiment.yaml")
    assert [c["seed"] for c in values] == [42, 43, 44]
    assert all(c["split_seed"] == 42 and c["pretrained"] and not c["allow_random_fallback"] for c in values)
    assert len({c["run_id"] for c in values}) == 3


def test_clone_prepared_run_reuses_fixed_split_without_overwriting(tmp_path):
    data_dir, artifacts_dir = tmp_path / "data", tmp_path / "artifacts"
    source_cfg = {"run_id": "full_seed42", "seed": 42, "split_seed": 42,
                  "data_dir": str(data_dir), "artifacts_dir": str(artifacts_dir),
                  "purpose": "research", "leaf_group_policy": "official_with_filename_fallback",
                  "max_groups_per_class": None}
    target_cfg = {**source_cfg, "run_id": "full_seed43", "seed": 43}
    source_manifest = manifest(data_dir)
    pinned = json.loads((ROOT / "research/configs/sources.json").read_text())
    source_manifest.update(
        run_id=source_cfg["run_id"], seed=source_cfg["seed"], split_seed=42,
        purpose="research", max_groups_per_class=None, full_inventory_verified=True,
        leaf_group_policy=source_cfg["leaf_group_policy"],
        leaf_id_fallback_count=0, leaf_id_fallback_counts={},
        sources={"acquisition_complete": True,
                 "plantvillage": {"revision": pinned["plantvillage"]},
                 "plantdoc": {"revision": pinned["plantdoc"]}})
    source_manifest_path = data_dir / "splits" / source_cfg["run_id"] / "manifest.json"
    source_manifest_path.parent.mkdir(parents=True)
    source_manifest_path.write_text(json.dumps(source_manifest))
    summary = {"run_id": source_cfg["run_id"], "counts": {}}
    processed_summary = data_dir / "processed" / source_cfg["run_id"] / "dataset_summary.json"
    processed_summary.parent.mkdir(parents=True)
    processed_summary.write_text(json.dumps(summary))
    report_dir = artifacts_dir / "reports" / source_cfg["run_id"]
    report_dir.mkdir(parents=True)
    (report_dir / "dataset_summary.json").write_text(json.dumps(summary))
    (report_dir / "preparation_review.json").write_text(json.dumps({"leakage_check": {}}))
    (report_dir / "plantvillage_audit.json").write_text("{}")
    (report_dir / "plantdoc_audit.json").write_text("{}")
    source_plot = artifacts_dir / "plots" / source_cfg["run_id"] / "dataset_distribution.png"
    source_plot.parent.mkdir(parents=True)
    source_plot.write_bytes(b"plot")

    cloned = clone_prepared_run(source_cfg, target_cfg)
    assert cloned["run_id"] == target_cfg["run_id"] and cloned["seed"] == 43
    assert cloned["train"] == source_manifest["train"]
    cloned_summary = json.loads(
        (data_dir / "processed" / target_cfg["run_id"] / "dataset_summary.json").read_text())
    assert cloned_summary["run_id"] == target_cfg["run_id"]
    with pytest.raises(FileExistsError, match="Preserving existing"):
        clone_prepared_run(source_cfg, target_cfg)


def test_no_random_final_initialization(tmp_path):
    text = (ROOT / "research/configs/full_experiment.yaml").read_text().replace("pretrained: true", "pretrained: false")
    file = tmp_path / "invalid.yaml"; file.write_text(text)
    with pytest.raises(ValueError, match="pretrained weights"):
        load_config(file)


def test_missing_official_metadata_blocks_final_preparation(tmp_path):
    from research.scripts.prepare_data import prepare
    cfg = {"purpose": "research", "run_id": "fixture", "data_dir": str(tmp_path), "artifacts_dir": str(tmp_path / "out")}
    with pytest.raises(ValueError, match="Official metadata missing"):
        prepare(cfg)


def test_existing_final_checkpoint_preserved(tmp_path):
    from research.src.train import train
    from research.src.utils import save_json
    value = manifest(tmp_path)
    value.update(purpose="research", sources={"acquisition_complete": True}, max_groups_per_class=None,
                 leaf_id_fallback_count=0, leaf_id_fallback_counts={}, leaf_group_policy="official_only",
                 full_inventory_verified=True)
    for name, revision in json.loads((ROOT / "research/configs/sources.json").read_text()).items():
        value["sources"][name] = {"revision": revision}
    for split in ["train", "val", "test", "external"]:
        path = Path(value[split][0]["path"]); path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(split.encode())
        from research.src.utils import digest
        value[split][0]["sha256"] = digest(path)
    save_json(tmp_path / "splits/final/manifest.json", value)
    checkpoint = tmp_path / "models/final/resnet18.pt"
    checkpoint.parent.mkdir(parents=True); checkpoint.write_bytes(b"existing evidence")
    cfg = {"seed": 42, "data_dir": str(tmp_path), "artifacts_dir": str(tmp_path), "run_id": "final", "purpose": "research"}
    with pytest.raises(FileExistsError, match="Preserving"):
        train(cfg, "resnet18")
    assert checkpoint.read_bytes() == b"existing evidence"


def test_low_disk_blocks_download_before_network(tmp_path, monkeypatch):
    from research.scripts import download_data
    from types import SimpleNamespace
    monkeypatch.setenv("RESEARCH_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setattr("sys.argv", ["download_data", "--download"])
    monkeypatch.setattr(download_data.shutil, "disk_usage", lambda _: SimpleNamespace(free=1024**3))
    def unexpected(*args, **kwargs):
        pytest.fail("Network called before disk-space rejection")
    monkeypatch.setattr(download_data, "fetch", unexpected)
    with pytest.raises(ValueError, match="GiB free before downloading"):
        download_data.main()


@pytest.mark.parametrize("smoke", [False, True])
def test_aggregation_statistics_and_smoke_rejection(tmp_path, smoke):
    from research.src.utils import save_json, digest
    values = []
    for seed, metric in zip([42, 43, 44], [0.2, 0.4, 0.6]):
        cfg = {"seed": seed, "run_id": f"fixture{seed}", "models": ["resnet18"], "data_dir": str(tmp_path / "data"), "artifacts_dir": str(tmp_path / "artifacts")}
        values.append(cfg)
        manifest_path = Path(cfg["data_dir"]) / "splits" / cfg["run_id"] / "manifest.json"
        save_json(manifest_path, manifest(Path(cfg["data_dir"])))
        checkpoint = Path(cfg["artifacts_dir"]) / "models" / cfg["run_id"] / "resnet18.pt"
        checkpoint.parent.mkdir(parents=True); checkpoint.write_bytes(b"unit-test fixture, not a model")
        domain = {"classification": {"accuracy": metric, "macro_f1": metric, "balanced_accuracy": metric, "macro_precision": metric, "macro_recall": metric},
                  "raw": {"ece": metric, "brier": metric}, "calibrated": {"ece": metric, "brier": metric},
                  "selective": {"coverage": 0, "risk": None, "accuracy": None}}
        result = {"purpose": "smoke" if smoke else "research", "id": cfg["run_id"], "model_name": "resnet18", "pretrained_loaded": True,
                  "config": cfg, "manifest_sha256": digest(manifest_path), "checkpoint_sha256": digest(checkpoint),
                  "efficiency": {"parameters": 100, "checkpoint_mb": 1., "latency_ms_median": metric, "latency_ms_p95": metric},
                  "domains": {name: domain for name in ["in_domain", "in_domain_common", "external"]}}
        save_json(Path(cfg["artifacts_dir"]) / "results" / cfg["run_id"] / "resnet18_evaluation.json", result)
    if smoke:
        with pytest.raises(ValueError, match="prespecified"):
            aggregate(values)
    else:
        aggregate(values)
        result = json.loads((tmp_path / "artifacts/aggregate/metrics.json").read_text())
        accuracy = next(r for r in result["metrics"] if r["metric"] == "classification.accuracy")
        assert accuracy["mean"] == pytest.approx(0.4) and accuracy["sample_sd"] == pytest.approx(0.2)
        risk = next(r for r in result["metrics"] if r["metric"] == "selective.risk")
        assert risk["mean"] is None and risk["sample_sd"] is None and risk["n_defined"] == 0


def test_leaf_fallback_requires_explicit_recorded_policy(tmp_path):
    value = manifest(tmp_path)
    value.update(purpose="research", sources={"acquisition_complete": True}, max_groups_per_class=None,
                 leaf_id_fallback_count=1, leaf_id_fallback_counts={"a": 1}, leaf_group_policy="official_only",
                 full_inventory_verified=True)
    for name, revision in json.loads((ROOT / "research/configs/sources.json").read_text()).items():
        value["sources"][name] = {"revision": revision}
    cfg = {"purpose": "research"}
    with pytest.raises(ValueError, match="explicit, recorded"):
        require_research_manifest(value, cfg)
    cfg["leaf_group_policy"] = "official_with_filename_fallback"
    with pytest.raises(ValueError, match="differs"):
        require_research_manifest(value, cfg)
    value["leaf_group_policy"] = cfg["leaf_group_policy"]
    require_research_manifest(value, cfg)
