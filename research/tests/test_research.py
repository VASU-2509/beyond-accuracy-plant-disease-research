import io
import zipfile
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image

from research.src.utils import load_config, ROOT
from research.src.calibration import fit_temperature, probabilities
from research.src.metrics import classification, reliability
from research.src.selective_prediction import select_threshold, selective_metrics, risk_curve
from research.src.data_audit import audit
from research.src.data_splitting import grouped_split
from research.src.label_mapping import map_label
from research.src.model_factory import build_model, load_checkpoint
from research.scripts.download_data import fetch, safe_extract, verify_full_inventory
from research.scripts.prepare_data import leaf_group_id

def test_config_validation(tmp_path):
    cfg = load_config(ROOT / "research/configs/quick_test.yaml")
    assert cfg["batch_size"] >= 2 and set(cfg["models"]) == {"resnet18", "mobilenet_v3_small"}
    path = tmp_path / "bad.yaml"
    path.write_text("epochs: -1")
    with pytest.raises(ValueError):
        load_config(path)

def test_audit_corruption_and_duplicates(tmp_path):
    folder = tmp_path / "class_a"
    folder.mkdir()
    pixels = np.random.default_rng(2).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(folder / "one.png")
    Image.fromarray(pixels).save(folder / "two.png")
    (folder / "corrupt.jpg").write_bytes(b"not an image")
    result = audit(tmp_path)
    assert result["count"] == 2 and len(result["corrupt"]) == 1
    assert len(result["duplicates"]["pixel_hash"]) == 1

def test_group_split_no_leakage_and_determinism():
    rows = [{"path": f"{c}{i}", "label": c, "pixel_hash": f"{c}{i}", "dhash": f"d{c}{i}", "leaf_id": f"leaf{c}{i//2}"} for c in ["a", "b"] for i in range(20)]
    first, _ = grouped_split(rows, 42)
    second, _ = grouped_split(rows, 42)
    assert first == second
    for key in ["path", "pixel_hash", "dhash", "leaf_id"]:
        sets = [{r[key] for r in first[s]} for s in ["train", "val", "test"]]
        assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert sum(map(len, first.values())) == len(rows)

def test_conflicting_groups_quarantined():
    rows = [{"label": c, "pixel_hash": f"{c}{i}", "dhash": f"{c}{i}"} for c in ["a", "b"] for i in range(5)]
    rows += [{"label": "a", "pixel_hash": "conflict", "dhash": "conflict"}, {"label": "b", "pixel_hash": "conflict", "dhash": "conflict"}]
    splits, excluded = grouped_split(rows)
    assert len(excluded) == 2
    assert sum(map(len, splits.values())) == 10

def test_mapping_excludes_ambiguous():
    assert map_label("Apple Scab Leaf", ["Apple___Apple_scab"]) == "Apple___Apple_scab"
    assert map_label("Corn leaf blight", ["Corn_(maize)___Northern_Leaf_Blight"]) is None
    assert map_label("Tomato leaf yellow virus", ["Tomato___Tomato_Yellow_Leaf_Curl_Virus"]) is None
    assert map_label("Apple leaf", ["different"]) is None

def test_metrics_known_values_and_boundary_confidence():
    p = np.array([[1., 0.], [0., 1.]])
    r = reliability(p, np.array([0, 1]))
    assert r["ece"] == 0 and r["brier"] == 0 and r["bins"][-1]["count"] == 2
    r = reliability(np.array([[.8, .2], [.7, .3]]), np.array([0, 1]), bins=1)
    assert r["ece"] == pytest.approx(.25)
    assert r["brier"] == pytest.approx(.53)
    c = classification(p, [0, 1], ["a", "b"])
    assert c["accuracy"] == c["macro_f1"] == c["balanced_accuracy"] == 1

def test_external_nonshared_prediction_remains_error():
    result = classification(np.array([[.1,.2,.7],[.8,.1,.1]]), [0,0], ["a","b","c"], [0])
    assert result["accuracy"] == .5
    assert result["confusion_matrix"][0][2] == 1
    assert result["macro_recall"] == .5

def test_temperature_reduces_nll_and_preserves_argmax():
    logits = np.array([[9.,0.],[9.,0.],[0.,9.],[0.,9.]])
    labels = np.array([0,1,1,0])
    fit = fit_temperature(logits, labels)
    assert fit["temperature"] > 1
    assert fit["validation_nll_after"] < fit["validation_nll_before"]
    assert np.array_equal(probabilities(logits).argmax(1), probabilities(logits, fit["temperature"]).argmax(1))

def test_selective_ties_and_no_feasible_threshold():
    p = np.array([[.9,.1],[.9,.1],[.55,.45]])
    chosen = select_threshold(p, [0,1,1], target_risk=0, min_coverage=.2)
    assert not chosen["constraint_satisfied"] and chosen["coverage"] == 0 and chosen["risk"] is None
    assert all(r["accepted"] != 1 for r in risk_curve(p, [0,1,1]))
    chosen = select_threshold(np.array([[.99,.01],[.6,.4],[.1,.9]]), [0,1,1], .1, .2)
    assert chosen["coverage"] == pytest.approx(2/3) and chosen["risk"] == 0

@pytest.mark.parametrize("name", ["resnet18", "mobilenet_v3_small"])
def test_model_checkpoint_and_feature_extraction(name, tmp_path):
    torch.set_num_threads(2)
    model = build_model(name, 3, mode="feature_extraction").eval()
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) < sum(p.numel() for p in model.parameters())
    sample = torch.randn(2,3,64,64)
    with torch.no_grad():
        expected = model(sample)
    assert expected.shape == (2,3)
    path = tmp_path / "checkpoint.pt"
    torch.save({"state_dict": model.state_dict(), "model_name": name, "classes": ["a","b","c"], "image_size":64, "run_id":"test", "purpose":"synthetic", "manifest_sha256":"fixture"}, path)
    restored, _ = load_checkpoint(path)
    with torch.no_grad():
        assert torch.allclose(expected, restored(sample))

def test_zip_traversal_rejected(tmp_path):
    path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("../escape.txt", "bad")
    with pytest.raises(ValueError, match="Unsafe"):
        safe_extract(path, tmp_path / "out")

def test_safe_extract_reuses_matching_files_and_replaces_changed_bytes(tmp_path):
    archive_path = tmp_path / "data.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("class_a/image.txt", "verified image")
    target = tmp_path / "out"
    safe_extract(archive_path, target)
    extracted = target / "class_a/image.txt"
    original_time = extracted.stat().st_mtime_ns
    safe_extract(archive_path, target)
    assert extracted.stat().st_mtime_ns == original_time
    extracted.write_text("changed bytes")
    safe_extract(archive_path, target)
    assert extracted.read_text() == "verified image"

def test_safe_extract_encodes_windows_invalid_filename_characters(tmp_path):
    archive_path = tmp_path / "data.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("Dataset/test/class/image?query=1.jpg", b"image bytes")
    target = tmp_path / "out"
    renamed = safe_extract(archive_path, target)
    extracted = target / "Dataset/test/class/image%3Fquery=1.jpg"
    assert extracted.read_bytes() == b"image bytes"
    assert renamed == [{
        "archive_path": "Dataset/test/class/image?query=1.jpg",
        "extracted_path": "Dataset/test/class/image%3Fquery=1.jpg",
    }]

def test_safe_extract_preserves_case_colliding_archive_files(tmp_path):
    archive_path = tmp_path / "data.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("Dataset/train/class/CAR1.jpg", b"uppercase image")
        archive.writestr("Dataset/train/class/car1.jpg", b"lowercase image")
    target = tmp_path / "out"
    renamed = safe_extract(archive_path, target)
    class_dir = target / "Dataset/train/class"
    assert len(list(class_dir.glob("*.jpg"))) == 2
    assert (class_dir / "CAR1.jpg").read_bytes() == b"uppercase image"
    alternate = next(class_dir.glob("car1__*.jpg"))
    assert alternate.read_bytes() == b"lowercase image"
    assert renamed == [{
        "archive_path": "Dataset/train/class/car1.jpg",
        "extracted_path": f"Dataset/train/class/{alternate.name}",
    }]

def test_safe_extract_reuses_legacy_escaped_file_without_leaving_duplicate(tmp_path):
    archive_path = tmp_path / "data.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("Dataset/test/class/leaf%20image?.jpg", b"verified bytes")
    target = tmp_path / "out"
    legacy = target / "Dataset/test/class/leaf%2520image%3F.jpg"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"verified bytes")
    safe_extract(archive_path, target)
    current = target / "Dataset/test/class/leaf%20image%3F.jpg"
    assert current.read_bytes() == b"verified bytes"
    assert not legacy.exists()

def test_full_inventory_verification_checks_declared_files_and_external_counts(tmp_path):
    plantvillage = tmp_path / "plantvillage"
    plantdoc = tmp_path / "plantdoc"
    (plantvillage / "archive" / "raw" / "color" / "class").mkdir(parents=True)
    (plantdoc / "archive" / "train" / "Apple").mkdir(parents=True)
    (plantvillage / "archive" / "raw" / "color" / "class" / "image.JPG").write_bytes(b"pv")
    (plantdoc / "archive" / "train" / "Apple" / "image.jpeg").write_bytes(b"pd")
    result = verify_full_inventory(
        plantvillage, plantdoc, {"train": ["raw/color/class/image.JPG"]},
        {"train/Apple": 1})
    assert result["plantvillage_image_count"] == 1
    assert result["plantdoc_image_count"] == 1
    with pytest.raises(ValueError, match="PlantDoc"):
        verify_full_inventory(
            plantvillage, plantdoc, {"train": ["raw/color/class/image.JPG"]},
            {"train/Apple": 2})

def test_fetch_resumes_a_partial_and_checks_the_completed_digest(tmp_path, monkeypatch):
    import hashlib
    from research.scripts import download_data

    class Response(io.BytesIO):
        def __init__(self, value, status, headers):
            super().__init__(value)
            self.status = status
            self.headers = headers

    payload, prefix = b"abcdef", b"abc"
    target = tmp_path / "archive.bin"
    target.with_suffix(".bin.partial").write_bytes(prefix)
    requests = []

    def open_partial(request, timeout=120):
        requests.append(request.get_header("Range"))
        return Response(payload[len(prefix):], 206, {
            "Content-Range": f"bytes {len(prefix)}-{len(payload)-1}/{len(payload)}",
            "Content-Length": str(len(payload)-len(prefix)),
        })

    monkeypatch.setattr(download_data, "open_url", open_partial)
    fetch("https://example.invalid/archive", target, hashlib.sha256(payload).hexdigest())
    assert requests == ["bytes=3-"]
    assert target.read_bytes() == payload
    assert not target.with_suffix(".bin.partial").exists()

def test_fetch_retries_a_transient_connection_error(tmp_path, monkeypatch):
    import urllib.error
    from research.scripts import download_data

    class Response(io.BytesIO):
        status = 200
        headers = {"Content-Length": "3"}

    attempts = 0

    def flaky_open(request, timeout=120):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise urllib.error.URLError("temporary network failure")
        return Response(b"ok!")

    monkeypatch.setattr(download_data, "open_url", flaky_open)
    monkeypatch.setattr(download_data.time, "sleep", lambda _: None)
    target = tmp_path / "retried.bin"
    fetch("https://example.invalid/retried", target)
    assert attempts == 2 and target.read_bytes() == b"ok!"

def test_fetch_resumes_after_an_incomplete_http_read(tmp_path, monkeypatch):
    import http.client
    from research.scripts import download_data

    class InterruptedResponse(io.BytesIO):
        status = 200
        headers = {"Content-Length": "4"}

        def __init__(self):
            super().__init__()
            self.read_count = 0

        def read(self, size=-1):
            self.read_count += 1
            if self.read_count == 1:
                return b"ab"
            raise http.client.IncompleteRead(b"", 2)

    class ResumedResponse(io.BytesIO):
        status = 206
        headers = {"Content-Range": "bytes 2-3/4", "Content-Length": "2"}

    requests = []
    responses = [InterruptedResponse(), ResumedResponse(b"cd")]

    def open_partial(request, timeout=120):
        requests.append(request.get_header("Range"))
        return responses.pop(0)

    monkeypatch.setattr(download_data, "open_url", open_partial)
    monkeypatch.setattr(download_data.time, "sleep", lambda _: None)
    target = tmp_path / "incomplete.bin"
    fetch("https://example.invalid/incomplete", target)
    assert requests == [None, "bytes=2-"]
    assert target.read_bytes() == b"abcd"

def test_leaf_map_requires_unambiguous_class_match_for_multiple_suggestions():
    assert leaf_group_id("apple", "Apple___Black_rot", {
        "apple": ["Apple_Frogeye Spot:::54.0"],
    }) == "Apple_Frogeye Spot:::54.0"
    assert leaf_group_id("healthy", "Apple___healthy", {
        "healthy": ["Apple___healthy:::1.0", "Apple___healthy:::2.0"],
    }) == "fallback_Apple___healthy_healthy"

def test_risk_curve_matches_brute_force():
    rng = np.random.default_rng(123)
    p = probabilities(rng.normal(size=(40, 3)))
    y = rng.integers(0, 3, 40)
    for row in risk_curve(p, y):
        expected = selective_metrics(p, y, row["threshold"])
        assert row["accepted"] == expected["accepted"]
        assert row["risk"] == pytest.approx(expected["risk"]) if row["risk"] is not None else expected["risk"] is None

def test_research_run_rejects_known_smoke_data(tmp_path):
    from research.scripts.prepare_data import prepare
    from research.src.utils import save_json
    save_json(tmp_path / "raw" / "sources.json", {"subset_per_class": 6})
    cfg = load_config(ROOT / "research/configs/full_experiment.yaml")
    cfg["data_dir"] = str(tmp_path)
    with pytest.raises(ValueError, match="cannot use"):
        prepare(cfg)
