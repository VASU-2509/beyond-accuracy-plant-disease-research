"""Fetch pinned metadata; explicitly download full corpora or a tiny source subset."""
import argparse
import base64
import http.client
import hashlib
import json
import os
import random
import re
import shutil
import urllib.request
import zipfile
import time
import urllib.error
import zlib
from collections import Counter
from pathlib import Path, PureWindowsPath
from urllib.parse import quote

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

from research.src.utils import ROOT, save_json, digest, read_json, git_blob_digest
from research.storage import paths, managed_path, configure_runtime

def open_url(request, timeout=120):
    for attempt in range(4):
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(1 + attempt)

def fetch_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "plant-reliability-research"})
    with open_url(request) as response:
        return json.load(response)

def preserve_invalid(path):
    path = Path(path)
    candidate = path.with_name(path.name + ".invalid")
    suffix = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.invalid-{suffix}")
        suffix += 1
    path.replace(candidate)
    return candidate

def valid_zip(path):
    try:
        with zipfile.ZipFile(path) as zipped:
            return zipped.testzip() is None
    except (OSError, zipfile.BadZipFile):
        return False

def archive_relative_path(name):
    if PureWindowsPath(name).is_absolute() or PureWindowsPath(name).drive:
        raise ValueError("Unsafe archive path")
    parts = name.replace("\\", "/").split("/")
    if any(part == ".." for part in parts):
        raise ValueError("Unsafe archive path")
    parts = [part for part in parts if part and part != "."]
    if not parts:
        raise ValueError("Unsafe archive path")
    portable = []
    for part in parts:
        trimmed = part.rstrip(" .")
        suffix = part[len(trimmed):]
        component = "".join(
            f"%{ord(char):02X}" if ord(char) < 32 or char in '<>:"|?*' else char
            for char in trimmed)
        component += "".join(f"%{ord(char):02X}" for char in suffix)
        if trimmed.split(".", 1)[0].upper() in {
                "CON", "PRN", "AUX", "NUL",
                *(f"COM{index}" for index in range(1, 10)),
                *(f"LPT{index}" for index in range(1, 10))}:
            component = f"%{ord(trimmed[0]):02X}{component[1:]}"
        portable.append(component)
    return Path(*portable)

def legacy_archive_relative_path(name):
    parts = [part for part in name.replace("\\", "/").split("/") if part and part != "."]
    return Path(*(quote(part, safe=" -_.()[]{}!,;=@+~") for part in parts))

def fetch(url, target, expected_sha256=None, verify_zip=False):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and (expected_sha256 is None or digest(target) == expected_sha256) and (
            not verify_zip or valid_zip(target)):
        return
    if target.exists():
        preserve_invalid(target)
    temporary = target.with_suffix(target.suffix + ".partial")
    for attempt in range(4):
        offset = temporary.stat().st_size if temporary.exists() else 0
        headers = {"User-Agent": "plant-reliability-research"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with open_url(request) as response:
                status = getattr(response, "status", 200)
                mode = "wb"
                if status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", content_range)
                    if not match or int(match.group(1)) != offset or int(match.group(2)) < offset:
                        raise ValueError(f"Invalid partial-download response: {content_range!r}")
                    mode = "ab"
                    response_bytes = int(match.group(2)) - offset + 1
                else:
                    response_bytes = None
                with temporary.open(mode) as output:
                    shutil.copyfileobj(response, output)
                content_length = response.headers.get("Content-Length")
                if content_length is not None and temporary.stat().st_size - (offset if mode == "ab" else 0) != int(content_length):
                    raise ConnectionError("Download ended before the advertised content length")
                if response_bytes is not None and temporary.stat().st_size != offset + response_bytes:
                    raise ConnectionError("Partial download ended before the advertised byte range")
            if verify_zip and not valid_zip(temporary):
                preserve_invalid(temporary)
                raise ValueError(f"Downloaded archive failed ZIP integrity checks: {target}")
            if expected_sha256 and digest(temporary) != expected_sha256:
                preserve_invalid(temporary)
                raise ValueError(f"Downloaded file checksum mismatch: {target}")
            temporary.replace(target)
            return target
        except urllib.error.HTTPError as exc:
            if offset and exc.code == 416:
                preserve_invalid(temporary)
                if attempt == 3:
                    raise
                continue
            if attempt == 3:
                raise
            time.sleep(1 + attempt)
        except (http.client.IncompleteRead, urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            if attempt == 3:
                raise
            time.sleep(1 + attempt)

def safe_extract(archive, target, predicate=lambda name: True):
    target = Path(target).resolve()
    renamed = []
    with zipfile.ZipFile(archive) as zipped:
        selected = [m for m in zipped.infolist() if not m.is_dir() and predicate(m.filename)]
        by_destination = {}
        for member in selected:
            relative = archive_relative_path(member.filename)
            by_destination.setdefault(relative.as_posix().casefold(), []).append((member, relative))
        destinations = []
        for group in by_destination.values():
            for index, (member, relative) in enumerate(sorted(group, key=lambda pair: pair[0].filename)):
                if index:
                    suffix = hashlib.sha256(member.filename.encode("utf-8")).hexdigest()[:10]
                    relative = relative.with_name(f"{relative.stem}__{suffix}{relative.suffix}")
                normalized_name = member.filename.replace("\\", "/")
                if relative.as_posix() != normalized_name:
                    renamed.append({"archive_path": member.filename, "extracted_path": relative.as_posix()})
                destination = (target / relative).resolve()
                if not destination.is_relative_to(target):
                    raise ValueError("Unsafe archive path")
                destinations.append((member, destination))
        destination_names = [destination.relative_to(target).as_posix().casefold()
                             for _, destination in destinations]
        if len(destination_names) != len(set(destination_names)):
            raise ValueError("Archive paths collide after Windows filename normalization")
        expected_destinations = set(destination_names)
        def matches(member, destination):
            if not destination.is_file() or destination.stat().st_size != member.file_size:
                return False
            checksum = 0
            with destination.open("rb") as existing:
                for block in iter(lambda: existing.read(1024 * 1024), b""):
                    checksum = zlib.crc32(block, checksum)
            return checksum & 0xFFFFFFFF == member.CRC

        pending = [(member, destination) for member, destination in destinations
                   if not matches(member, destination)]
        target.mkdir(parents=True, exist_ok=True)
        required = sum(member.file_size for member, _ in pending) + 5 * 1024**3
        if shutil.disk_usage(target).free < required:
            raise ValueError("Insufficient space for extraction plus 5 GiB reserve")
        for member, destination in pending:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".partial")
            legacy = (target / legacy_archive_relative_path(member.filename)).resolve()
            if legacy != destination and matches(member, legacy):
                legacy.replace(temporary)
            else:
                with zipped.open(member) as source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output)
            temporary.replace(destination)
        for member, destination in destinations:
            legacy = (target / legacy_archive_relative_path(member.filename)).resolve()
            legacy_name = legacy.relative_to(target).as_posix().casefold()
            if (legacy != destination and legacy_name not in expected_destinations
                    and matches(member, legacy)):
                preserve_invalid(legacy)
    return renamed

def verify_full_inventory(plantvillage_root, plantdoc_root, declared_paths, expected_plantdoc_counts):
    plantvillage_images = [
        path for path in plantvillage_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    declared_plantvillage = {
        (Path(line).parent.name, Path(line).name)
        for split in declared_paths.values() for line in split
    }
    actual_plantvillage = [(path.parent.name, path.name) for path in plantvillage_images]
    if len(actual_plantvillage) != len(declared_plantvillage) or set(actual_plantvillage) != declared_plantvillage:
        raise ValueError("Extracted PlantVillage images do not match the pinned split inventory")

    actual_plantdoc_counts = Counter()
    for path in plantdoc_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            parts = path.relative_to(plantdoc_root).parts
            bucket = next((part for part in parts if part in {"train", "test"}), None)
            if bucket and len(parts) >= 3:
                actual_plantdoc_counts[f"{bucket}/{parts[-2]}"] += 1
    if dict(actual_plantdoc_counts) != expected_plantdoc_counts:
        raise ValueError("Extracted PlantDoc images do not match the pinned class inventory")
    return {
        "plantvillage_image_count": len(plantvillage_images),
        "plantdoc_image_count": sum(actual_plantdoc_counts.values()),
        "plantdoc_class_counts": dict(sorted(actual_plantdoc_counts.items())),
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Without this flag only inspect metadata")
    parser.add_argument("--subset", type=int, help="Images per class, two apple classes, upstream GitHub color images")
    parser.add_argument("--skip-metadata", action="store_true", help="Smoke subset only: skip blocked HF raw metadata; record filename leaf-ID fallback")
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    if args.skip_metadata and not args.subset:
        parser.error("--skip-metadata is only supported with --subset")
    root = managed_path(args.data_dir) if args.data_dir else paths("smoke" if args.subset else "research")["data_dir"]
    configure_runtime(create=True)
    root.mkdir(parents=True, exist_ok=True)
    if args.download and not args.subset:
        minimum = read_json(ROOT / "research/configs/storage.json")["minimum_free_gib"]
        if shutil.disk_usage(root).free < minimum * 1024**3:
            raise ValueError(f"Full acquisition requires {minimum} GiB free before downloading")
    previous_path = root / "raw" / "sources.json"
    previous = read_json(previous_path) if previous_path.exists() else None
    if args.download and previous and bool(previous.get("subset_per_class")) != bool(args.subset):
        raise ValueError("Do not mix full and subset acquisitions. Set DATA_DIR to a separate directory.")
    pinned = read_json(ROOT / "research/configs/sources.json")
    hf = {"sha": pinned["plantvillage"]}
    pd = {"sha": pinned["plantdoc"]}
    metadata = {"plantvillage": {"source": "https://huggingface.co/datasets/mohanty/PlantVillage", "revision": hf["sha"], "license": "CC-BY-SA-3.0"},
                "plantdoc": {"source": "https://github.com/pratikkayal/PlantDoc-Dataset", "revision": pd["sha"], "license": "CC-BY-4.0"}, "subset_per_class": args.subset}
    if args.download and previous and any(previous.get(key, {}).get("revision") != metadata[key]["revision"] for key in ["plantvillage", "plantdoc"]):
        raise ValueError("Dataset source revision changed. Use a fresh DATA_DIR to preserve provenance.")
    if args.download and not args.subset:
        save_json(previous_path, {**(previous or {}), **metadata, "acquisition_complete": False})
    base = f"https://huggingface.co/datasets/mohanty/PlantVillage/raw/{hf['sha']}"
    inventory = read_json(ROOT / "research/configs/source_inventory.json")
    for name in ([] if args.skip_metadata else ["README.md", "leaf_grouping/leaf-map.json", "splits/color_train.txt", "splits/color_test.txt"]):
        target = root / "raw" / "metadata" / name
        fetch(f"{base}/{name}", target)
        oid = git_blob_digest(target)
        if oid != inventory["plantvillage_metadata_git_oids"][name]:
            raise ValueError(f"Pinned metadata checksum mismatch: {target}")
    counts = {}
    for split in ([] if args.skip_metadata else ["train", "test"]):
        lines = (root / "raw" / "metadata" / "splits" / f"color_{split}.txt").read_text().splitlines()
        counts[split] = dict(Counter(Path(line).parent.name for line in lines if line))
    metadata["plantvillage"]["declared_color_split_counts"] = counts
    metadata["plantvillage"]["metadata_skipped"] = args.skip_metadata
    save_json(root / "raw" / "inspected_sources.json", metadata)
    print(json.dumps(metadata, indent=2))
    if not args.download:
        print("Metadata inspected only. Add --download --subset 12 for a small two-class smoke sample; omit --subset for full data.")
        return
    if args.subset:
        if args.subset < 3:
            raise ValueError("Use at least three images per class")
        pv = {"sha": pinned["plantvillage_subset"]}
        if previous and previous.get("plantvillage", {}).get("subset_upstream_revision") != pv["sha"]:
            raise ValueError("Upstream subset revision changed. Use a fresh DATA_DIR.")
        metadata["plantvillage"]["subset_upstream_revision"] = pv["sha"]
        metadata["plantvillage"]["subset_upstream_source"] = "https://github.com/spMohanty/PlantVillage-Dataset"
        sources = [("spMohanty/PlantVillage-Dataset", pv["sha"], "plantvillage", ["Apple___Apple_scab", "Apple___healthy"], "raw/color/"),
                   ("pratikkayal/PlantDoc-Dataset", pd["sha"], "plantdoc", ["Apple Scab Leaf", "Apple leaf"], "test/")]
        downloaded = []
        for repo, sha, dataset, labels, prefix in sources:
            for label in labels:
                listing = fetch_json(f"https://api.github.com/repos/{repo}/contents/" + urllib.parse.quote(prefix+label) + f"?ref={sha}")
                files = [r for r in listing if r["type"] == "file" and Path(r["path"]).suffix.lower() in {".jpg", ".png", ".jpeg"}]
                random.Random(42).shuffle(files)
                for entry in files[:args.subset]:
                    path = entry["path"]
                    target = root / "raw" / dataset / Path(path)
                    if not target.exists():
                        blob = fetch_json(entry["git_url"])
                        if blob.get("encoding") != "base64":
                            raise ValueError("Unsupported GitHub blob encoding")
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(base64.b64decode(blob["content"]))
                    downloaded.append({"dataset": dataset, "source_path": path, "sha256": digest(target)})
        metadata["files"] = downloaded
    else:
        archive = root / "raw" / "downloads" / "data.zip"
        # Reuse a completed direct download; the pinned checksum below remains mandatory.
        if archive.exists() and digest(archive) != inventory["plantvillage_archive_sha256"]:
            preserve_invalid(archive)
        if not archive.exists():
            from huggingface_hub import hf_hub_download
            archive = hf_hub_download("mohanty/PlantVillage", "data.zip", repo_type="dataset", revision=hf["sha"], local_dir=root / "raw" / "downloads")
        if digest(archive) != inventory["plantvillage_archive_sha256"]:
            raise ValueError("PlantVillage archive checksum does not match pinned source")
        safe_extract(archive, root / "raw" / "plantvillage", lambda name: "raw/color/" in name)
        pd_archive = root / "raw" / "downloads" / "plantdoc.zip"
        fetch(f"https://codeload.github.com/pratikkayal/PlantDoc-Dataset/zip/{pd['sha']}", pd_archive, verify_zip=True)
        with zipfile.ZipFile(pd_archive) as zipped:
            archive_roots = {PureWindowsPath(name).parts[0] for name in zipped.namelist() if name}
        expected_root = f"PlantDoc-Dataset-{pd['sha']}"
        if archive_roots != {expected_root}:
            raise ValueError("PlantDoc archive root does not match the pinned revision")
        metadata["plantdoc"]["windows_filename_escapes"] = safe_extract(
            pd_archive, root / "raw" / "plantdoc",
            lambda name: bool({"train", "test"} & set(PureWindowsPath(name).parts))
            or PureWindowsPath(name).name in {"LICENSE.txt", "README.md"})
        metadata["archive_sha256"] = {"plantvillage": digest(archive), "plantdoc": digest(pd_archive)}
        with zipfile.ZipFile(pd_archive) as zipped:
            archived_plantdoc_counts = Counter()
            for member in zipped.infolist():
                if member.is_dir() or Path(member.filename).suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                parts = PureWindowsPath(member.filename).parts
                bucket = next((part for part in parts if part in {"train", "test"}), None)
                if bucket:
                    archived_plantdoc_counts[f"{bucket}/{parts[-2]}"] += 1
        if dict(archived_plantdoc_counts) != inventory["plantdoc_image_counts"]:
            raise ValueError("Pinned PlantDoc archive inventory does not match the source inventory")
        declared_paths = {}
        for split in ["train", "test"]:
            path = root / "raw" / "metadata" / "splits" / f"color_{split}.txt"
            declared_paths[split] = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        metadata["verified_inventory"] = verify_full_inventory(
            root / "raw" / "plantvillage", root / "raw" / "plantdoc",
            declared_paths, inventory["plantdoc_image_counts"])
    metadata["acquisition_complete"] = True
    save_json(root / "raw" / "sources.json", metadata)
    print("Download complete. Run prepare_data next to measure actual decoded counts.")

if __name__ == "__main__":
    main()
