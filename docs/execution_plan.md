# Full experiment execution plan

Updated 28 October 2026 (Asia/Kolkata). **Full acquisition, checksum/archive verification, image audit and fixed-split preparation are complete. One ResNet-18 `full_seed42` checkpoint is saved (best epoch 5, six epochs recorded); the prescribed six-run suite and final research evaluation are incomplete.** The existing application and its smoke/final output separation have been preserved.

## Storage and hardware

Measured before acquisition: D: 195.14 GiB available; C: 4.76 GiB available. After archives, extracted images, caches, audit reports and manifests, D: had 189.13 GiB available. Source, the existing Python environment and frontend dependencies remain in the VS Code workspace.

`research/configs/storage.json` is the single default storage setting: `D:/plant-disease-research-data`. Set `RESEARCH_STORAGE_ROOT` to another absolute non-C: root to relocate it. DATA_DIR and ARTIFACTS_DIR overrides must remain inside that root. Python research commands configure Torch, Hugging Face hub/datasets/Xet, matplotlib, pip cache, and temporary paths before library use; inherited cache locations are overridden. Xet is disabled to avoid a redundant chunk cache. No persistent system environment changes were made. Pip commands outside the research process must explicitly set PIP_CACHE_DIR and TEMP/TMP to directories on D:.

| Item | Budget | Basis |
|---|---:|---|
| PlantVillage downloaded archive | 2.03 GiB | Pinned file listing: 2,184,723,441 bytes; contains more than the selected color images |
| Extracted PlantVillage color images | 54,305 images; 38 classes | Extracted inventory matched both pinned color split lists |
| PlantDoc archive plus extracted images | Archive 984,592,720 bytes; 2,578 images | Image blobs total 996,322,499 bytes; both folders are external-only |
| ImageNet initialization weights | 57,137,122 bytes total | Both pinned files are cached, SHA-256 verified and loaded |
| Six best checkpoints, metrics/logits, manifests, audits, figures and logs | 1–2 GiB allowance | Best checkpoints only, no processed image copies; transient atomic checkpoint copy included |
| Temporary downloads, retries, caches and contingency | 5–8 GiB allowance | Archives retained for provenance; peak usage exceeds extracted dataset size |
| **Planned working budget** | **15–20 GiB** | Acquisition requires at least **30 GiB free**, extraction checks member sizes plus 5 GiB reserve |

D: has adequate headroom. The risks are network resets, unmeasured extraction sizes, and CPU duration. C: has insufficient headroom for moving large dependencies or caches back there. A future separate CUDA environment would need an additional approximately 8–12 GiB allowance on D:, subject to actual package resolution; no CUDA installation is included in these commands.

Hardware: Intel i5-11400H, 12 logical processors, 8,358,518,784 bytes RAM (7.78 GiB); NVIDIA GTX 1650, 4,096 MiB VRAM, driver 610.62. Installed PyTorch is **2.6.0+cpu**, Torchvision 0.21.0+cpu; CUDA is unavailable to this runtime. The study runs sequentially, with batch size 8, zero loader workers and two Torch CPU threads.

The short random-tensor timing probe used 224×224 inputs, 38 output classes, batch 8, two warmups and eight measured training/evaluation steps per model. It downloaded nothing and produced no checkpoints or accuracy metrics. Observed training batch times: ResNet-18 1.351 s; MobileNetV3-Small 0.437 s. Estimated compute-only epochs: 1.93 h and 0.60 h. Three seeds × 30 epochs × both models totals **about 228 compute hours (9.5 days)**. Budget **12–19 days** including image decoding, augmentation, evaluation and laptop thermal/power effects. Early stopping may shorten this. Actual sustained throughput may change it substantially. GPU runtime is unmeasured and must not be inferred from these CPU numbers. Raw timing record: `preflight_hardware.json` and the D: `preflight/` directory.

## What is verified, and what remains pending

Pinned source versions are in `research/configs/sources.json`:

- PlantVillage Hugging Face: `9e97599868962bd0079b8db4b7f1efa9185fa1e7`.
- Cropped PlantDoc classification repository: `5467f6012d78d1c446145d5f582da6096f852ae8`.
- Historical smoke PlantVillage images came from upstream GitHub commit `7f7ecc7e1eaca78107e3affe7cb5abd9427e139a`; they are not the full archive.

The [PlantVillage card](https://huggingface.co/datasets/mohanty/PlantVillage) declares 43,596 color training paths and 10,709 test paths: **54,305**, differing by one from its 54,306 headline count. Both pinned lists and the extracted paths match exactly. All 54,305 images decode across 38 classes; none are corrupt. The image audit found 21 pixel-identical and 23 dHash candidate duplicate groups. The official leaf map has 40,328 keys matching 41,111 image rows; 13,194 rows require the recorded filename fallback across 15 classes. A unique upstream mapping is retained for 621 legacy Apple Black Rot entries whose official map alias is `Apple_Frogeye Spot`; this is grouping metadata only and does not change labels.

The [pinned PlantDoc tree](https://api.github.com/repos/pratikkayal/PlantDoc-Dataset/git/trees/5467f6012d78d1c446145d5f582da6096f852ae8?recursive=1) was not truncated: **2,578 image files**, 2,342 under train and 236 under test, with 28 distinct folder labels in their union. This differs from the repository's publication abstract count of 2,598. All 2,578 images decode; none are corrupt. Both folders remain external-only; this is a combined external-corpus protocol, not a claim to reproduce the published PlantDoc test benchmark. Folder labels give 2,146 potentially eligible images in 23 mapped classes. The frozen mapping and duplicate screening leave **2,107 eligible external images in 23 classes**; the other 471 include 432 unmappable-label exclusions and 39 duplicate external images. No source/external hash overlap was detected. Exact counts are in `preflight_sources.json`, `sources.json` and the saved audits.

The local real-image smoke sample was decoded and hashed again: PlantVillage 12 images (6 scab, 6 healthy), PlantDoc 12 (6 scab, 6 healthy), no corrupt images or equal pixel/dHash duplicates. Its saved manifest has 8/2/2 source train/validation/test images plus 12 external images; path/pixel/dHash/leaf overlap checks passed. Both saved training records match the original manifest hash. All 12 source leaves used filename fallback identifiers, so this is not sufficient evidence for a final leaf-disjoint study. See `preflight_local_audit.json`. Existing smoke files were copied to D: byte-for-byte; historical manifests still contain their original C: source references, preserving checkpoint hashes. New preparations use D: paths.

Full preparation checked observed source filenames and class folders against the official lists, checked PlantDoc per-folder counts, recorded SHA-256/pixel/dHash values, grouped matching leaves/hashes before splitting, and excluded source/external overlap and repeated external hashes. No source conflicts or corrupt files were found. The 3 manifests share the fixed split seed 42 and identical membership: **37,863 train, 8,256 validation, 8,186 in-domain test, and 2,107 external**. The leakage check passed path containment, label vocabulary, and pairwise path/SHA-256/pixel/dHash/source-leaf overlap checks; zero source rows were quarantined. Its `source_bytes_verified` field remains false because preparation does not reread every source file; the research train/evaluation stages recompute and verify these hashes before model use. Research manifests explicitly record the 13,194 filename fallbacks rather than implying complete official leaf coverage. Equal dHash is a conservative candidate test, not an exhaustive near-duplicate search.

## Prespecified study

Six training runs: ResNet-18 and MobileNetV3-Small for seeds **42, 43, 44**. Fixed **split seed 42** keeps the same group-aware source split across every run. The verified source manifests contain 37,863 train, 8,256 validation and 8,186 in-domain test images. Report training-seed variability on these fixed holdouts; do not describe it as variability across independently sampled datasets.

Both architectures use 224-pixel ImageNet-normalized inputs, the existing identical augmentation policy, fine-tuning, AdamW LR 0.0001, batch 8, maximum 30 epochs, validation-loss scheduling and early stopping patience 5. ImageNet `IMAGENET1K_V1` weights are pinned explicitly, cached and loaded. SHA-256 values are ResNet-18 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` and MobileNetV3-Small `047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f`. Random fallback is prohibited for research. Full-suite training requires these matching local weight records and does not silently continue with random initialization.

Only PlantVillage training images enter gradient updates; only PlantVillage validation images enter checkpoint selection, temperature fitting and threshold selection. Neither source test nor **any PlantDoc image** is used for tuning. The same validation set serves three selection roles, which remains a disclosed limitation. Test performance must not determine changes to epochs, learning rate, mapping or thresholds.

Evaluation retains the full classifier vocabulary; save full-source, common-class source, and external metrics. The suite writes resolved per-seed configurations, source/checkpoint hashes, logs and hardware status, best checkpoints, raw logits, predictions, per-seed metrics and PNG/PDF figures. Aggregation verifies six matching completed runs and identical split membership, then writes per-seed values, mean and sample SD with the count of defined observations. Undefined selective risks stay null. There is no significance claim or automatic population confidence interval.

Existing research manifests, checkpoints, evaluation records and aggregates cannot be overwritten. A failed partial training run requires a fresh run ID/configuration (change `study_id` in the full YAML for a fresh suite) and deliberate recovery; the suite does not resume optimizer state. All training finishes before any held-out evaluation begins.

## Exact commands

Run from the existing repository root in PowerShell. Full acquisition, weight verification, preflight refresh and three-manifest preparation have completed. The acquisition/preparation commands below are recorded for reproducibility; do not rerun them unless intentionally revalidating the data.

```powershell
Set-Location C:\Users\Acer\Desktop\plant-disease-research
$env:RESEARCH_STORAGE_ROOT = 'D:\plant-disease-research-data'
# Clear old overrides so the configured full/smoke defaults are used.
Remove-Item Env:DATA_DIR, Env:ARTIFACTS_DIR -ErrorAction SilentlyContinue
$py = '.\.venv\Scripts\python.exe'
& $py -m research.scripts.preflight
& $py -m research.scripts.full_suite plan
# Completed acquisition and preparation commands; do not repeat just to resume training.
& $py -m research.scripts.download_data --download
& $py -m research.scripts.cache_weights
& $py -m research.scripts.full_suite prepare --config research/configs/full_experiment.yaml
```

Preparation outputs are under `D:\plant-disease-research-data\full\data` and `...\full\artifacts`. The 13,194 explicit filename fallbacks are a documented limitation, not missing data silently treated as official IDs. Review `D:\plant-disease-research-data\full\artifacts\reports\full_seed42\preparation_review.json`, both audits, the `sources.json` inventory, and all three summaries before making scientific claims. The verified source and external images currently have no corrupt rows; 39 duplicate external images and 432 semantically unmappable labels are excluded.

The prescribed six-run training/evaluation suite is **incomplete**: its original schedule is estimated at about 228 compute hours on the measured CPU before overhead. One saved ResNet-18 checkpoint is reused by the separately labelled teacher demo; this does not complete or replace the original 2-model × 3-seed, 30-epoch design. Once suitable hardware and sufficient time are available, continue that unchanged design sequentially and stop on any error:

```powershell
& $py -m research.scripts.full_suite train
& $py -m research.scripts.full_suite evaluate
& $py -m research.scripts.full_suite aggregate
```

A CPU run carries the 12–19 day budget. If CUDA support is desired, prepare a separate D: environment and measure it before revising the schedule; do not replace the working C: environment or guess GPU timing.

The backend defaults to the full D: artifact directory for research. The teacher demo has a separate launch recipe in [demo_guide.md](demo_guide.md); smoke/synthetic outputs remain separate. Existing purpose badges identify non-research evidence. Output trees are separate and no placeholder metrics were created.

Source report results remain pending. Only verified acquisition/audit facts and measured hardware timing have been added to documentation. Full findings will be written after the approved experiments and aggregation.
