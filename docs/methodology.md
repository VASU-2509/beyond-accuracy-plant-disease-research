# Methodology and preregistered operating rules

Project: **Beyond Accuracy: Evaluating Domain Generalization and Uncertainty Calibration in Deep Learning-Based Plant Disease Recognition**. Authors: Vasu Mehrotra and Ashutosh Bansal.

## Research questions

1. How do accuracy and macro F1 change from controlled PlantVillage images to eligible PlantDoc images?
2. Does temperature scaling fitted on source validation improve ECE and Brier score on each test domain?
3. Does a validation-selected abstention threshold transfer to external images, and at what coverage?
4. How do ResNet-18 and MobileNetV3-Small compare in reliability, parameter count, storage and latency?

This is a single-source generalization evaluation, not a new domain-generalization training algorithm. No target-domain adaptation is performed.

## Sources and acquisition

[PlantVillage on Hugging Face](https://huggingface.co/datasets/mohanty/PlantVillage) contains an archive plus split text files and leaf-group metadata. Its dataset card lists color, grayscale and segmented variants; the viewer can expose paths as text. The pipeline downloads the archive directly and selects color images instead of assuming `load_dataset` returns decoded images. The card's color split counts (43,596 and 10,709) do not exactly match its headline 54,306 count; these are source-declared statistics, not this project's audited counts. Runtime inventory is authoritative for each run.

The tiny acquisition mode samples two apple classes from the [upstream repository linked by the card](https://github.com/spMohanty/PlantVillage-Dataset), avoiding the multi-gigabyte archive. This is explicitly a smoke sample, not a representative benchmark. It saves the upstream commit and hashes. Full mode saves the Hugging Face revision and archive checksum. Metadata and source licenses accompany the acquisition record.

[Cropped PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset) is the classification dataset, not the object-detection annotation repository. Full mode treats both provided folders as an external-only corpus; subset mode uses the provided test folder. No PlantDoc image or label participates in optimization, checkpoint selection, temperature fitting or threshold selection. Class-name alignment and duplicate auditing define eligibility, not hyperparameters. Its images are internet sourced and should be described as field-like, not a prospective field trial.

PlantVillage's card specifies CC BY-SA 3.0; PlantDoc's repository specifies CC BY 4.0. Retain attribution and consult the linked licenses before redistribution. PlantDoc's repository license does not independently establish the provenance of every original internet image. Do not commit raw images or publish an image mirror as part of this project.

## Audit, processing and split rules

Images are decoded with Pillow, EXIF orientation is applied and RGB content is hashed. Reports include dimensions, formats, corrupt paths, observed class counts and exact pixel/perceptual-hash duplicate lists. Equal 64-bit difference hashes are conservative near-duplicate candidates. This does not detect every crop, rotation or re-encoding. Hash collisions can over-group distinct images and should be reviewed in final data audits.

PlantVillage leaf IDs are resolved from the source leaf map where available; filename-derived fallback IDs are counted and reported. Connected components link matching pixel hashes, difference hashes or leaf IDs. Groups with conflicting class labels are quarantined. Per-class groups are deterministically shuffled (seed 42), then approximately 15% test, 15% validation and the remainder train. Every retained class needs at least three groups. Group size makes image-level fractions approximate. This custom split is not the source's published benchmark split.

The quick subset caps groups per class, never divides a group. Raw images are referenced from saved manifests; `data/processed/<run>/dataset_summary.json` stores processing metadata rather than redundant image copies. Training-time transformations occur on demand. Every selected image appears in exactly one source split. External images sharing a pixel or perceptual hash with any audited PlantVillage image are excluded; repeated external hashes are deduplicated. Full acquisition can therefore lose samples relative to published counts, and exclusions must be reported.

## Explicit label alignment

The executable mapping is `research/src/label_mapping.py`; it is copied into each manifest and result. Exact crop/disease correspondences include apple scab, cedar apple rust, potato early/late blight, tomato early/late blight, Septoria, bacterial spot and mosaic virus, corn gray leaf spot/common rust, grape black rot, squash powdery mildew, and named healthy classes.

Healthy labels are interpreted as the dataset's healthy category, not independently verified health. `Corn leaf blight`, `Bell_pepper leaf spot`, `Tomato mold leaf`, and `Tomato leaf yellow virus` are excluded because broad symptom wording does not establish a unique compatible etiology. The spider-mite label is excluded pending manual review. Missing training classes and all unknown names are excluded rather than fuzzy matched. The manifest lists each exclusion and reason. Do not revise mappings in response to model accuracy on the test set.

## Models and model selection

The prepared full suite prespecifies training seeds 42, 43 and 44 while holding split seed 42 fixed. Both models therefore see identical source partitions across seeds. Aggregate means and sample standard deviations describe training randomness conditional on these data; they do not estimate dataset sampling uncertainty. Final research prohibits random initialization/fallback and blocks incomplete source inventories or unresolved official leaf IDs. See [execution plan](execution_plan.md) and [frozen mapping](dataset_mapping.md).

Both Torchvision architectures replace their final linear layer with the observed class count. Full mode requests ImageNet pretrained weights; unavailable weights cause a clear failure unless random fallback is explicitly enabled. The checkpoint records whether pretraining actually loaded. Quick and synthetic smoke modes deliberately use random initialization and one epoch.

RGB inputs are resized to the configured square size, converted to tensors, and normalized with ImageNet mean/std. Training adds horizontal flips, rotations up to 15 degrees, and mild color jitter. Validation, test and inference use the same deterministic transformation. Square resizing may distort aspect ratios and is a stated limitation.

AdamW, validation-loss ReduceLROnPlateau scheduling and configurable early stopping are used. The checkpoint with lowest validation loss is saved. Fine-tuning trains all weights. Feature extraction freezes backbone gradients and batch-normalization running statistics. A singleton last training batch is dropped to avoid BatchNorm failure; test and validation samples are never dropped. Seeds and package/hardware metadata are recorded. Deterministic cuDNN settings improve repeatability but do not promise bitwise reproducibility across devices and package versions.

## Evaluation and comparable domain gaps

Report full-source in-domain performance separately from the common-class source subset. External outputs retain the full training-class softmax and argmax: predictions outside the mapped subset remain errors. Do not renormalize away such predictions. The domain gap equals common-class source metric minus external metric, with accuracy optionally expressed in percentage points. Macro precision/recall/F1 average over stated eligible ground-truth classes; confusion matrices and per-class reports retain the full classifier vocabulary. Balanced accuracy is mean recall over the evaluated labels.

Different class priors remain a confound even on shared classes. The domain gap is a descriptive difference, not a causal estimate of background shift.

## Calibration and selective prediction

Temperature minimizes mean negative log likelihood of **validation logits only**. Optimization bounds log(T) to [-4,4]; a positive scalar preserves argmax. Save temperature, validation NLL before/after and checkpoint/manifest checksums. Evaluate raw and scaled probabilities separately. ECE uses 15 equal-width confidence bins, weighted by bin support; confidence exactly one belongs to the last bin. Empty bins have null statistics. Multiclass Brier is mean over images of the sum of squared class-probability errors (not divided by class count).

The operational threshold is selected on calibrated validation probabilities: maximize empirical coverage subject to error among accepted predictions <=0.10 and coverage >=0.20. Enumerate unique confidence cutoffs so ties are accepted together. If no cutoff qualifies, save 1.000001 and abstain on all images. At zero coverage, accepted accuracy/risk are null. This empirical heuristic is **not** the finite-sample risk guarantee of selective-classification research.

The same validation set selects checkpoints, fits temperature and selects threshold; this reuse can overfit. A stronger follow-up should allocate separate model-selection/calibration/threshold subsets or nested cross-validation before test inspection. Never replace a threshold based on either test set. Test coverage-risk curves are diagnostic, not tuning tools.

## Efficiency and outputs

Count all parameters and measure actual checkpoint bytes. Forward-only latency uses batch size one, five warmups, 30 repeats, two CPU threads and CUDA synchronization around timings. Save median/p95 and hardware. It excludes decoding, preprocessing and HTTP; it is not user-visible end-to-end latency. Compare measurements on the same device and image size.

Each run saves split/configuration provenance, training history, validation/test logits, per-image prediction CSVs, JSON metrics, comparison CSV, PNG/PDF figures and checkpoint/calibration sidecars. Checkpoint loading uses `weights_only=True`; serve only trusted locally generated weights. Re-evaluation rejects changed split manifests. Use new run IDs to preserve independent experimental evidence.

No significance claim is justified by a single seed. Final reporting should repeat prespecified seeds, quantify variability and avoid selecting favorable runs. Synthetic texture and tiny real-data smoke outputs verify execution only and must not appear as final research findings.
