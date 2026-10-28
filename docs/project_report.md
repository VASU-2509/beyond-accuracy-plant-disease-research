# Beyond Accuracy: Evaluating Domain Generalization and Uncertainty Calibration in Deep Learning-Based Plant Disease Recognition

**Vasu Mehrotra and Ashutosh Bansal**  
College deep learning research project · Draft report · 27 October 2026

## Abstract

This project defines and implements a reproducible evaluation of plant-disease classifiers under controlled-to-external image shift. ResNet-18 and MobileNetV3-Small are trained using PlantVillage color images and evaluated on source test images and defensibly mapped PlantDoc classes. The protocol measures classification, confidence calibration, confidence-based abstention and local inference efficiency. Temperature and rejection thresholds are fitted using source validation data only. A FastAPI service and Next.js dashboard expose saved experimental artifacts and checkpoint predictions. Full research findings are pending execution of the final experimental protocol; software smoke outputs are not evidence of agricultural diagnostic performance.

**Keywords:** plant disease recognition; domain shift; domain generalization evaluation; temperature scaling; confidence calibration; selective prediction; reproducibility.

## 1. Introduction

Controlled visual datasets make model development tractable, but deployment images differ in background, illumination, crop composition and symptom presentation. A classifier can be accurate within its training distribution and unreliable elsewhere. This study considers confidence quality and abstention alongside classification and efficiency.

## 2. Problem statement

Determine whether two practical classifiers trained only on controlled images retain classification and confidence reliability on an external image corpus. Separate empirical generalization from claims of reliable real-world diagnosis.

## 3. Research questions

RQ1: What is the source-to-external performance difference on shared classes? RQ2: How does validation-only temperature scaling affect each domain's ECE and Brier score? RQ3: What coverage and accepted error result from a frozen validation threshold? RQ4: Which accuracy/reliability/efficiency trade-offs are observed between architectures?

## 4. Related work

PlantVillage recognition [1], PlantDoc [2], modern network calibration [3], selective classification [4], agricultural domain adaptation [5], and directly related cross-domain reliability benchmarking [6] motivate the study. See [literature_review.md](literature_review.md) for distinctions in data access and evaluation settings, and [references.md](references.md) for verified bibliographic records.

## 5. Research gap and contribution

The general topic is established. The proposed contribution is a transparent, reproducible comparison of two selected architectures on locally available hardware with an artifact-backed demonstration interface. This project does not introduce a new calibration or generalization algorithm. A comparative empirical conclusion requires completed research runs and variability analysis.

## 6. Dataset descriptions

PlantVillage supplies the controlled source images; only color images are used. PlantDoc's cropped classification images form the external corpus. Actual revision, counts, class distributions, corrupt files, duplicate groups and label exclusions must be taken from the completed run's audit and manifest, not copied from publication headline counts.

| Corpus | Pinned revision | Pinned image files / audited images | Classes | Corrupt | Duplicate candidate groups (pixel / dHash) | Role |
|---|---|---:|---:|---:|---:|---|
| PlantVillage color | `9e97599868962bd0079b8db4b7f1efa9185fa1e7` | 54,305 / 54,305 | 38 | 0 | 21 / 23 | Source training, validation and test |
| Cropped PlantDoc | `5467f6012d78d1c446145d5f582da6096f852ae8` | 2,578 / 2,578 | 28 folder labels | 0 | 17 / 47 | External evaluation only |

PlantVillage's pinned split lists contain 43,596 training and 10,709 test paths; the experiment uses its own group-aware split across all 54,305 color images. PlantDoc's two supplied folders both remain external-only. After frozen label mapping and duplicate/overlap screening, 2,107 PlantDoc images in 23 mapped classes are eligible; 471 are excluded (432 because their folder labels are not safely mappable and 39 as duplicate external images). No corrupted images or source/external hash overlaps were observed. Per-class counts and exclusion reasons are in [dataset_mapping.md](dataset_mapping.md) and the saved audit reports.

The official PlantVillage leaf map contains 40,328 keys matching 41,111 image rows. The remaining 13,194 rows use the prespecified, explicitly recorded class-scoped filename fallback. A unique upstream mapping is retained even when a legacy source class alias differs; 621 such unique-map mismatches belong to Apple Black Rot files labeled with the legacy `Apple_Frogeye Spot` map alias. Group IDs do not change disease labels. Filename fallback can group repeated suffixes, but cannot prove physical-leaf identity. Tiny two-class or synthetic datasets remain software fixtures only.

## 7. Methodology

Use approximately 70/15/15 group-aware source train/validation/test splits with fixed seeds. Group repeated source leaves and matching image hashes before splitting. Train both architectures using the same manifests and preprocessing policy; select lowest validation-loss checkpoints. Retain the full classifier vocabulary during external evaluation and compute descriptive domain gaps using the shared-class source test subset.

Fit a bounded positive temperature by validation NLL minimization. Select the maximum-coverage validation threshold with empirical risk <=0.10 and coverage >=0.20; abstain on all when infeasible. Keep external test data out of fitting. The detailed executable protocol and its limitations are in [methodology.md](methodology.md).

## 8. Experimental setup

Readiness inspection on 27 October 2026: the pinned full archives, extracted inventories, zero-corrupt image audits, and three fixed-split manifests are complete. Each manifest has 37,863 train, 8,256 validation, 8,186 in-domain test and 2,107 external images; all three have identical membership and source hashes. D: had 189.13 GiB free after acquisition, caches and preparation. A saved `full_seed42` ResNet-18 checkpoint has a six-epoch training history with best epoch 5; no matching final evaluation was present when this demo was prepared. The other prescribed model/seed runs remain incomplete. The current `.venv` is CPU-only. The complete prescribed suite was estimated at 228 compute hours before overhead; this planning estimate is not a research result. See [execution plan](execution_plan.md), [pinned file inventories](preflight_sources.json), and [label mapping](dataset_mapping.md).

Local VS Code environment, Python 3.11, PyTorch/Torchvision, FastAPI and Next.js. The inspected machine has an Intel Core i5-11400H, 8,358,518,784 bytes RAM and a GTX 1650 with 4 GB VRAM; the current installed PyTorch 2.6.0+cpu/Torchvision 0.21.0+cpu pair cannot use CUDA. ResNet-18 and MobileNetV3-Small ImageNet `IMAGENET1K_V1` weights are cached, SHA-256 verified and confirmed loaded by the model factory. Full configuration: 224-pixel inputs, batch size 8, learning rate 0.0001, maximum 30 epochs, patience 5, training seeds 42/43/44 with fixed split seed 42, required ImageNet initialization. These settings are preserved; the complete full training/evaluation suite remains incomplete.

Save exact installed packages, device, checkpoint hash, split hash, configuration and history with each run. Prespecify additional seeds before inspecting test results. Report any hardware-dependent configuration change.

## 9. Metrics

Accuracy; balanced accuracy; macro precision, recall and F1; per-class reports and confusion matrices; ECE with 15 equal-width bins; multiclass summed Brier score; coverage, accepted accuracy and selective risk. Report raw and calibrated values separately. Efficiency includes total parameter count, serialized checkpoint MiB, median/p95 forward latency after warmup. Calibration preserves argmax, so it does not itself change accuracy.

## 10. Results

**[PENDING FULL EXPERIMENTS — DO NOT REPLACE WITH SMOKE METRICS]**

| Model | Common-class source accuracy/F1 | External accuracy/F1 | Raw/scaled ECE | Raw/scaled Brier | Coverage/risk | Checkpoint/latency |
|---|---|---|---|---|---|---|
| ResNet-18 | Pending | Pending | Pending | Pending | Pending | Pending |
| MobileNetV3-Small | Pending | Pending | Pending | Pending | Pending | Pending |

Insert figures generated under `artifacts/plots/<full-run>/`: class distribution, learning curves, confusion matrices, domain comparison, reliability, confidence histograms, risk-coverage and efficiency. Attribute every table to a run ID/configuration, image count and class set. Include failed runs, unavailable metrics and zero-coverage cases explicitly. Do not infer final model rankings from tiny smoke runs.

## 11. Discussion

**[PENDING OBSERVED RESULTS]** Discuss whether domain gaps persist on common classes, whether calibration transfers, and whether confidence ranking remains useful for rejection. Distinguish classification changes from confidence changes. Interpret latency only within the recorded measurement protocol.

## 12. Threats to validity

Residual related-image leakage may survive perceptual hashing; over-grouping can remove legitimate examples. Filename leaf-ID fallbacks can miss repeated leaves. Dataset class priors differ. Label compatibility is semantic and not independently adjudicated. The shared-class gap changes the source evaluation population. Validation reuse for selection/calibration/thresholding can overfit. Hardware/package variation affects timing and reproducibility.

## 13. Limitations

Single-source training and one external corpus cannot establish general farm deployment performance. Cropped internet images are not a prospective diagnosis study. Softmax confidence is not a calibrated correctness guarantee under arbitrary shift. Single-seed results do not establish statistical significance. This project offers no disease treatment recommendations.

## 14. Conclusion and future work

The software and full data preparation implement the proposed study and separate saved evidence from missing results. No model superiority or external-reliability conclusion is asserted before final experiments. Next steps are the preserved prespecified six-run training/evaluation suite on suitable hardware, followed by stronger image-similarity checks and prospective external validation.

## 15. References

See the complete verified bibliography in [references.md](references.md). All numerical experimental claims in the final report must originate in this project's saved outputs.
