# Implementation and validation record

## Teacher demo — 28 October 2026

The existing `full_seed42` ResNet-18 checkpoint was verified against its original manifest and tested through the real-image FastAPI upload endpoint before reuse. It is an ImageNet-initialized 38-class, 224×224 checkpoint with a six-epoch history and best epoch 5. The current Python environment is CPU-only. No model was retrained and no original research checkpoint, fixed manifest, configuration, or result was overwritten.

A separate fixed balanced demo manifest selected 20 train-reference, 20 validation and 10 held-out test images per PlantVillage class. Evaluation reused the existing model, fitted temperature and an abstention rule on demo validation rows only, and evaluated up to 8 PlantDoc images per eligible class separately. Saved results: **98.16% accuracy on 380 PlantVillage demo test images** and **9.24% on 184 PlantDoc external images across 23 mapped classes**. Validation selected a zero threshold accepting all images, so the prediction UI does not present it as a meaningful low-confidence warning threshold. These are demo measurements, not full research findings.

| Check | Executed outcome |
|---|---|
| Real-image FastAPI upload | HTTP 200; returned a predicted class and confidence from the demo checkpoint |
| Demo test set and confusion matrix | 380 held-out PlantVillage images, 38 classes; PNG served by the API |
| Separate external evaluation | 184 PlantDoc images, 23 mapped classes; no external data used for fitting |
| Python backend/research tests | **55 passed**, one existing Starlette/httpx deprecation warning |
| Frontend unit tests | **5 passed** |
| TypeScript and production build | Passed; `/demo-evaluation` prerendered |
| Six-run research suite | Not completed; no six-run training/evaluation was launched for the demo |

Reproduce the existing demo artifacts only if they have not already been created. See the step-by-step [demo guide](demo_guide.md). The setup command refuses to overwrite conflicting demo outputs.

## Full-corpus continuation — 27 October 2026

The full pinned acquisition, extraction, audit and fixed-split preparation have completed. Existing archives were reused. The extractor verified the PlantVillage archive SHA-256 and ZIP CRCs, PlantDoc archive integrity/revision root, safe Windows destination names and expected inventories. All images decoded: PlantVillage 54,305/54,305 (38 classes), PlantDoc 2,578/2,578 (28 folder labels); zero corrupt images. PlantDoc extraction recorded 93 Windows filename mappings, preserving case-colliding names as separate files.

| Check | Executed outcome |
|---|---|
| Three full research manifests | `full_seed42`, `full_seed43`, `full_seed44` completed; identical membership and SHA-256 values, split seed 42; 37,863 train, 8,256 validation, 8,186 in-domain test, 2,107 external |
| Manifest leakage validation | Passed path containment, label vocabulary, and pairwise path/SHA-256/pixel/dHash/source-leaf checks; zero source rows quarantined |
| PlantVillage leaf grouping | 13,194/54,305 filename fallbacks across 15 classes explicitly recorded; 621 unique legacy Apple Black Rot map aliases retained for grouping only |
| PlantDoc external eligibility | 2,107 images across 23 mapped classes; 432 unmappable-label exclusions and 39 duplicate external images; no source/external hash overlap |
| Pretrained weights | Both ImageNet files SHA-256 verified and loaded in the model factory; ResNet-18 and MobileNetV3-Small report `pretrained_loaded=True` |
| Python tests | **51 passed**, one existing Starlette/httpx deprecation warning |
| Python syntax | `py_compile` passed for all modified research and test modules |
| Frontend unit tests | **5 passed** across 2 files |
| TypeScript | `npm.cmd run typecheck` passed |
| Next.js production build | `npm.cmd run build` passed; all 8 application routes prerendered |
| Full training/evaluation | Incomplete; one ResNet-18 checkpoint has a six-epoch history, but the other prescribed runs and final research evaluations remain pending |

The required Python test command was:

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp D:\plant-disease-research-data\tmp\pytest-final-suite -o cache_dir=D:\plant-disease-research-data\cache\pytest
```

The preparation audit uses the audited seed-42 split as the source for seeds 43 and 44, then validates and writes each run-specific manifest and report; it does not decode the same full corpora three times. Preparation-time leakage checks use the SHA-256 values recorded during audit; `source_bytes_verified` remains false. Further research train/evaluation stages must recompute and verify source bytes before model use. Detailed counts, label exclusions and filename fallback limitations are in [dataset_mapping.md](dataset_mapping.md) and [project_report.md](project_report.md). Full research metrics remain pending. Smoke, synthetic and teacher-demo results are labelled separately.

The 5 development- and 5 production-browser tests in the earlier validation history were not rerun for the demo. No six-run research suite or fabricated metrics were produced. Cloud deployment remains unperformed.

## Implemented files

- `frontend/`: six App Router pages, reusable layout/states/charts, typed API boundary, request hook, strict TypeScript, responsive styling, favicon, unit tests, browser tests, environment example and locked dependencies.
- `backend/`: modular FastAPI routers and response schemas, settings/CORS, image validation, bounded uploads, checkpoint cache, result/figure services, tests, inference dependency profile and container recipe.
- `research/`: acquisition, audit, class mapping, group-aware preparation, two architectures, training/checkpoint selection, calibration, selective prediction, evaluation, plots, CLI orchestration, tests, and quick/full/demo configurations.
- Root: editable Python package configuration, exact Python environment lock, Git/Docker ignore rules, README and local environment.
- `docs/`: methodology, references, literature review, report draft, deployment guide and this verification record.
- Generated `data/` and `artifacts/` are intentionally ignored by Git. They exist locally for this demonstration.

## Executed checks

| Check | Actual outcome |
|---|---|
| Python/Node/npm/Git and hardware inspection | Completed; Python 3.11.9 installed locally; Node 24.14.1; i5-11400H; ~8 GB RAM; GTX 1650 4 GB |
| PyTorch import and CUDA probe | PyTorch 2.6.0+cpu / Torchvision 0.21.0+cpu import successfully; CUDA unavailable in the installed CPU build |
| Initial Next.js and FastAPI startup | Both returned HTTP 200 locally |
| Python compilation | `python -m compileall -q research backend` passed |
| Research and backend tests | **24 passed**, one upstream Starlette/httpx deprecation warning |
| TypeScript | `npm.cmd run typecheck` passed |
| Frontend unit tests | **5 passed** |
| Next.js production build | `npm.cmd run build` passed; all six application routes prerendered |
| Edge browser suite against development server | **5 passed**: routes/live API, empty results, offline backend, mobile layout, image upload/inference |
| Production server startup | `npm.cmd run start` started successfully on 127.0.0.1:3000 |
| Edge browser suite against final production build | **5 passed** in 17.6 seconds, including loaded mobile layout and actual checkpoint inference |

The final production frontend and FastAPI backend were left running at `http://127.0.0.1:3000` and `http://127.0.0.1:8000`. Reopen using the README's terminal commands after those processes stop.

Python tests cover config validation, corrupt-image audit, deterministic duplicate/leaf grouping, conflict quarantine, explicit mapping exclusions, known metric values, external predictions outside mapped classes, temperature NLL/argmax behavior, tied confidence thresholds, no-feasible-threshold behavior, optimized risk-curve agreement with a brute-force implementation, both architecture/checkpoint round trips, safe archive extraction, prevention of research runs on known smoke datasets, API readiness/results/status, CORS, invalid/oversized uploads (including chunked bodies), missing checkpoints, safe plot access, synthetic checkpoint predictions/cache and malformed result warnings.

## Real-image smoke experiment

Executed `python -m research.scripts.download_data --download --subset 6 --skip-metadata`, then `python -m research.scripts.run_experiments --config research/configs/quick_test.yaml` from the repository root.

The acquisition used a real, small two-class sample from the upstream PlantVillage repository linked by the Hugging Face card and from Cropped PlantDoc's test folder. Recorded revisions:

- Hugging Face PlantVillage metadata API: `9e97599868962bd0079b8db4b7f1efa9185fa1e7`.
- Actual upstream PlantVillage sample: `7f7ecc7e1eaca78107e3affe7cb5abd9427e139a`.
- PlantDoc: `5467f6012d78d1c446145d5f582da6096f852ae8`.

Authoritative acquisition details and individual image hashes are in `data/raw/sources.json`; manifests are under `data/splits/quick/`.

Observed preparation counts: **8 train, 2 validation, 2 in-domain test, 12 external test**, with two apple classes. Both architectures completed one epoch at 64×64, CPU, batch size 4, random initialization. Checkpoints, validation fitting, in-domain/external evaluation, latency measurement, prediction CSVs and plots completed successfully. These are software smoke outputs with `purpose: smoke`; the sample is too small to support comparative research conclusions.

Hugging Face raw/resolve download endpoints repeatedly reset connections, including with Windows curl. The reachable GitHub API was used for image blobs. The smoke run explicitly skipped HF raw leaf metadata and used class-scoped filename grouping; this is recorded rather than hidden. Full official leaf-group coverage remains unverified.

## Synthetic smoke experiment

Executed `python -m research.scripts.synthetic_smoke`. Both architectures completed training and evaluation on generated colored random texture fixtures, with **16 train, 4 validation, 4 in-domain-role test and 12 external-role test** images. These domain names describe software roles only; synthetic images are not observations from either named dataset. Results carry `purpose: synthetic` and a separate run ID.

Across both smoke runs, generated artifacts include **4 model checkpoints, 4 evaluation JSON files, 14 CSV files, 58 PNG figures and 58 PDF figures**, plus training histories, raw logits and audit reports. Browser screenshots are stored separately under `artifacts/reports/`.

## Failures encountered and addressed

- PowerShell blocked `npm.ps1`; commands use `npm.cmd`.
- System Python resolved to 3.14; installed project-local 3.11 for the pinned ML stack.
- C: had little free disk space; used CPU wheels, avoided full corpora and cleared only the project's regenerable npm download cache.
- Hugging Face downloads reset; added bounded retry and a documented small-sample GitHub API path, retaining source limitations.
- Initial pytest fixture setup could not access the default temp directory; reran using `.tools/pytest-temp` outside the sandbox. All tests then passed.
- Synthetic generic filenames initially caused incorrect cross-class fallback leaf grouping; scoped fallback identifiers by class and reran both smoke pipelines successfully.
- Browser assertions initially matched Next.js's route announcer; scoped assertions to page content. Inference test initially ran before checkpoints existed; reran after successful smoke completion.
- Empty-set selective accuracy/risk is represented by null, and infeasible thresholds display an explicit abstain-all message.

## Pending and practical limits

1. Full PlantVillage and PlantDoc acquisition/audit, official leaf-map coverage review and final mapped-class review.
2. Pretrained-weight download, full multi-epoch training, CUDA execution and prespecified repeated-seed experiments. These were not run.
3. Final research tables, findings and conclusions. The report intentionally retains pending result fields.
4. Docker image build/runtime and cloud deployment. The container recipe is supplied but was not built; limited C: space makes additional large image layers inappropriate during this validation.
5. GitHub repository creation/push and Vercel/backend account configuration. No accounts, paid resources or deployments were created.

The project is a tested local software implementation with smoke evidence, not a completed empirical research study or validated agricultural diagnostic product.
