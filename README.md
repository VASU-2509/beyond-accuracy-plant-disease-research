# Beyond Accuracy

**Evaluating Domain Generalization and Uncertainty Calibration in Deep Learning-Based Plant Disease Recognition**

A reproducible research codebase and local demonstration app for studying how plant-disease image classifiers behave across controlled and external image domains. It includes a FastAPI inference API, a Next.js dashboard, and scripts for dataset auditing, model training, calibration, selective prediction, and evaluation.

> **Project status:** The recorded college-demo results are small-scale demonstrations, not final research findings or evidence of safe agricultural diagnosis. The prescribed six full research runs are still pending.

## Research question and objectives

**Research question:** How well do classifiers trained on controlled PlantVillage images generalize to external PlantDoc images, and how reliable are their confidence estimates?

The planned study compares ResNet-18 and MobileNetV3-Small on:

- In-domain and external classification performance.
- Validation-fitted temperature calibration.
- Confidence-based selective prediction.
- Checkpoint size and local inference latency.

## Features and technology

- **Research pipeline:** pinned source inventories, image audits, explicit label mapping, group-aware splits, training, calibration, metrics, and plots.
- **Local dashboard:** image upload and predictions, experiment summaries, model comparison, calibration views, and demo evaluation with a confusion matrix.
- **Backend:** Python 3.11, PyTorch/Torchvision, FastAPI, Pillow, scikit-learn.
- **Frontend:** Next.js, React, TypeScript, Vitest, and Playwright.
- **Reproducibility:** fixed configurations, saved split manifests, checkpoint metadata, and validation records.

## Datasets

- [PlantVillage dataset card](https://huggingface.co/datasets/mohanty/PlantVillage), pinned revision `9e97599868962bd0079b8db4b7f1efa9185fa1e7`. The project uses the 54,305 color images across 38 classes.
- [PlantDoc classification dataset](https://github.com/pratikkayal/PlantDoc-Dataset), pinned revision `5467f6012d78d1c446145d5f582da6096f852ae8`. The audited source contains 2,578 images. The project maps only labels considered compatible with PlantVillage classes; unmappable labels are excluded.

Datasets and downloaded archives are not included in this repository. Follow the source terms and review [dataset mapping](docs/dataset_mapping.md) and [methodology](docs/methodology.md) before acquiring or reusing data.

## Models and evaluation method

The planned full protocol trains both architectures with ImageNet initialization on the same group-aware PlantVillage split, then evaluates the in-domain holdout and mapped PlantDoc images. Temperature and the optional confidence-rejection threshold are fitted using PlantVillage validation data only. External images are evaluation-only. See the [methodology](docs/methodology.md) and [execution plan](docs/execution_plan.md).

### College demonstration results

The current demo reuses a saved 38-class, ImageNet-initialized ResNet-18 checkpoint. It evaluates a fixed, balanced held-out sample:

| Dataset | Images | Result |
|---|---:|---:|
| PlantVillage held-out demo test set | 380 | **98.16% accuracy** (373/380) |
| PlantDoc external demo sample | 184 across 23 mapped classes | **9.24% accuracy** (17/184) |

These are measurements from the saved demo outputs, not the completed research study. PlantDoc performance is low and demonstrates a substantial domain gap for this model and sample. The demo does not establish real-world diagnostic safety.

The prescribed study—two architectures across three seeds (six full runs)—has **not been completed**. One saved `full_seed42` ResNet-18 checkpoint exists; the remaining prescribed runs and final research evaluation are pending. Full training is not required to browse the source, but the saved demo model and prepared local datasets are not distributed here.

## Checkpoint availability and licensing

Model checkpoints, calibration sidecars, datasets, and generated results are excluded from Git. The local demo checkpoint is approximately 44.8 MB; the repository also contains additional smoke and synthetic checkpoints. Although individual files are below GitHub's 100 MB hard limit, placing large binary experiment artifacts in ordinary Git would add repository history weight, and the checkpoint's training-data redistribution provenance has not been established for public redistribution.

To run the saved demo, obtain the compatible `demo/resnet18.pt` checkpoint and its `.calibration.json` sidecar from the project owner through an approved file-sharing method, and place them under `D:\plant-disease-research-data\demo\artifacts\models\demo\`. The app also requires the prepared PlantVillage/PlantDoc data and demo evaluation artifacts at the paths in the [teacher demonstration guide](docs/demo_guide.md). These files remain local and are not uploaded by these instructions. No project-level software license has been assigned; dataset and third-party component terms remain separate.

## Setup and launch

### Prerequisites

- Python 3.11.
- Node.js 22.9 or later and npm.
- For the saved demo, separately supplied local dataset and demo artifacts as described above.

Clone the repository, then from its root in PowerShell create an environment and install dependencies:

```powershell
git clone https://github.com/VASU-2509/beyond-accuracy-plant-disease-research.git
Set-Location .\beyond-accuracy-plant-disease-research
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
```

In the frontend directory, install the locked packages and configure the local API address:

```powershell
cd frontend
npm.cmd ci
Copy-Item .env.example .env.local
```

Run the backend and frontend in two terminals. The demo backend uses the existing full data directory and separate demo artifacts:

**Terminal 1 — backend:**

```powershell
Set-Location C:\Users\Acer\Desktop\plant-disease-research
$env:RESEARCH_STORAGE_ROOT = 'D:\plant-disease-research-data'
$env:DATA_DIR = 'D:\plant-disease-research-data\full\data'
$env:ARTIFACTS_DIR = 'D:\plant-disease-research-data\demo\artifacts'
Set-Location backend
..\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

**Terminal 2 — frontend:**

```powershell
cd C:\Users\Acer\Desktop\plant-disease-research\frontend
npm.cmd run dev
```

Open <http://127.0.0.1:3000>. Upload an image at **Image prediction** or open **Demo evaluation**. The API documentation is at <http://127.0.0.1:8000/docs>. Stop each server with Ctrl+C.

For detailed setup and a presentation walkthrough, see the [teacher demonstration guide](docs/demo_guide.md). Its preparation command expects the source checkpoint and prepared full dataset to already exist; it does not download data or replace a missing checkpoint.

## Limitations

- PlantVillage images are controlled; PlantDoc has different image conditions and a conservative, imperfect class mapping.
- The external demo sample is small and its accuracy is low.
- A high softmax score is not a guarantee of correctness; the demo may confidently misclassify unfamiliar images.
- The full six-run experiment schedule and final research analysis remain incomplete.
- The prototype is for research demonstration, not field deployment, diagnosis, or treatment advice.
- This repository does not include datasets, model weights, or experiment outputs.

## Screenshots

No current screenshots are included because the existing local captures show older smoke-test content rather than the present demo. After capturing current pages, add presentation-safe images under `docs/screenshots/` and link them here.

## Documentation

- [Teacher demonstration guide](docs/demo_guide.md)
- [Dataset mapping](docs/dataset_mapping.md)
- [Methodology](docs/methodology.md)
- [Literature review](docs/literature_review.md)
- [Execution plan](docs/execution_plan.md)
- [Draft project report](docs/project_report.md)
- [Validation record](docs/validation.md)
- [Deployment notes](docs/deployment.md)
