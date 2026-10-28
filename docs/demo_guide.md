# Teacher demonstration guide

## Required local files

The public source repository does not contain datasets, checkpoints, or demo evaluation artifacts. A fresh clone cannot serve predictions or the evaluation page until the project owner separately provides the prepared files under `D:\plant-disease-research-data`. For a new setup, obtain the complete `demo\artifacts` bundle and the corresponding `full\data` directory through an approved method. Do not copy model weights or datasets into the Git repository.

The artifact preparation command below is only for a workspace that already has the compatible `full_seed42` checkpoint, training history, source manifest, and prepared data. It does not download these prerequisites. If the demo artifact bundle is already present, skip preparation and launch the app.

## What this demo reuses

The demo reuses the saved ResNet-18 checkpoint from `full_seed42`: 38 PlantVillage classes, 224×224 inputs, ImageNet initialization, and the checkpoint selected at epoch 5. The checkpoint was verified against its original full-run manifest. The demo creates a metadata-adjusted copy under `D:\plant-disease-research-data\demo\artifacts`; it does not edit the source checkpoint or run any of the six full experiments. The current `.venv` is CPU-only, so inference and evaluation use the CPU.

The demo makes a deterministic, balanced sample from the already separate full-run splits: 20 training-reference, 20 validation, and 10 test images per PlantVillage class. The existing model is **not retrained** on that sample; the small train sample is retained in the demo manifest to document its source split. Temperature scaling and the optional abstention threshold use validation images only. The held-out demo test sample is used only for evaluation. A capped sample of up to 8 PlantDoc images per eligible class is evaluated separately and never used for training or validation fitting.

The confidence warning is shown only when the validation-selected rule satisfies both constraints: empirical validation risk at most 10% and validation coverage at least 20%, and the selected cutoff is nonzero. If no threshold meets those constraints or the selected rule accepts every prediction, the UI shows that no meaningful warning threshold is available; it does not label every prediction low-confidence.

## Create the demo artifacts once

Open PowerShell at the repository root:

```powershell
cd C:\Users\Acer\Desktop\plant-disease-research
$env:RESEARCH_STORAGE_ROOT = 'D:\plant-disease-research-data'
.\.venv\Scripts\python.exe -m research.scripts.prepare_demo --config research\configs\demo.yaml
```

This reuses the existing full manifest and model; it does not download data or train a model. Expected demo sizes are 760 train-reference images, 760 validation images, and 380 held-out PlantVillage test images. The PlantDoc sample is capped at up to 8 images per eligible mapped class. Actual counts and all metrics are written from the selected rows; use the saved results rather than assuming an accuracy.

The currently generated demo result has **98.16% accuracy on 380 PlantVillage test images** and **9.24% accuracy on 184 PlantDoc images across 23 mapped classes**. These figures are the saved demo evaluation, not full-study results. The external result is substantially lower and should be shown as evidence of this model's difficulty under external image shift, not hidden or presented as a reliable diagnosis. Validation selected a zero confidence cutoff (accept all), so the UI correctly displays no meaningful low-confidence warning.

The demo manifest is added at `D:\plant-disease-research-data\full\data\splits\demo`. Demo checkpoints, evaluation JSON, calibration, plots, and dataset summary are written only under `D:\plant-disease-research-data\demo\artifacts`. Existing research manifests, checkpoints, and outputs are preserved. The runner refuses to replace conflicting demo artifacts.

## Launch the dashboard

Use two PowerShell terminals.

**Terminal 1 — backend:**

```powershell
cd C:\Users\Acer\Desktop\plant-disease-research
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

Open <http://127.0.0.1:3000>. The API is at <http://127.0.0.1:8000/docs>.

## Teacher walkthrough

1. Open **Image prediction** and choose a leaf image as JPEG, PNG, or WebP (up to 8 MiB).
2. Leave the ResNet-18 demo checkpoint selected and click **Analyze image**.
3. Point out the predicted PlantVillage class, raw/calibrated confidence, and top probabilities. Explain that confidence is not a guarantee of correctness.
4. Open **Demo evaluation**. Show the test accuracy alongside the exact test-image count and confusion matrix. These values come from the balanced held-out PlantVillage demo sample.
5. Point out the separately reported PlantDoc external score. It uses no PlantDoc images for training, calibration, or threshold selection.
6. Explain that this is a small college demonstration using one saved model, not the six-run research study or a validated farm diagnostic system.

No experiment is launched from the web app. If the demo artifacts already exist, do not rerun setup over them; use the existing artifacts.
