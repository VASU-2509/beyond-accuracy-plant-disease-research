# Public deployment preparation

This repository is prepared for a **Vercel frontend and Render FastAPI backend**. No service has been created or deployed from this repository, and no cloud URL has been tested. The deployment requires separately transferring the model/evaluation artifacts described below; neither Git nor this configuration uploads them.

## Readiness and runtime requirements

- The frontend reads the backend origin from `NEXT_PUBLIC_API_BASE_URL`. Set it in Vercel before building; it is public and compiled into browser code.
- FastAPI reads `CORS_ORIGINS` as a JSON list of exact browser origins. Set it in Render after Vercel provides the production URL.
- Set `RESEARCH_STORAGE_ROOT` to an absolute Linux path. Without it, the research storage configuration defaults to a Windows `D:` path and backend startup fails on Render/Linux.
- FastAPI imports CPU PyTorch and serves a checkpoint from `ARTIFACTS_DIR`. The Render blueprint installs the CPU-only PyTorch wheels and mounts a 1 GB persistent disk at `/var/data`.
- The app does not need the raw PlantVillage or PlantDoc images at inference time. The demo prediction/evaluation pages do need the checkpoint, calibration sidecar, evaluation JSON, dataset summary, and result plots.

The local demo artifact bundle is currently present under `D:\plant-disease-research-data\demo\artifacts`, but is intentionally excluded from GitHub. Its required runtime files are:

```text
artifacts/
  models/demo/resnet18.pt
  models/demo/resnet18.calibration.json
  results/demo/resnet18_evaluation.json
  reports/demo/dataset_summary.json
  plots/demo/resnet18_in_domain_confusion.png
```

The checkpoint is about 44.9 MB. The calibration sidecar is bound to the checkpoint checksum; transfer both without modifying either. Do not upload model files to a public Git repository or public bucket. The checkpoint's public redistribution rights have not been established. Transfer only to a private Render disk using a private, approved method, and verify the checkpoint SHA-256 after transfer. Do not copy raw datasets, training histories, logits, prediction CSV/NPZ files, or PDFs unless there is a separate need.

These are the minimum files for the demo prediction and evaluation pages. The UI displays its confusion matrix from the PNG above; model comparison and calibration charts are rendered from the evaluation JSON. Other plot images are not required.

## Cost and availability

- Vercel Hobby currently lists a $0/month plan; usage, account eligibility, and plan terms can change.
- Render Free web services have 512 MB RAM, spin down after 15 minutes idle, and do not support persistent disks. That is not a suitable dependable configuration for this PyTorch checkpoint plus persistent result bundle.
- The included Render blueprint uses a paid Standard service (1 CPU, 2 GB RAM) and a 1 GB persistent disk. Check [Render's live pricing](https://render.com/pricing) before creating the service; the total is the service compute plus disk and any applicable usage/tax. There is no cost commitment until you create/upgrade the service.

For a one-time teacher demonstration, start the Render paid service only when needed and review its billing settings. A free backend is possible only as a best-effort experiment with a potentially unavailable checkpoint after restarts; it is not the recommended configuration.

## Deploy Next.js to Vercel

1. Import the GitHub repository into Vercel and set **Root Directory** to `frontend`.
2. Use the detected Next.js framework, `npm ci` for installation, and `npm run build` for the build. The frontend declares Node.js `>=22.9`; use a supported Node 24 release.
3. Deploy once to obtain the exact production origin, for example `https://YOUR-PROJECT.vercel.app`. API-backed pages will remain unavailable until the backend URL is configured.

## Deploy FastAPI to Render

1. Sign in to Render and create a Blueprint from this GitHub repository. Select the repository root so Render reads [`render.yaml`](../render.yaml).
2. Review the proposed paid Standard web service and 1 GB disk before confirming. Do not proceed if you do not approve the displayed billing.
3. When prompted for `CORS_ORIGINS`, enter a JSON array with the exact production frontend origin from Vercel:

   ```text
   ["https://YOUR-PROJECT.vercel.app"]
   ```

   Do not include a path or trailing slash. Add an exact preview origin only if needed. Avoid `*`.
4. Wait for the backend health check at `/health`. An HTTP success alone does not mean model artifacts are installed; `/api/models` and `/health` show whether the checkpoint and evaluation results are available.
5. Transfer the five required files to the persistent disk, preserving their paths under `/var/data/artifacts/`, using a private approved method. Do not place credentials in the repository or a `NEXT_PUBLIC_` variable. The service start command creates the writable temp and artifact directories on the mounted disk.
6. Confirm the following backend endpoints:

   ```text
   https://YOUR-BACKEND.onrender.com/health
   https://YOUR-BACKEND.onrender.com/api/models
   https://YOUR-BACKEND.onrender.com/api/results/summary
   https://YOUR-BACKEND.onrender.com/docs
   ```

   Confirm the `demo__resnet18` model reports `checkpoint_available: true` and calibration/results are present.

## Connect and verify the frontend

1. In Vercel, set `NEXT_PUBLIC_API_BASE_URL` to the backend HTTPS origin with no trailing slash, such as `https://YOUR-BACKEND.onrender.com`. Set it for Production and any Preview environments you intend to use.
2. Redeploy the frontend so the public environment value is included in its build.
3. Open the Vercel site, upload a small JPEG/PNG/WebP image, and verify a prediction. Open **Demo evaluation** and confirm the evaluation summary and confusion-matrix plot load.

## Keep local development unchanged

The checked-in frontend example remains local:

```text
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

For local FastAPI, continue using the commands and artifact paths in the [teacher demonstration guide](demo_guide.md). Cloud environment variables belong in the provider dashboards, not `.env.local` or source control. The Render blueprint's Linux storage root applies only to the cloud service.

## Remaining limitations

- Deployment has not been performed or verified; service URLs, provider account setup, billing approval, artifact transfer, and production image-upload testing remain user/provider steps.
- No model or dataset distribution permission is implied by this configuration.
- A cold start or CPU inference can take longer than local development. Confirm the deployed provider's timeout and memory limits under real use.
- This demonstration is not a validated agricultural diagnostic service.
