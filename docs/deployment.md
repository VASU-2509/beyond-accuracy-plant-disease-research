# Local verification and deployment

## Architecture decision

Deploy Next.js to Vercel and FastAPI to a separate persistent Python/container host. Training stays offline. The backend needs only code, runtime dependencies, selected trusted checkpoints, calibration sidecars and selected result/plot files. Raw datasets are not required for serving.

Vercel supports Python, but the [current limits page](https://vercel.com/docs/functions/limitations) lists a 500 MB uncompressed Python function limit, with larger-function options in beta. It also imposes memory, execution-time and request-size limits. The installed Windows research environment is about 1.6 GiB; this is not a Linux deployment-size measurement, but it makes assuming a small serverless bundle unjustified. PyTorch imports and checkpoint loading also affect cold starts. Large-function beta availability would still require independent packaging, cold-start and request-limit validation. No combined Vercel/PyTorch deployment is implemented or claimed tested.

The supplied Dockerfile is a separate service option. Choose a host that supports persistent Python processes or containers and enough disk/RAM. A local machine behind an appropriately secured gateway is another demonstration option. Free tiers may sleep or have insufficient resources; paid hosting is not required by this code. Compare current provider terms yourself before enabling billing. The framework imposes no specific hosting provider.

## 1. Verify locally

From **repository root**, run the README's dependency and quick-run commands, then:

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp .tools/pytest-temp
```

From **frontend/**:

```powershell
npm.cmd ci
npm.cmd run typecheck
npm.cmd test
npm.cmd run build
npm.cmd run start
```

From **backend/** in a separate terminal:

```powershell
..\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Check health, actual image upload, charts, missing artifacts, and CORS before using cloud services. Run `npx.cmd playwright test` from **frontend/** with the servers running. Do not deploy a smoke model as if it were a validated disease classifier.

## 2. Create a GitHub repository

Create an empty repository in your own GitHub account. From **repository root**, use the README's `git init`, add, review, commit, remote and push commands. No credentials belong in the repository. `.gitignore` excludes data, checkpoints, generated outputs, `.env`, virtual environments and build caches. Review staged files before the first push. GitHub account creation and repository publishing are user actions; they were not performed automatically.

## 3. Prepare a backend artifact bundle

Select a single verified run. Preserve this structure on the backend host:

```text
/srv/artifacts/
  models/<run>/resnet18.pt
  models/<run>/resnet18.calibration.json
  models/<run>/mobilenet_v3_small.pt
  models/<run>/mobilenet_v3_small.calibration.json
  results/<run>/*_evaluation.json
  reports/<run>/dataset_summary.json
  plots/<run>/*.png
```

Transfer with an authenticated host console, SSH/SFTP or a private object-store download using server-side credentials. Verify SHA-256 hashes against saved evaluation records after transfer. Calibration sidecars contain the expected checkpoint checksum. Never accept arbitrary public checkpoint URLs or untrusted pickle artifacts. Mount artifacts read-only for serving when practical. Do not commit weights just to deploy them.

## 4. Run FastAPI on a Python host

Without Docker, from **repository root on the host**:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r backend/requirements-inference.txt
.venv/bin/python -m pip install --no-deps -e .
export ARTIFACTS_DIR=/srv/artifacts
export CORS_ORIGINS='["https://YOUR-FRONTEND.vercel.app"]'
.venv/bin/python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

Use the host's process manager, HTTPS endpoint and assigned port. One worker is the default to avoid duplicating model RAM. Rate-limit public uploads and set a reverse-proxy body limit consistent with the app's 8 MiB file policy; authentication can be added at the gateway if the demonstration should be private. This application has no remote training endpoint. No private environment variable should use a `NEXT_PUBLIC_` prefix.

### Container option

From **repository root**, with Docker installed:

```bash
docker build -f backend/Dockerfile -t plant-reliability-api .
docker run --rm -p 8000:8000 \
  -v /ABSOLUTE/PATH/TO/artifacts:/srv/artifacts:ro \
  -e ARTIFACTS_DIR=/srv/artifacts \
  -e 'CORS_ORIGINS=["https://YOUR-FRONTEND.vercel.app"]' \
  plant-reliability-api
```

This Docker path is prepared, not validated unless explicitly recorded in `validation.md`. Do not assume host-local paths exist inside a container; use the volume mount. Host download credentials must remain server-side. Persistent disk/object storage setup is a user/provider action.

## 5. Deploy Next.js to Vercel

1. Sign in to Vercel and import your GitHub repository.
2. Set **Root Directory** to `frontend` and framework to Next.js.
3. Use Node.js 24 LTS where supported, install command `npm ci`, build command `npm run build`, and the framework default output directory.
4. Set `NEXT_PUBLIC_API_BASE_URL=https://YOUR-BACKEND-DOMAIN` for the intended production/preview environments. This value is public and compiled into the browser bundle; never place secrets here.
5. Deploy. The backend must be HTTPS to avoid mixed-content blocking.
6. Configure the backend's `CORS_ORIGINS` with the exact deployed frontend origin. Add specific preview origins only when needed; do not use a blanket wildcard for a private service.
7. After env changes, redeploy Next.js and restart FastAPI. Git pushes trigger deployment if the repository integration is enabled.

## 6. Verify production

Open `/health`, `/api/models`, `/api/results/summary`, and `/docs` on the backend domain. Health status means the HTTP service is running; inspect checkpoint/result flags for actual readiness. Then open the Vercel application and visit all six pages. Upload a small image to each available model, verify purpose/calibration/abstention labels, and open one figure URL. Test an invalid file and a missing-model selection through API docs. Verify an unavailable backend displays the retry message.

Inspect provider logs for cold starts, memory pressure, request rejections and inference failures. Measure cloud latency separately; laptop forward-only latency is not a cloud service-level claim. Configure spending controls before using any paid resource. This project does not perform account creation, paid provisioning or production deployment automatically.
