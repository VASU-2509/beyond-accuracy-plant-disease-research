FROM python:3.11-slim
WORKDIR /srv/app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 ARTIFACTS_DIR=/srv/artifacts
COPY backend/requirements*.txt ./backend/
RUN pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r backend/requirements-inference.txt
COPY pyproject.toml ./
COPY backend ./backend
COPY research ./research
RUN pip install --no-cache-dir --no-deps -e . && useradd --create-home appuser
USER appuser
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
