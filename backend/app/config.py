from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from research.storage import paths

ROOT = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    artifacts_dir: Path = paths()["artifacts_dir"]
    data_dir: Path = paths()["data_dir"]
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    max_upload_mb: int = 8
    model_config = SettingsConfigDict(env_file=Path(__file__).resolve().parents[1] / ".env", extra="ignore")
