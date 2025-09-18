import os
from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
ENV_PATH = (HERE / ".." / ".." / ".env").resolve()

if ENV_PATH.exists():
    load_dotenv(dotenv_path=str(ENV_PATH))
else:
    print(f"[WARN] .env not found at expected location: {ENV_PATH}")

class Settings(BaseSettings):
    GOOGLE_APPLICATION_CREDENTIALS: str | None = None
    GCP_PROJECT: str | None = None
    GCP_REGION: str = "us-central1"
    TEXT_MODEL_ID: str = "gemini-2.5-flash"

    # Custom Search (programmable search)
    GOOGLE_SEARCH_API_KEY: str | None = None
    GOOGLE_SEARCH_CX: str | None = None

    SMTP_HOST: str | None = None
    SMTP_PORT: int | None = None
    SMTP_USER: str | None = None
    SMTP_PASS: str | None = None

    class Config:
        env_file = str(ENV_PATH)
        env_file_encoding = "utf-8"

settings = Settings(_env_file=str(ENV_PATH))
