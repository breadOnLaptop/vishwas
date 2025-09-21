import os
from pathlib import Path
from pydantic_settings import BaseSettings

# ----------------------------------------
# Paths and Environment
# ----------------------------------------
HERE = Path(__file__).resolve().parent
LOCAL_ENV_PATH = (HERE / ".." / ".." / ".env").resolve()

# Load local .env if exists (for local dev)
if LOCAL_ENV_PATH.exists():
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=str(LOCAL_ENV_PATH))
else:
    print(f"[INFO] Local .env not found at {LOCAL_ENV_PATH}, relying on system envs or Render secrets.")

# Render secret paths
RENDER_SECRETS_PATH = Path("/etc/secrets")

ENV_FILE_RENDER = RENDER_SECRETS_PATH / ".env"
GOOGLE_CREDS_RENDER = RENDER_SECRETS_PATH / "google-credentials.json"

# ----------------------------------------
# Settings
# ----------------------------------------
class Settings(BaseSettings):
    # Google AI & GCP
    GOOGLE_APPLICATION_CREDENTIALS: str | None = None
    GCP_PROJECT: str | None = None
    GCP_REGION: str = "us-central1"
    TEXT_MODEL_ID: str = "gemini-2.5-flash"

    # Custom Search
    GOOGLE_SEARCH_API_KEY: str | None = None
    GOOGLE_SEARCH_CX: str | None = None

    # SMTP Email
    SMTP_HOST: str | None = None
    SMTP_PORT: int | None = None
    SMTP_USER: str | None = None
    SMTP_PASS: str | None = None
    REPORT_TO_EMAIL: str | None = None

    class Config:
        # Local dev fallback, Render secrets take priority
        env_file = str(ENV_FILE_RENDER) if ENV_FILE_RENDER.exists() else str(LOCAL_ENV_PATH)
        env_file_encoding = "utf-8"

# ----------------------------------------
# Instantiate settings
# ----------------------------------------
settings = Settings()

# Override Google credentials path if Render secret exists
if GOOGLE_CREDS_RENDER.exists():
    settings.GOOGLE_APPLICATION_CREDENTIALS = str(GOOGLE_CREDS_RENDER)
