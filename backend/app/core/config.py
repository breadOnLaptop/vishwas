# app/core/config.py
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Resolve .env reliably: backend/.env (two levels above app/core)
HERE = Path(__file__).resolve().parent   # .../backend/app/core
ENV_PATH = (HERE / ".." / ".." / ".env").resolve()  # .../backend/.env

# Load .env into process env (so pydantic can also see it)
if ENV_PATH.exists():
    load_dotenv(dotenv_path=str(ENV_PATH))
else:
    # Helpful debug message if .env not found
    print(f"[WARN] .env not found at expected location: {ENV_PATH}")

class Settings(BaseSettings):
    GOOGLE_APPLICATION_CREDENTIALS: str
    GCP_PROJECT: str
    GCP_REGION: str = "us-central1"
    TEXT_MODEL_ID: str = "text-bison@001"

    SMTP_HOST: str
    SMTP_PORT: int
    SMTP_USER: str
    SMTP_PASS: str

    class Config:
        # also point Pydantic to the same file (redundant but safe)
        env_file = str(ENV_PATH)
        env_file_encoding = "utf-8"

# instantiate settings explicitly pointing to env file
settings = Settings(_env_file=str(ENV_PATH))

# debug prints
# print(f"[DEBUG] Using .env path: {ENV_PATH}")
# print(f"[DEBUG] GOOGLE_APPLICATION_CREDENTIALS = {settings.GOOGLE_APPLICATION_CREDENTIALS!r}")
# print(f"[DEBUG] GCP_PROJECT = {settings.GCP_PROJECT!r}")
