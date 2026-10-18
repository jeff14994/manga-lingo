"""
Configuration: loads environment variables from .env
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

# Google Cloud
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "")  # set in .env
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
GOOGLE_APPLICATION_CREDENTIALS_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON", "{}")

# Gemini (generateContent) for dialogue OCR, translation and transcription.
# gemini-3-flash-preview (global only) picks the right kanji for homophones
# far more often than gemini-2.5-flash (e.g. 移れ vs 映れ in anime audio).
# Fallback if the preview is retired: GEMINI_MODEL=gemini-2.5-flash,
# GEMINI_LOCATION=us-central1.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
GEMINI_LOCATION = os.getenv("GEMINI_LOCATION", "global")

# AlloyDB
ALLOYDB_HOST = os.getenv("ALLOYDB_HOST", "127.0.0.1")
ALLOYDB_PORT = int(os.getenv("ALLOYDB_PORT", "5432"))
ALLOYDB_USER = os.getenv("ALLOYDB_USER", "postgres")
ALLOYDB_PASSWORD = os.getenv("ALLOYDB_PASSWORD", "")
ALLOYDB_DATABASE = os.getenv("ALLOYDB_DATABASE", "postgres")
ALLOYDB_SSLMODE = os.getenv("ALLOYDB_SSLMODE", "require")

# Data directory for local file storage
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
(DATA_DIR / "images").mkdir(exist_ok=True)
(DATA_DIR / "audio").mkdir(exist_ok=True)
(DATA_DIR / "video").mkdir(exist_ok=True)


def get_gcp_credentials():
    """Parse the JSON credentials string from .env"""
    creds_json = GOOGLE_APPLICATION_CREDENTIALS_JSON
    if creds_json and creds_json != "{}":
        return json.loads(creds_json)
    return None
