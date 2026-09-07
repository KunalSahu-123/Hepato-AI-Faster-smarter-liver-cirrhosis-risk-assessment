"""
config.py — saxenashvam0321@gmail.com
----------
Central configuration for the Flask application.

Secrets are read from environment variables with local-dev fallbacks. In
production every value marked "CHANGE IN PRODUCTION" must be supplied via the
environment (see .env.example).
"""

import os
from datetime import timedelta

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Load .env before any class body reads os.environ, so a local .env actually
# takes effect. Values already exported in the real environment win, which is
# what production platforms (Render, Railway, PythonAnywhere) rely on.
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(BASE_DIR, ".env"), override=False)
except ImportError:  # python-dotenv is optional; env vars still work without it
    pass


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _normalise_db_url(url: str) -> str:
    """Heroku/Render style `postgres://` URLs need the psycopg driver prefix."""
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+psycopg2://", 1)
    return url


class Config:
    # --- Core -------------------------------------------------------------
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
    ENV_NAME = os.environ.get("ENV_NAME", "development")

    SQLALCHEMY_DATABASE_URI = _normalise_db_url(
        os.environ.get(
            "DATABASE_URL", f"sqlite:///{os.path.join(BASE_DIR, 'liver_cirrhosis.db')}"
        )
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {"pool_pre_ping": True}

    # --- Sessions & cookies ----------------------------------------------
    PERMANENT_SESSION_LIFETIME = timedelta(days=7)
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    # Enabled automatically when served over HTTPS; override with COOKIE_SECURE=1
    SESSION_COOKIE_SECURE = _env_bool("COOKIE_SECURE", False)
    REMEMBER_COOKIE_HTTPONLY = True
    REMEMBER_COOKIE_DURATION = timedelta(days=7)
    REMEMBER_COOKIE_SECURE = _env_bool("COOKIE_SECURE", False)

    # --- CSRF -------------------------------------------------------------
    WTF_CSRF_ENABLED = _env_bool("WTF_CSRF_ENABLED", True)
    WTF_CSRF_TIME_LIMIT = None  # tie CSRF token lifetime to the session

    # --- Uploads ----------------------------------------------------------
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
    MAX_REPORT_SIZE_MB = int(os.environ.get("MAX_REPORT_SIZE_MB", "10"))
    # Rejected by Werkzeug before the body is buffered into memory.
    MAX_CONTENT_LENGTH = MAX_REPORT_SIZE_MB * 1024 * 1024
    ALLOWED_REPORT_EXTENSIONS = {".png", ".jpg", ".jpeg", ".pdf", ".webp", ".bmp", ".tiff"}
    ALLOWED_BATCH_EXTENSIONS = {".csv"}

    # --- Model artefacts --------------------------------------------------
    MODEL_DIR = os.path.join(BASE_DIR, "saved_model")
    MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_model.joblib")
    ENCODER_PATH = os.path.join(MODEL_DIR, "encoders.joblib")
    METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
    DATASET_PATH = os.path.join(BASE_DIR, "dataset", "cirrhosis.csv")

    # --- Bootstrap admin --------------------------------------------------
    ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", "admin@liverai.com")
    ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "Admin@123")

    # --- Auth hardening ---------------------------------------------------
    MAX_LOGIN_ATTEMPTS = int(os.environ.get("MAX_LOGIN_ATTEMPTS", "8"))
    LOGIN_LOCKOUT_MINUTES = int(os.environ.get("LOGIN_LOCKOUT_MINUTES", "15"))
    PASSWORD_MIN_LENGTH = 8
    RESET_TOKEN_MAX_AGE_SECONDS = int(os.environ.get("RESET_TOKEN_MAX_AGE_SECONDS", "3600"))

    # --- OCR --------------------------------------------------------------
    # Optional explicit path to the Tesseract binary (Windows installs often
    # aren't on PATH). Example: C:\Program Files\Tesseract-OCR\tesseract.exe
    TESSERACT_CMD = os.environ.get("TESSERACT_CMD", "")

    # --- Pagination -------------------------------------------------------
    ITEMS_PER_PAGE = int(os.environ.get("ITEMS_PER_PAGE", "10"))


class TestConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    WTF_CSRF_ENABLED = False
    SECRET_KEY = "test-secret-key"
    MAX_LOGIN_ATTEMPTS = 1000  # don't trip the limiter during test runs
