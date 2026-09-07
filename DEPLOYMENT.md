# LiverAI — Deployment Guide

## Quick Start (Local Development)

```bash
# 1. Clone / extract the project
cd liver-cirrhosis-prediction

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install Tesseract OCR for lab report scanning
#    Windows: winget install UB-Mannheim.TesseractOCR
#    Ubuntu:  sudo apt install tesseract-ocr
#    macOS:   brew install tesseract

# 5. Train the ML model (first time only)
python train_model.py

# 6. (Optional) Copy and edit the environment file
copy .env.example .env         # Windows
# cp .env.example .env         # Linux / macOS

# 7. Run the application
python app.py
```

The app starts on **http://localhost:5000**. A default admin account is created automatically:
- Email: `admin@liverai.com`
- Password: `Admin@123`

## Production Deployment

### Environment Variables

Copy `.env.example` to `.env` and set at minimum:

| Variable | Required | Description |
|----------|----------|-------------|
| `SECRET_KEY` | **Yes** | Long random string. Generate: `python -c "import secrets; print(secrets.token_hex(32))"` |
| `DATABASE_URL` | No | SQLAlchemy URL. Defaults to SQLite. PostgreSQL: `postgresql+psycopg2://user:pass@host/db` |
| `ENV_NAME` | No | Set to `production` to hide dev helpers |
| `COOKIE_SECURE` | No | Set to `1` when serving over HTTPS |
| `ADMIN_EMAIL` | No | Bootstrap admin email (default: `admin@liverai.com`) |
| `ADMIN_PASSWORD` | No | Bootstrap admin password (default: `Admin@123`) |
| `TESSERACT_CMD` | No | Explicit path to Tesseract binary if not on PATH |

### Option A: PythonAnywhere (Free Tier)

1. Upload the project files (or clone from your repo)
2. Create a virtualenv and install requirements
3. Run `python train_model.py` in a Bash console
4. Set up a Web app → Manual Configuration → Python 3.11+
5. Edit the WSGI file to point at the app:
   ```python
   import sys
   sys.path.insert(0, '/home/yourusername/liver-cirrhosis-prediction')
   from wsgi import app as application
   ```
6. Set environment variables in the WSGI file or a `.env`
7. Reload the web app

### Option B: Render

1. Push to a Git repository
2. Create a new Web Service on Render
3. Set the build command: `pip install -r requirements.txt && python train_model.py`
4. Set the start command: `gunicorn -b 0.0.0.0:$PORT wsgi:app`
5. Add environment variables (`SECRET_KEY`, `DATABASE_URL` for Render PostgreSQL)
6. Deploy

### Option C: Railway

1. Push to a Git repository
2. Create a new project on Railway
3. Add a PostgreSQL plugin (sets `DATABASE_URL` automatically)
4. Set the start command: `gunicorn wsgi:app`
5. Add `SECRET_KEY` and `ENV_NAME=production` as variables
6. Deploy

### Option D: Local Production (Windows)

```bash
# Use waitress (already in requirements.txt)
waitress-serve --port=8000 wsgi:app
```

### Option E: Local Production (Linux/macOS)

```bash
# Use gunicorn
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 wsgi:app
```

## Database

- **Development**: SQLite (zero config, file-based)
- **Production**: PostgreSQL recommended. Set `DATABASE_URL` in `.env`
- Tables are auto-created on first run (`db.create_all()`)
- The admin account is auto-created if it doesn't exist

## Model Retraining

```bash
python train_model.py
```

This regenerates:
- `saved_model/random_forest_model.joblib` — trained model
- `saved_model/encoders.joblib` — fitted preprocessors
- `saved_model/imputer_values.joblib` — imputation values
- `saved_model/metrics.json` — evaluation metrics
- `static/images/charts/*.png` — evaluation plots

To use the real PBC dataset instead of synthetic data, replace `dataset/cirrhosis.csv` with the Kaggle/UCI version and re-run.

## Running Tests

```bash
python -m pytest test_app.py -v
```

All 187 tests cover: authentication, predictions, batch processing, history, admin, API endpoints, and security features.

## Project Structure

```
liver-cirrhosis-prediction/
├── app.py                  # Flask application (routes, views, API)
├── database.py             # SQLAlchemy models (User, Prediction, AuditLog, ContactMessage)
├── model.py                # ML model serving wrapper
├── preprocessing.py        # Feature validation, encoding, reference ranges
├── clinical_scores.py      # MELD, FIB-4, APRI, Child-Pugh scoring
├── ocr_extractor.py        # Tesseract OCR lab report extraction
├── pdf_report.py           # PDF report generation (ReportLab)
├── config.py               # Configuration (env vars, defaults)
├── wsgi.py                 # WSGI entry point
├── train_model.py          # Model training pipeline
├── generate_dataset.py     # Synthetic dataset generator
├── test_app.py             # Test suite (187 tests)
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── dataset/
│   └── cirrhosis.csv       # Training data
├── saved_model/            # Trained model artifacts
├── static/
│   ├── css/style.css       # Full CSS with light/dark themes
│   ├── js/main.js          # Client-side JavaScript
│   ├── images/charts/      # Evaluation plots
│   └── uploads/            # User uploads (OCR)
└── templates/              # 22 Jinja2 templates
```
