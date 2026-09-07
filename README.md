# 🩺 LiverAI — Liver Cirrhosis Prediction System

An AI-powered clinical decision-support web application that predicts a patient's
risk of liver cirrhosis using a tuned **Random Forest Classifier**, built with
Flask, SQLAlchemy, and a healthcare-themed Bootstrap 5 frontend.

> ⚠️ **Disclaimer:** This system is a decision-support tool for academic
> demonstration. It is **not** a medical device and must never be used as a
> substitute for diagnosis by a licensed physician.

---

## 1. Project Introduction

Liver cirrhosis is frequently diagnosed late because early biomarkers are
subtle and reviewing dozens of lab values manually is slow. LiverAI lets a
clinician (or student, for this project) enter a patient's lab panel and
receive an instant risk classification, a calibrated probability score, the
top features driving that prediction, and a downloadable PDF report.

## 2. Problem Statement

Manual triage of liver panels doesn't scale, and important risk signals
(e.g. rising bilirubin + falling albumin + low platelets) can be missed when
reviewed one value at a time. An ML-assisted first-pass triage tool can help
flag likely-high-risk patients for prioritized follow-up.

## 3. Objectives

- Predict cirrhosis risk from 16 standard clinical features
- Give a transparent, explainable result (feature importances, not a black box)
- Track every prediction per user with a searchable history
- Generate professional PDF/CSV reports
- Provide secure authentication, an audit trail, and an admin panel

## 4. Architecture

```
Browser (Bootstrap 5 + Chart.js)
        │
        ▼
Flask App (app.py) ── Flask-Login (auth) ── SQLAlchemy ORM ── SQLite
        │
        ▼
model.py → CirrhosisPredictor ── preprocessing.py ── saved_model/*.joblib
        (Random Forest trained via train_model.py on dataset/cirrhosis.csv)
```

- **Frontend:** Server-rendered Jinja2 templates, Bootstrap 5, Bootstrap Icons,
  Chart.js, custom CSS (glassmorphism cards, dark/light theme, toasts, loaders).
- **Backend:** Flask app factory-style routes in `app.py`, business/ML logic
  isolated in `model.py` / `preprocessing.py` / `train_model.py`.
- **Database:** SQLite via SQLAlchemy ORM (`database.py`) — `users`,
  `predictions`, `audit_logs` tables.
- **ML:** scikit-learn `RandomForestClassifier`, tuned with `GridSearchCV`,
  evaluated with accuracy/precision/recall/F1/ROC-AUC + curves.

## 5. Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML5, CSS3, Bootstrap 5, JavaScript (ES6), Chart.js |
| Backend | Python, Flask, Jinja2, Flask-Login, Werkzeug |
| ML | scikit-learn, pandas, NumPy, joblib, matplotlib, seaborn |
| Database | SQLite, SQLAlchemy ORM |
| Reports | ReportLab (PDF), csv (CSV) |

## 6. Dataset

`dataset/cirrhosis.csv` follows the same schema as the well-known **Mayo
Clinic Primary Biliary Cirrhosis (PBC) study** dataset (Age, Sex, Ascites,
Hepatomegaly, Spiders, Edema, Bilirubin, Cholesterol, Albumin, Copper,
Alk_Phos, SGOT, Tryglicerides, Platelets, Prothrombin, Stage).

> **Important:** this sandboxed build environment has no internet access to
> Kaggle/UCI, so `generate_dataset.py` produces a **statistically realistic
> synthetic stand-in** (same columns, realistic ranges, and clinically
> plausible correlations between markers, stage, and risk) so the entire
> pipeline is genuinely trainable end-to-end.
>
> **To use the real dataset:** download `cirrhosis.csv` from Kaggle/UCI,
> place it at `dataset/cirrhosis.csv` with matching column names, and re-run
> `python train_model.py` — no other code changes are needed.

## 7. ML Workflow

`train_model.py` runs the full pipeline:

1. Load & de-duplicate the dataset
2. Generate EDA charts (age/gender/stage distributions, correlation heatmap)
3. Impute missing numeric values (median) and encode categoricals
4. Train/test split (80/20, stratified)
5. `GridSearchCV` hyperparameter tuning (n_estimators, max_depth, min_samples_split/leaf)
6. 5-fold cross-validation
7. Evaluate: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, classification report
8. Plot: confusion matrix, ROC curve, precision-recall curve, feature importance, learning curve
9. Serialize the model + preprocessor with `joblib` to `saved_model/`

Current run on the synthetic dataset achieves **~91% accuracy** and **~0.97 ROC-AUC**
(see `saved_model/metrics.json` for the exact numbers from the last training run).

## 8. Auto-Fill from Lab Report (OCR)

Typing 16 clinical values by hand for every patient is slow. The **Prediction**
page includes an upload panel above the manual form: drop in a photo or scan
of the patient's liver function test report and it auto-fills whatever it can
confidently read (age, gender, bilirubin, cholesterol, albumin, copper,
alkaline phosphatase, SGOT, triglycerides, platelets, prothrombin time).

- Manual entry is **never removed** — every auto-filled field stays in a
  normal editable input, so you can review and correct anything before
  submitting.
- Physical-exam/staging fields (Ascites, Hepatomegaly, Spiders, Edema,
  Disease Stage) aren't on lab reports, so those always need manual selection.
- Extraction runs via Tesseract OCR + regex pattern matching (`ocr_extractor.py`)
  and includes a plausibility check that self-corrects common OCR decimal-point
  errors (e.g. a misread "3.4" as "34").
- Supported formats: JPG, PNG, WEBP, BMP, TIFF, and single-page PDF, up to 10MB.

**System dependency:** OCR needs the Tesseract binary installed on the machine
(separate from the `pytesseract` Python wrapper in `requirements.txt`):

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr poppler-utils   # poppler-utils enables PDF support

# macOS
brew install tesseract poppler

# Windows
# Install from https://github.com/UB-Mannheim/tesseract/wiki, add it to PATH
```

If Tesseract isn't installed, the rest of the app still works fine — the
upload panel will just return an error and the manual form remains fully usable.

## 9. Installation

```bash
git clone <your-repo-url>
cd Liver-Cirrhosis-Prediction-System

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

# 1) Generate the (synthetic) dataset — skip if you supplied a real cirrhosis.csv
python generate_dataset.py

# 2) Train the model (writes saved_model/*.joblib + static/images/charts/*.png)
python train_model.py

# 3) Run the app (creates the SQLite DB + a default admin on first run)
python app.py
```

Visit **http://127.0.0.1:5000**

**Default admin login:** `admin@liverai.com` / `Admin@123`
(change `ADMIN_EMAIL` / `ADMIN_PASSWORD` env vars before deploying).

## 10. Environment Variables

| Variable | Purpose | Default |
|---|---|---|
| `SECRET_KEY` | Flask session signing key | dev key (change in production!) |
| `DATABASE_URL` | SQLAlchemy DB URI | `sqlite:///liver_cirrhosis.db` |
| `ADMIN_EMAIL` | Bootstrap admin email | `admin@liverai.com` |
| `ADMIN_PASSWORD` | Bootstrap admin password | `Admin@123` |

## 11. Project Structure

```
Liver-Cirrhosis-Prediction-System/
├── app.py                 # Flask app: routes, auth, dashboard, API
├── model.py                # CirrhosisPredictor wrapper (load model + predict)
├── train_model.py          # Full ML training + evaluation pipeline
├── preprocessing.py        # Shared cleaning/encoding logic
├── ocr_extractor.py         # OCR lab-report parsing (Tesseract + regex)
├── generate_dataset.py     # Synthetic dataset generator
├── database.py              # SQLAlchemy models (User, Prediction, AuditLog)
├── config.py                # App configuration
├── test_app.py               # Unit + integration test suite (pytest)
├── requirements.txt
├── dataset/cirrhosis.csv
├── saved_model/            # Trained model, encoders, metrics.json
├── static/
│   ├── css/style.css
│   ├── js/main.js
│   └── images/charts/       # Auto-generated EDA & evaluation plots
└── templates/               # Jinja2 templates (see below)
```

## 12. Pages

Home · About · Services · Prediction · Dashboard · History · Reports ·
Admin · Profile · Contact · Login · Register · Forgot Password · 404/403/500

## 13. API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/predict` (form) / `/api/predict` (JSON) | Run a prediction |
| POST | `/prediction/extract` | Upload a lab report image/PDF, get back auto-filled field values (JSON) |
| GET | `/api/patients` | List current user's predictions (JSON) |
| GET | `/api/history` | Last 50 predictions (JSON) |
| GET | `/api/dashboard` | Dashboard summary counts (JSON) |
| POST | `/register` / `/login` / `/logout` | Auth |
| GET | `/reports/csv` | Export history as CSV |
| GET | `/reports/pdf/<id>` | Export a single report as PDF |

## 14. Security

- Passwords hashed with Werkzeug (`generate_password_hash` / `check_password_hash`)
- Session-based auth via Flask-Login, `remember me` support
- SQL injection protected by SQLAlchemy ORM parameterization
- Server-side + client-side input validation
- Role-based access control (`admin_required` decorator) for the admin panel
- Audit log of logins, registrations, predictions, and admin actions
- Ownership checks on delete/edit routes (users can only touch their own records, admins can touch any)

## 15. Testing

```bash
python -m pytest test_app.py -v
```

Covers: preprocessing/encoding unit tests, model prediction unit tests, and
integration tests for public pages, auth flow, the prediction workflow,
history/CSV export, and admin access control (20 tests, all passing).

## 16. Deployment Notes

- **Localhost:** `python app.py` (Flask dev server)
- **Render / Railway / PythonAnywhere:** set `SECRET_KEY` and `DATABASE_URL`
  env vars, use a production WSGI server (e.g. `gunicorn app:app`), and run
  `bootstrap_database()` once (or call it in an entrypoint script) to create
  tables and the default admin.
- **Docker:** add a `Dockerfile` that installs `requirements.txt`, copies the
  project, runs `train_model.py` during the image build (or mounts a
  pre-trained `saved_model/`), and starts `gunicorn -b 0.0.0.0:8000 app:app`.

## 17. Future Scope

- Deep learning (MLP/TabNet) and XGBoost/LightGBM/CatBoost comparison
- Explainable AI via SHAP/LIME for per-prediction explanations
- Hospital EHR integration, real-time monitoring dashboards
- Multi-language support, Android companion app

## 18. Contributors

Built as a final-year B.Tech major project deliverable.

## 19. License

For academic and educational use.
