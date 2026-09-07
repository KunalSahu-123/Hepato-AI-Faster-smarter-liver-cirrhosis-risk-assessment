"""
app.py — saxenashvam0321@gmail.com
-------
Flask application for the Liver Cirrhosis Prediction System.

Local run:
    python app.py                 (dev server, auto-creates DB + admin)
Production:
    waitress-serve --port=8000 wsgi:app
    gunicorn -b 0.0.0.0:8000 wsgi:app

Layout:
  * app factory (`create_app`) so tests and WSGI servers build their own instance
  * blueprint-free single module for readability, grouped by section
  * every mutating route is CSRF-protected and ownership-checked
"""

import csv
import io
import json
import os
from datetime import datetime, timedelta, timezone
from functools import wraps

import pandas as pd
from flask import (
    Flask, abort, current_app, flash, jsonify, redirect, render_template,
    request, send_file, session, url_for,
)
from flask_login import (
    LoginManager, current_user, login_required, login_user, logout_user,
)
from flask_wtf.csrf import CSRFProtect, CSRFError
from sqlalchemy import func, cast, Date
from werkzeug.utils import secure_filename

from clinical_scores import compute_all
from config import Config
from database import (
    AuditLog, ContactMessage, Prediction, User, db, log_action,
)
from model import CirrhosisPredictor, ModelNotTrainedError
from ocr_extractor import (
    OCRUnavailableError, configure_tesseract, extract_lab_report, is_ocr_available,
)
from pdf_report import build_batch_pdf, build_prediction_pdf
from preprocessing import (
    FIELD_META, REQUIRED_FORM_FIELDS, ValidationError, flag_abnormal,
    rows_from_dataframe, validate_clinical_input,
)
from recommendations import get_recommendations, sort_by_distance
from health_assistant import get_health_response

csrf = CSRFProtect()
login_manager = LoginManager()

# Module-level cache for the loaded model. Loading the forest takes a moment and
# it is read-only at request time, so one instance is shared across requests.
_predictor = None
_predictor_error = None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def create_app(config_object=Config):
    app = Flask(__name__)
    app.config.from_object(config_object)

    db.init_app(app)
    csrf.init_app(app)

    login_manager.init_app(app)
    login_manager.login_view = "login"
    login_manager.login_message = "Please log in to access this page."
    login_manager.login_message_category = "warning"
    login_manager.session_protection = "strong"

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    configure_tesseract(app.config.get("TESSERACT_CMD", ""))

    register_routes(app)
    register_health_chat(app)
    register_error_handlers(app)
    register_template_helpers(app)

    return app


@login_manager.user_loader
def load_user(user_id):
    try:
        return db.session.get(User, int(user_id))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def get_predictor():
    """Load the predictor once, remembering a load failure.

    A missing/corrupt model must produce a clear message on the pages that need
    it rather than a 500 on every request.
    """
    global _predictor, _predictor_error

    if _predictor is None and _predictor_error is None:
        try:
            _predictor = CirrhosisPredictor(
                current_app.config["MODEL_PATH"],
                current_app.config["ENCODER_PATH"],
                current_app.config["METRICS_PATH"],
            )
        except Exception as e:
            _predictor_error = str(e)
    if _predictor_error:
        raise ModelNotTrainedError(_predictor_error)
    return _predictor


def reset_predictor_cache():
    """Drop the cached model so the next request reloads it (used after retrain)."""
    global _predictor, _predictor_error
    _predictor = None
    _predictor_error = None


def client_ip():
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        # The left-most entry is the original client.
        return forwarded.split(",")[0].strip()
    return request.remote_addr or ""


def admin_required(func_):
    @wraps(func_)
    @login_required
    def wrapper(*args, **kwargs):
        if not current_user.is_admin:
            abort(403)
        return func_(*args, **kwargs)
    return wrapper


def owned_prediction_or_404(prediction_id):
    """Fetch a prediction the current user is allowed to touch.

    Returns 404 (not 403) for someone else's record so the endpoint doesn't
    confirm that an ID exists to a user who shouldn't see it. Admins may access
    any record.
    """
    record = db.session.get(Prediction, prediction_id)
    if record is None:
        abort(404)
    if record.user_id != current_user.id and not current_user.is_admin:
        abort(404)
    return record


def wants_json():
    return (
        request.path.startswith("/api/")
        or request.accept_mimetypes.best == "application/json"
    )


def _day_bucket_expression():
    """Date-truncation expression that works on SQLite and PostgreSQL alike.

    The original code used SQLite's `strftime`, which breaks the dashboard the
    moment the app is pointed at Postgres in deployment.
    """
    if current_app.config["SQLALCHEMY_DATABASE_URI"].startswith("sqlite"):
        return func.strftime("%Y-%m-%d", Prediction.created_at)
    return cast(Prediction.created_at, Date)


# ---------------------------------------------------------------------------
# Template helpers
# ---------------------------------------------------------------------------
def register_template_helpers(app):
    @app.context_processor
    def inject_globals():
        return {
            "current_year": datetime.now(timezone.utc).year,
            "ocr_available": is_ocr_available(),
            "field_meta": FIELD_META,
            "app_env": app.config.get("ENV_NAME", "development"),
        }

    @app.template_filter("pct")
    def pct(value, digits=1):
        if value is None:
            return "N/A"
        return f"{round(float(value) * 100, digits)}%"

    @app.template_filter("humandate")
    def humandate(value, fmt="%b %d, %Y %H:%M"):
        if not value:
            return "-"
        return value.strftime(fmt)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
def register_routes(app):

    # -- request lifecycle ------------------------------------------------
    @app.before_request
    def enforce_account_active():
        """Log out an account that an admin deactivated mid-session."""
        if current_user.is_authenticated and not current_user.is_active_account:
            logout_user()
            flash("Your account has been deactivated. Contact an administrator.", "warning")
            return redirect(url_for("login"))
        return None

    @app.after_request
    def security_headers(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault(
            "Permissions-Policy", "geolocation=(), microphone=(), camera=()"
        )
        return response

    # ------------------------------------------------------------------
    # Public pages
    # ------------------------------------------------------------------
    @app.route("/")
    def index():
        metrics = {}
        try:
            metrics = get_predictor().metrics
        except ModelNotTrainedError:
            pass
        return render_template("index.html", metrics=metrics)

    @app.route("/about")
    def about():
        metrics = {}
        try:
            metrics = get_predictor().metrics
        except ModelNotTrainedError:
            pass
        return render_template("about.html", metrics=metrics)

    @app.route("/services")
    def services():
        return render_template("services.html")

    @app.route("/contact", methods=["GET", "POST"])
    def contact():
        if request.method == "POST":
            name = request.form.get("name", "").strip()
            email = request.form.get("email", "").strip().lower()
            subject = request.form.get("subject", "").strip()
            message = request.form.get("message", "").strip()

            errors = []
            if len(name) < 2:
                errors.append("Please enter your name.")
            if "@" not in email or "." not in email.split("@")[-1]:
                errors.append("Please enter a valid email address.")
            if len(message) < 10:
                errors.append("Please write a message of at least 10 characters.")

            if errors:
                for error in errors:
                    flash(error, "danger")
                return render_template(
                    "contact.html", name=name, email=email, subject=subject, message=message
                )

            db.session.add(ContactMessage(
                name=name, email=email, subject=subject[:200], message=message
            ))
            db.session.commit()
            flash("Thanks for reaching out. Our team will respond within 1-2 business days.",
                  "success")
            return redirect(url_for("contact"))

        return render_template("contact.html")

    @app.route("/health")
    def health():
        """Liveness/readiness probe for deployment platforms."""
        checks = {"database": "unknown", "model": "unknown", "ocr": "unknown"}
        healthy = True

        try:
            db.session.execute(db.text("SELECT 1"))
            checks["database"] = "ok"
        except Exception as e:
            checks["database"] = f"error: {e}"
            healthy = False

        try:
            get_predictor()
            checks["model"] = "ok"
        except ModelNotTrainedError as e:
            checks["model"] = f"unavailable: {e}"
            healthy = False

        # OCR is optional, so it never fails the health check.
        checks["ocr"] = "ok" if is_ocr_available() else "unavailable (optional)"

        return jsonify({
            "status": "healthy" if healthy else "degraded",
            "checks": checks,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }), (200 if healthy else 503)

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------
    @app.route("/register", methods=["GET", "POST"])
    def register():
        if current_user.is_authenticated:
            return redirect(url_for("dashboard"))

        if request.method == "POST":
            full_name = request.form.get("full_name", "").strip()
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")
            confirm_password = request.form.get("confirm_password", "")
            organisation = request.form.get("organisation", "").strip()
            specialisation = request.form.get("specialisation", "").strip()

            min_length = app.config["PASSWORD_MIN_LENGTH"]
            errors = []
            if len(full_name) < 2:
                errors.append("Please enter your full name.")
            if "@" not in email or "." not in email.split("@")[-1]:
                errors.append("Please enter a valid email address.")
            if len(password) < min_length:
                errors.append(f"Password must be at least {min_length} characters long.")
            elif password.isdigit() or password.isalpha():
                errors.append("Password must mix letters and numbers.")
            if password != confirm_password:
                errors.append("Passwords do not match.")
            if User.query.filter_by(email=email).first():
                errors.append("An account with this email already exists.")

            if errors:
                for error in errors:
                    flash(error, "danger")
                return render_template(
                    "register.html", full_name=full_name, email=email,
                    organisation=organisation, specialisation=specialisation,
                )

            user = User(
                full_name=full_name, email=email, role="user",
                organisation=organisation[:140], specialisation=specialisation[:140],
            )
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            log_action(user.id, "Account registered", client_ip(), category="auth")

            flash("Account created successfully. Please log in.", "success")
            return redirect(url_for("login"))

        return render_template("register.html")

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if current_user.is_authenticated:
            return redirect(url_for("dashboard"))

        if request.method == "POST":
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")
            remember = bool(request.form.get("remember"))

            user = User.query.filter_by(email=email).first()

            if user and user.is_locked:
                minutes_left = max(
                    1, int((user.locked_until - datetime.now(timezone.utc).replace(tzinfo=None)).total_seconds() // 60)
                )
                flash(
                    f"Too many failed attempts. This account is locked for {minutes_left} "
                    "more minute(s).",
                    "danger",
                )
                return render_template("login.html", email=email)

            if user and user.check_password(password):
                if not user.is_active_account:
                    log_action(user.id, "Blocked login (account deactivated)", client_ip(),
                               category="auth")
                    flash("This account has been deactivated. Contact an administrator.",
                          "warning")
                    return render_template("login.html", email=email)

                user.register_successful_login()
                db.session.commit()
                login_user(user, remember=remember)
                session.permanent = True
                log_action(user.id, "Logged in", client_ip(), category="auth")
                flash(f"Welcome back, {user.full_name.split(' ')[0]}.", "success")

                next_page = request.args.get("next", "")
                # Only follow a same-site relative path, so `next` can't be used
                # as an open redirect to an attacker's domain.
                if next_page.startswith("/") and not next_page.startswith("//"):
                    return redirect(next_page)
                return redirect(url_for("dashboard"))

            if user:
                user.register_failed_login(
                    app.config["MAX_LOGIN_ATTEMPTS"], app.config["LOGIN_LOCKOUT_MINUTES"]
                )
                db.session.commit()
                log_action(user.id, "Failed login attempt", client_ip(), category="auth")

            flash("Invalid email or password.", "danger")
            return render_template("login.html", email=email)

        return render_template("login.html")

    @app.route("/logout")
    @login_required
    def logout():
        log_action(current_user.id, "Logged out", client_ip(), category="auth")
        logout_user()
        flash("You have been logged out.", "info")
        return redirect(url_for("index"))

    @app.route("/forgot-password", methods=["GET", "POST"])
    def forgot_password():
        """Issue a signed, time-limited reset link.

        No mail server is configured in this project, so the link is shown
        on-screen in development. In production, wire `send_reset_email` to your
        SMTP provider and stop echoing the URL.
        """
        if request.method == "POST":
            email = request.form.get("email", "").strip().lower()
            user = User.query.filter_by(email=email).first()

            reset_url = None
            if user and user.is_active_account:
                token = user.generate_reset_token(app.config["SECRET_KEY"])
                reset_url = url_for("reset_password", token=token, _external=True)
                log_action(user.id, "Requested password reset", client_ip(), category="auth")

            # Same response either way, so the page can't be used to enumerate
            # which email addresses have accounts.
            flash(
                "If an account with that email exists, reset instructions have been issued.",
                "info",
            )
            if reset_url and app.config.get("ENV_NAME") != "production":
                return render_template("forgot_password.html", dev_reset_url=reset_url)
            return redirect(url_for("login"))

        return render_template("forgot_password.html")

    @app.route("/reset-password/<token>", methods=["GET", "POST"])
    def reset_password(token):
        user = User.verify_reset_token(
            token, app.config["SECRET_KEY"], app.config["RESET_TOKEN_MAX_AGE_SECONDS"]
        )
        if user is None:
            flash("That reset link is invalid or has expired. Please request a new one.",
                  "danger")
            return redirect(url_for("forgot_password"))

        if request.method == "POST":
            password = request.form.get("password", "")
            confirm_password = request.form.get("confirm_password", "")
            min_length = app.config["PASSWORD_MIN_LENGTH"]

            errors = []
            if len(password) < min_length:
                errors.append(f"Password must be at least {min_length} characters long.")
            elif password.isdigit() or password.isalpha():
                errors.append("Password must mix letters and numbers.")
            if password != confirm_password:
                errors.append("Passwords do not match.")

            if errors:
                for error in errors:
                    flash(error, "danger")
                return render_template("reset_password.html", token=token)

            user.set_password(password)
            user.failed_login_attempts = 0
            user.locked_until = None
            db.session.commit()
            log_action(user.id, "Password reset completed", client_ip(), category="auth")

            flash("Password updated. Please log in with your new password.", "success")
            return redirect(url_for("login"))

        return render_template("reset_password.html", token=token)

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------
    @app.route("/dashboard")
    @login_required
    def dashboard():
        is_admin = current_user.is_admin

        if is_admin:
            base_query = Prediction.query
            user_filter = True
        else:
            base_query = Prediction.query.filter_by(user_id=current_user.id)
            user_filter = Prediction.user_id == current_user.id

        total_patients = base_query.count()
        high_risk = base_query.filter_by(result="High Risk").count()
        low_risk = base_query.filter_by(result="Low Risk").count()
        recent = base_query.order_by(Prediction.created_at.desc()).limit(6).all()

        # 30-day trend, split by outcome
        thirty_days_ago = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=30)
        bucket = _day_bucket_expression()
        trend_query = db.session.query(
            bucket.label("day"), Prediction.result, func.count(Prediction.id)
        ).filter(Prediction.created_at >= thirty_days_ago)
        if not is_admin:
            trend_query = trend_query.filter(Prediction.user_id == current_user.id)
        rows = trend_query.group_by("day", Prediction.result).order_by("day").all()

        per_day: dict[str, dict[str, int]] = {}
        for day, result, count in rows:
            key = day if isinstance(day, str) else day.strftime("%Y-%m-%d")
            bucket_entry = per_day.setdefault(key, {"High Risk": 0, "Low Risk": 0})
            if result in bucket_entry:
                bucket_entry[result] = count

        trend_labels = sorted(per_day)
        trend_high = [per_day[d]["High Risk"] for d in trend_labels]
        trend_low = [per_day[d]["Low Risk"] for d in trend_labels]

        # Risk-band split for the doughnut
        band_query = db.session.query(
            Prediction.risk_level, func.count(Prediction.id)
        )
        if not is_admin:
            band_query = band_query.filter(Prediction.user_id == current_user.id)
        band_rows = band_query.group_by(Prediction.risk_level).all()
        band_counts = {level: count for level, count in band_rows if level}

        # Stage distribution
        stage_query = db.session.query(
            Prediction.stage, func.count(Prediction.id)
        )
        if not is_admin:
            stage_query = stage_query.filter(Prediction.user_id == current_user.id)
        stage_rows = stage_query.group_by(Prediction.stage).order_by(Prediction.stage).all()
        stage_counts = {int(stage): count for stage, count in stage_rows if stage is not None}

        avg_query = db.session.query(
            func.avg(Prediction.probability),
            func.avg(Prediction.age),
            func.avg(Prediction.bilirubin),
            func.avg(Prediction.albumin),
        )
        if not is_admin:
            avg_query = avg_query.filter(Prediction.user_id == current_user.id)
        averages = avg_query.first()

        extra = {}
        if is_admin:
            total_users = User.query.count()
            active_users = User.query.filter_by(is_active_account=True).count()
            top_users = (
                db.session.query(
                    User.full_name, User.email, func.count(Prediction.id).label("cnt")
                )
                .join(Prediction, Prediction.user_id == User.id)
                .group_by(User.id, User.full_name, User.email)
                .order_by(func.count(Prediction.id).desc())
                .limit(5)
                .all()
            )
            extra = dict(
                total_users=total_users,
                active_users=active_users,
                top_users=top_users,
            )

        return render_template(
            "dashboard.html",
            total_patients=total_patients,
            high_risk=high_risk,
            low_risk=low_risk,
            recent=recent,
            trend_labels=trend_labels,
            trend_high=trend_high,
            trend_low=trend_low,
            band_counts=band_counts,
            stage_counts=stage_counts,
            avg_probability=round((averages[0] or 0) * 100, 1),
            avg_age=round(averages[1] or 0, 1),
            avg_bilirubin=round(averages[2] or 0, 2),
            avg_albumin=round(averages[3] or 0, 2),
            is_admin=is_admin,
            **extra,
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    @app.route("/prediction", methods=["GET", "POST"])
    @login_required
    def prediction():
        try:
            predictor = get_predictor()
        except ModelNotTrainedError as e:
            flash(f"The prediction model isn't available: {e}", "danger")
            return render_template("prediction.html", model_error=str(e))

        if request.method == "POST":
            form = request.form

            try:
                cleaned = validate_clinical_input(form)
            except ValidationError as e:
                flash("Please correct the highlighted fields.", "danger")
                return render_template(
                    "prediction.html", form_data=form, field_errors=e.errors
                )

            result_label, probability, risk_level = predictor.predict(cleaned)
            explanation = predictor.explain(cleaned)
            scores = compute_all(cleaned)

            record = Prediction(
                user_id=current_user.id,
                patient_name=(form.get("patient_name") or "").strip()[:120] or "Unnamed Patient",
                patient_id=(form.get("patient_id") or "").strip()[:60],
                result=result_label,
                probability=probability,
                risk_level=risk_level,
                model_name=predictor.model_name,
                explanation_json=json.dumps(explanation),
                meld_score=scores["meld"]["value"],
                fib4_score=scores["fib4"]["value"],
                apri_score=scores["apri"]["value"],
                source=form.get("source") or "manual",
                **cleaned,
            )
            db.session.add(record)
            db.session.commit()
            log_action(current_user.id, f"Ran prediction #{record.id}", client_ip(),
                       category="prediction")

            recommendations = get_recommendations(risk_level, result_label)

            return render_template(
                "prediction.html",
                result=record,
                probability_pct=record.probability_pct,
                explanation=explanation,
                scores=scores,
                abnormal=flag_abnormal(cleaned),
                top_features=predictor.labelled_top_features(5),
                recommendations=recommendations,
            )

        return render_template("prediction.html")

    @app.route("/prediction/<int:prediction_id>")
    @login_required
    def prediction_detail(prediction_id):
        record = owned_prediction_or_404(prediction_id)
        scores = compute_all({k: v for k, v in record.clinical_inputs().items() if v is not None})
        recommendations = get_recommendations(
            record.risk_level or "Low", record.result or "Low Risk",
        )
        return render_template(
            "prediction_detail.html",
            record=record,
            scores=scores,
            explanation=record.explanation,
            abnormal=flag_abnormal(
                {k: v for k, v in record.clinical_inputs().items() if v is not None}
            ),
            recommendations=recommendations,
        )

    @app.route("/prediction/extract", methods=["POST"])
    @login_required
    def extract_report():
        """OCR an uploaded lab report and return any values it could read.

        The manual form always stays editable, so a partial or failed read is
        never a dead end.
        """
        if "report" not in request.files:
            return jsonify({"error": "No file uploaded."}), 400

        file = request.files["report"]
        if not file or not file.filename:
            return jsonify({"error": "No file selected."}), 400

        filename = secure_filename(file.filename)
        extension = os.path.splitext(filename)[1].lower()
        if extension not in app.config["ALLOWED_REPORT_EXTENSIONS"]:
            return jsonify({
                "error": f"Unsupported file type '{extension}'. Upload a JPG, PNG or PDF."
            }), 400

        # Werkzeug already rejects bodies over MAX_CONTENT_LENGTH before we get
        # here, so reading now can't blow up memory.
        file_bytes = file.read()
        if not file_bytes:
            return jsonify({"error": "The uploaded file is empty."}), 400

        try:
            result = extract_lab_report(file_bytes, filename)
        except OCRUnavailableError as e:
            return jsonify({"error": str(e), "ocr_unavailable": True}), 503
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:  # pragma: no cover - defensive
            return jsonify({"error": f"Could not read the report: {e}"}), 422

        log_action(current_user.id, f"OCR extraction from {filename}", client_ip(),
                   category="prediction")

        fields_found = result["fields_found"]
        return jsonify({
            "extracted_fields": result["extracted_fields"],
            "fields_found": fields_found,
            "message": (
                f"Extracted {fields_found} value(s). Review them and complete the remaining "
                "fields manually."
                if fields_found
                else "No values could be read confidently. Enter them manually, or try a "
                     "clearer scan."
            ),
        })

    @app.route("/prediction/<int:prediction_id>/notes", methods=["POST"])
    @login_required
    def save_notes(prediction_id):
        record = owned_prediction_or_404(prediction_id)
        record.doctor_notes = request.form.get("notes", "").strip()[:5000]
        db.session.commit()
        log_action(current_user.id, f"Updated notes on prediction #{prediction_id}",
                   client_ip(), category="prediction")
        flash("Notes saved.", "success")
        return redirect(request.referrer or url_for("history"))

    # ------------------------------------------------------------------
    # Batch prediction
    # ------------------------------------------------------------------
    @app.route("/batch", methods=["GET", "POST"])
    @admin_required
    def batch_predict():
        """Score a whole CSV of patients in one pass.

        Rows that fail validation are reported individually so one bad row
        doesn't discard the rest of the file.
        """
        try:
            predictor = get_predictor()
        except ModelNotTrainedError as e:
            flash(f"The prediction model isn't available: {e}", "danger")
            return render_template("batch.html", model_error=str(e))

        if request.method == "POST":
            file = request.files.get("batch_file")
            if not file or not file.filename:
                flash("Please choose a CSV file to upload.", "danger")
                return render_template("batch.html")

            extension = os.path.splitext(secure_filename(file.filename))[1].lower()
            if extension not in app.config["ALLOWED_BATCH_EXTENSIONS"]:
                flash("Only .csv files are supported for batch prediction.", "danger")
                return render_template("batch.html")

            try:
                dataframe = pd.read_csv(file)
            except Exception as e:
                flash(f"Could not read that CSV: {e}", "danger")
                return render_template("batch.html")

            if dataframe.empty:
                flash("That CSV has no data rows.", "warning")
                return render_template("batch.html")

            MAX_BATCH_ROWS = 500
            if len(dataframe) > MAX_BATCH_ROWS:
                flash(
                    f"That file has {len(dataframe)} rows. Only the first {MAX_BATCH_ROWS} "
                    "were processed.",
                    "warning",
                )
                dataframe = dataframe.head(MAX_BATCH_ROWS)

            records = rows_from_dataframe(dataframe)
            outcomes = predictor.predict_batch(records)

            saved, failed = [], []
            for outcome in outcomes:
                if not outcome["ok"]:
                    failed.append(outcome)
                    continue
                cleaned = outcome["cleaned"]
                scores = compute_all(cleaned)
                record = Prediction(
                    user_id=current_user.id,
                    patient_name=outcome["patient_name"][:120],
                    patient_id=outcome["patient_id"][:60],
                    result=outcome["result"],
                    probability=outcome["probability"],
                    risk_level=outcome["risk_level"],
                    model_name=predictor.model_name,
                    meld_score=scores["meld"]["value"],
                    fib4_score=scores["fib4"]["value"],
                    apri_score=scores["apri"]["value"],
                    source="batch",
                    **cleaned,
                )
                db.session.add(record)
                saved.append(record)

            db.session.commit()
            log_action(
                current_user.id,
                f"Batch prediction: {len(saved)} saved, {len(failed)} rejected",
                client_ip(), category="prediction",
            )

            if saved:
                # Remember the batch so the summary PDF can be generated for it.
                session["last_batch_ids"] = [r.id for r in saved]

            flash(
                f"Processed {len(records)} row(s): {len(saved)} scored, {len(failed)} rejected.",
                "success" if saved else "warning",
            )
            return render_template(
                "batch.html", saved=saved, failed=failed, processed=len(records)
            )

        return render_template("batch.html")

    @app.route("/batch/template.csv")
    @admin_required
    def batch_template():
        """Download a correctly-shaped CSV so users don't guess the columns."""
        output = io.StringIO()
        writer = csv.writer(output)
        columns = ["patient_name", "patient_id"] + REQUIRED_FORM_FIELDS
        writer.writerow(columns)
        writer.writerow([
            "Jane Doe", "MRN-00123", 58, "F", 3.4, 320, 2.9, 180, 1980, 112, 158, 195,
            12.4, "Y", "Y", "N", "S", 3,
        ])
        writer.writerow([
            "John Smith", "MRN-00124", 44, "M", 0.8, 190, 4.2, 32, 110, 28, 120, 260,
            10.4, "N", "N", "N", "N", 1,
        ])

        return send_file(
            io.BytesIO(output.getvalue().encode("utf-8")),
            mimetype="text/csv", as_attachment=True,
            download_name="hepatoai_batch_template.csv",
        )

    @app.route("/batch/summary.pdf")
    @admin_required
    def batch_summary_pdf():
        ids = session.get("last_batch_ids") or []
        if not ids:
            flash("No recent batch to export. Run a batch prediction first.", "warning")
            return redirect(url_for("batch_predict"))

        records = (
            Prediction.query
            .filter(Prediction.id.in_(ids), Prediction.user_id == current_user.id)
            .order_by(Prediction.id)
            .all()
        )
        if not records:
            flash("Those batch records are no longer available.", "warning")
            return redirect(url_for("batch_predict"))

        buffer = io.BytesIO()
        build_batch_pdf(buffer, records, clinician_name=current_user.full_name)
        buffer.seek(0)
        return send_file(
            buffer, mimetype="application/pdf", as_attachment=True,
            download_name=f"hepatoai_batch_summary_{datetime.now(timezone.utc):%Y%m%d_%H%M}.pdf",
        )

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------
    @app.route("/compare")
    @admin_required
    def compare():
        """Side-by-side comparison of up to four selected predictions.

        Useful for tracking one patient across visits, or contrasting patients.
        """
        raw_ids = request.args.getlist("ids", type=int)[:4]
        records = []
        if raw_ids:
            records = (
                Prediction.query
                .filter(Prediction.id.in_(raw_ids), Prediction.user_id == current_user.id)
                .order_by(Prediction.created_at)
                .all()
            )

        available = (
            Prediction.query.filter_by(user_id=current_user.id)
            .order_by(Prediction.created_at.desc())
            .limit(60)
            .all()
        )

        score_map = {
            record.id: compute_all(
                {k: v for k, v in record.clinical_inputs().items() if v is not None}
            )
            for record in records
        }

        return render_template(
            "compare.html", records=records, available=available, score_map=score_map,
            selected_ids=raw_ids,
        )

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------
    @app.route("/history")
    @login_required
    def history():
        page = request.args.get("page", 1, type=int)
        search = request.args.get("search", "").strip()
        risk_filter = request.args.get("risk", "")
        stage_filter = request.args.get("stage", type=int)
        sort = request.args.get("sort", "newest")

        if current_user.is_admin:
            query = Prediction.query
        else:
            query = Prediction.query.filter_by(user_id=current_user.id)
        if search:
            pattern = f"%{search}%"
            query = query.filter(
                db.or_(
                    Prediction.patient_name.ilike(pattern),
                    Prediction.patient_id.ilike(pattern),
                )
            )
        if risk_filter in ("High Risk", "Low Risk"):
            query = query.filter(Prediction.result == risk_filter)
        if stage_filter in (1, 2, 3, 4):
            query = query.filter(Prediction.stage == stage_filter)

        sort_options = {
            "newest": Prediction.created_at.desc(),
            "oldest": Prediction.created_at.asc(),
            "risk_high": Prediction.probability.desc(),
            "risk_low": Prediction.probability.asc(),
            "name": Prediction.patient_name.asc(),
        }
        query = query.order_by(sort_options.get(sort, sort_options["newest"]))

        pagination = db.paginate(
            query, page=page, per_page=app.config["ITEMS_PER_PAGE"], error_out=False
        )

        return render_template(
            "history.html", pagination=pagination, search=search,
            risk_filter=risk_filter, stage_filter=stage_filter, sort=sort,
        )

    @app.route("/history/<int:prediction_id>/delete", methods=["POST"])
    @login_required
    def delete_prediction(prediction_id):
        record = owned_prediction_or_404(prediction_id)
        db.session.delete(record)
        db.session.commit()
        log_action(current_user.id, f"Deleted prediction #{prediction_id}", client_ip(),
                   category="data")
        flash("Record deleted.", "info")
        return redirect(request.referrer or url_for("history"))

    @app.route("/history/bulk-delete", methods=["POST"])
    @login_required
    def bulk_delete_predictions():
        ids = request.form.getlist("prediction_ids", type=int)
        if not ids:
            flash("No records selected.", "warning")
            return redirect(url_for("history"))

        # Scope the delete to the current user so a forged ID can't touch
        # someone else's records.
        deleted = (
            Prediction.query
            .filter(Prediction.id.in_(ids), Prediction.user_id == current_user.id)
            .delete(synchronize_session=False)
        )
        db.session.commit()
        log_action(current_user.id, f"Bulk-deleted {deleted} prediction(s)", client_ip(),
                   category="data")
        flash(f"Deleted {deleted} record(s).", "info")
        return redirect(url_for("history"))

    # ------------------------------------------------------------------
    # Reports & exports
    # ------------------------------------------------------------------
    @app.route("/reports")
    @admin_required
    def reports():
        if current_user.is_admin:
            predictions = Prediction.query.order_by(Prediction.created_at.desc()).all()
        else:
            predictions = (
                Prediction.query.filter_by(user_id=current_user.id)
                .order_by(Prediction.created_at.desc())
                .all()
            )
        metrics = {}
        comparison = []
        try:
            predictor = get_predictor()
            metrics = predictor.metrics
            comparison = predictor.model_comparison
        except ModelNotTrainedError:
            flash("Model metrics are unavailable until the model is trained.", "warning")

        return render_template(
            "reports.html", predictions=predictions, metrics=metrics,
            model_comparison=comparison,
        )

    @app.route("/reports/csv")
    @admin_required
    def export_csv():
        if current_user.is_admin:
            predictions = Prediction.query.order_by(Prediction.created_at.desc()).all()
        else:
            predictions = (
                Prediction.query.filter_by(user_id=current_user.id)
                .order_by(Prediction.created_at.desc())
                .all()
            )

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "ID", "Patient Name", "Patient ID", "Age", "Gender", "Bilirubin", "Cholesterol",
            "Albumin", "Copper", "Alk_Phos", "SGOT", "Triglycerides", "Platelets",
            "Prothrombin", "Ascites", "Hepatomegaly", "Spiders", "Edema", "Stage",
            "Result", "Probability (%)", "Risk Level", "MELD", "FIB-4", "APRI",
            "Source", "Doctor Notes", "Date",
        ])
        for p in predictions:
            writer.writerow([
                p.id, p.patient_name, p.patient_id, p.age, p.gender, p.bilirubin,
                p.cholesterol, p.albumin, p.copper, p.alk_phos, p.sgot, p.triglycerides,
                p.platelets, p.prothrombin, p.ascites, p.hepatomegaly, p.spiders, p.edema,
                p.stage, p.result, p.probability_pct, p.risk_level, p.meld_score,
                p.fib4_score, p.apri_score, p.source,
                (p.doctor_notes or "").replace("\n", " "),
                p.created_at.strftime("%Y-%m-%d %H:%M"),
            ])

        return send_file(
            io.BytesIO(output.getvalue().encode("utf-8-sig")),
            mimetype="text/csv", as_attachment=True,
            download_name=f"prediction_history_{datetime.now(timezone.utc):%Y%m%d}.csv",
        )

    @app.route("/reports/pdf/<int:prediction_id>")
    @login_required
    def export_pdf(prediction_id):
        record = owned_prediction_or_404(prediction_id)

        buffer = io.BytesIO()
        build_prediction_pdf(buffer, record, clinician_name=current_user.full_name)
        buffer.seek(0)

        safe_name = "".join(
            c for c in (record.patient_name or "patient") if c.isalnum() or c in " -_"
        ).strip().replace(" ", "_") or "patient"

        log_action(current_user.id, f"Exported PDF for prediction #{record.id}", client_ip(),
                   category="data")
        return send_file(
            buffer, mimetype="application/pdf", as_attachment=True,
            download_name=f"hepatoai_report_{safe_name}_{record.id}.pdf",
        )

    # ------------------------------------------------------------------
    # Profile
    # ------------------------------------------------------------------
    @app.route("/profile", methods=["GET", "POST"])
    @login_required
    def profile():
        if request.method == "POST":
            action = request.form.get("action", "details")

            if action == "password":
                current_password = request.form.get("current_password", "")
                new_password = request.form.get("new_password", "")
                confirm_password = request.form.get("confirm_password", "")
                min_length = app.config["PASSWORD_MIN_LENGTH"]

                errors = []
                # Require the existing password: without this, a hijacked session
                # could silently change the account's credentials.
                if not current_user.check_password(current_password):
                    errors.append("Your current password is incorrect.")
                if len(new_password) < min_length:
                    errors.append(f"New password must be at least {min_length} characters.")
                elif new_password.isdigit() or new_password.isalpha():
                    errors.append("New password must mix letters and numbers.")
                if new_password != confirm_password:
                    errors.append("New passwords do not match.")

                if errors:
                    for error in errors:
                        flash(error, "danger")
                else:
                    current_user.set_password(new_password)
                    db.session.commit()
                    log_action(current_user.id, "Changed password", client_ip(),
                               category="auth")
                    flash("Password updated successfully.", "success")

            else:
                full_name = request.form.get("full_name", "").strip()
                if len(full_name) < 2:
                    flash("Please enter a valid name.", "danger")
                else:
                    current_user.full_name = full_name[:120]
                    current_user.organisation = request.form.get("organisation", "").strip()[:140]
                    current_user.specialisation = request.form.get("specialisation", "").strip()[:140]
                    current_user.phone = request.form.get("phone", "").strip()[:40]
                    db.session.commit()
                    flash("Profile updated successfully.", "success")

            return redirect(url_for("profile"))

        stats = (
            db.session.query(
                func.count(Prediction.id),
                func.sum(db.case((Prediction.result == "High Risk", 1), else_=0)),
                func.avg(Prediction.probability),
            )
            .filter(Prediction.user_id == current_user.id)
            .first()
        )
        recent_activity = (
            AuditLog.query.filter_by(user_id=current_user.id)
            .order_by(AuditLog.timestamp.desc())
            .limit(10)
            .all()
        )

        return render_template(
            "profile.html",
            total_predictions=stats[0] or 0,
            high_risk_count=int(stats[1] or 0),
            avg_probability=round((stats[2] or 0) * 100, 1),
            recent_activity=recent_activity,
        )

    @app.route("/profile/export")
    @login_required
    def export_my_data():
        """Download everything the system holds about this account (JSON)."""
        payload = {
            "account": current_user.to_dict(),
            "predictions": [
                p.to_dict() for p in
                Prediction.query.filter_by(user_id=current_user.id)
                .order_by(Prediction.created_at).all()
            ],
            "activity": [
                a.to_dict() for a in
                AuditLog.query.filter_by(user_id=current_user.id)
                .order_by(AuditLog.timestamp).all()
            ],
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }
        buffer = io.BytesIO(json.dumps(payload, indent=2).encode("utf-8"))
        return send_file(
            buffer, mimetype="application/json", as_attachment=True,
            download_name=f"hepatoai_my_data_{datetime.now(timezone.utc):%Y%m%d}.json",
        )

    @app.route("/profile/delete", methods=["POST"])
    @login_required
    def delete_my_account():
        password = request.form.get("password", "")
        if not current_user.check_password(password):
            flash("Password incorrect. Account not deleted.", "danger")
            return redirect(url_for("profile"))
        if current_user.is_admin and User.query.filter_by(role="admin").count() <= 1:
            flash("You are the only administrator. Promote another admin first.", "danger")
            return redirect(url_for("profile"))

        user_id, email = current_user.id, current_user.email
        logout_user()
        user = db.session.get(User, user_id)
        if user:
            db.session.delete(user)
            db.session.commit()
        log_action(None, f"Account self-deleted ({email})", client_ip(), category="admin",
                   actor_email=email)
        flash("Your account and all associated records have been deleted.", "info")
        return redirect(url_for("index"))

    # ------------------------------------------------------------------
    # Admin
    # ------------------------------------------------------------------
    @app.route("/admin")
    @admin_required
    def admin():
        total_users = User.query.count()
        active_users = User.query.filter_by(is_active_account=True).count()
        total_predictions = Prediction.query.count()
        high_risk = Prediction.query.filter_by(result="High Risk").count()
        unread_messages = ContactMessage.query.filter_by(is_read=False).count()

        users = User.query.order_by(User.created_at.desc()).all()
        recent_logs = AuditLog.query.order_by(AuditLog.timestamp.desc()).limit(25).all()

        # Per-user activity, for the admin table
        counts = dict(
            db.session.query(Prediction.user_id, func.count(Prediction.id))
            .group_by(Prediction.user_id).all()
        )

        model_info = {"available": False}
        try:
            predictor = get_predictor()
            model_info = {
                "available": True,
                "name": predictor.model_name,
                "trained_at": predictor.metrics.get("trained_at", "unknown"),
                "accuracy": predictor.metrics.get("accuracy"),
                "roc_auc": predictor.metrics.get("roc_auc"),
                "sklearn_version": predictor.metrics.get("sklearn_version", "unknown"),
                "is_stale": predictor.is_stale,
            }
        except ModelNotTrainedError as e:
            model_info["error"] = str(e)

        return render_template(
            "admin.html", total_users=total_users, active_users=active_users,
            total_predictions=total_predictions, high_risk=high_risk, users=users,
            recent_logs=recent_logs, prediction_counts=counts,
            unread_messages=unread_messages, model_info=model_info,
            ocr_available=is_ocr_available(),
        )

    @app.route("/admin/users/<int:user_id>/delete", methods=["POST"])
    @admin_required
    def admin_delete_user(user_id):
        if user_id == current_user.id:
            flash("You cannot delete your own account from here. Use your profile page.",
                  "danger")
            return redirect(url_for("admin"))

        user = db.session.get(User, user_id)
        if user is None:
            abort(404)
        if user.is_admin and User.query.filter_by(role="admin").count() <= 1:
            flash("Cannot delete the last remaining administrator.", "danger")
            return redirect(url_for("admin"))

        email = user.email
        db.session.delete(user)
        db.session.commit()
        log_action(current_user.id, f"Admin deleted user {email}", client_ip(),
                   category="admin")
        flash(f"User {email} and all their records were deleted.", "info")
        return redirect(url_for("admin"))

    @app.route("/admin/users/<int:user_id>/toggle-active", methods=["POST"])
    @admin_required
    def admin_toggle_active(user_id):
        user = db.session.get(User, user_id)
        if user is None:
            abort(404)
        if user.id == current_user.id:
            flash("You cannot deactivate your own account.", "danger")
            return redirect(url_for("admin"))

        user.is_active_account = not user.is_active_account
        # Clear any lockout when reactivating, so the user isn't locked out twice.
        if user.is_active_account:
            user.failed_login_attempts = 0
            user.locked_until = None
        db.session.commit()

        state = "activated" if user.is_active_account else "deactivated"
        log_action(current_user.id, f"Admin {state} user {user.email}", client_ip(),
                   category="admin")
        flash(f"User {state}.", "success")
        return redirect(url_for("admin"))

    @app.route("/admin/users/<int:user_id>/toggle-role", methods=["POST"])
    @admin_required
    def admin_toggle_role(user_id):
        user = db.session.get(User, user_id)
        if user is None:
            abort(404)
        if user.id == current_user.id:
            flash("You cannot change your own role.", "danger")
            return redirect(url_for("admin"))

        user.role = "user" if user.is_admin else "admin"
        db.session.commit()
        log_action(current_user.id, f"Admin set {user.email} role to {user.role}",
                   client_ip(), category="admin")
        flash(f"{user.email} is now a{'n administrator' if user.is_admin else ' standard user'}.",
              "success")
        return redirect(url_for("admin"))

    @app.route("/admin/messages")
    @admin_required
    def admin_messages():
        messages = ContactMessage.query.order_by(ContactMessage.created_at.desc()).all()
        return render_template("admin_messages.html", messages=messages)

    @app.route("/admin/messages/<int:message_id>/read", methods=["POST"])
    @admin_required
    def admin_mark_message_read(message_id):
        message = db.session.get(ContactMessage, message_id)
        if message is None:
            abort(404)
        message.is_read = not message.is_read
        db.session.commit()
        return redirect(url_for("admin_messages"))

    @app.route("/admin/messages/<int:message_id>/delete", methods=["POST"])
    @admin_required
    def admin_delete_message(message_id):
        message = db.session.get(ContactMessage, message_id)
        if message is None:
            abort(404)
        db.session.delete(message)
        db.session.commit()
        flash("Message deleted.", "info")
        return redirect(url_for("admin_messages"))

    @app.route("/admin/audit")
    @admin_required
    def admin_audit():
        page = request.args.get("page", 1, type=int)
        category = request.args.get("category", "")
        query = AuditLog.query
        if category:
            query = query.filter(AuditLog.category == category)
        pagination = db.paginate(
            query.order_by(AuditLog.timestamp.desc()), page=page, per_page=40, error_out=False
        )
        return render_template("admin_audit.html", pagination=pagination, category=category)

    @app.route("/admin/export/patients.csv")
    @admin_required
    def admin_export_patients():
        rows = (
            db.session.query(Prediction, User.email)
            .join(User, Prediction.user_id == User.id)
            .order_by(Prediction.created_at.desc())
            .all()
        )
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "ID", "Clinician Email", "Patient", "Patient ID", "Age", "Gender", "Stage",
            "Result", "Probability (%)", "Risk Level", "MELD", "Source", "Created At",
        ])
        for prediction, email in rows:
            writer.writerow([
                prediction.id, email, prediction.patient_name, prediction.patient_id,
                prediction.age, prediction.gender, prediction.stage, prediction.result,
                prediction.probability_pct, prediction.risk_level, prediction.meld_score,
                prediction.source, prediction.created_at.strftime("%Y-%m-%d %H:%M"),
            ])

        log_action(current_user.id, "Admin exported all patient data", client_ip(),
                   category="admin")
        return send_file(
            io.BytesIO(output.getvalue().encode("utf-8-sig")),
            mimetype="text/csv", as_attachment=True, download_name="all_patients.csv",
        )

    @app.route("/admin/reload-model", methods=["POST"])
    @admin_required
    def admin_reload_model():
        """Pick up newly-trained artefacts without restarting the server."""
        reset_predictor_cache()
        try:
            predictor = get_predictor()
            flash(f"Model reloaded: {predictor.model_name}.", "success")
            log_action(current_user.id, "Reloaded prediction model", client_ip(),
                       category="admin")
        except ModelNotTrainedError as e:
            flash(f"Reload failed: {e}", "danger")
        return redirect(url_for("admin"))

    # ------------------------------------------------------------------
    # JSON API
    # ------------------------------------------------------------------
    @app.route("/api/predict", methods=["POST"])
    @login_required
    @csrf.exempt  # session-authenticated JSON endpoint for scripted clients
    def api_predict():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Request body must be a JSON object."}), 400

        try:
            predictor = get_predictor()
        except ModelNotTrainedError as e:
            return jsonify({"error": str(e)}), 503

        try:
            cleaned = validate_clinical_input(payload)
        except ValidationError as e:
            return jsonify({"error": "Validation failed.", "fields": e.errors}), 422

        label, probability, band = predictor.predict(cleaned)
        explanation = predictor.explain(cleaned)
        scores = compute_all(cleaned)

        record = None
        if payload.get("save", True):
            record = Prediction(
                user_id=current_user.id,
                patient_name=str(payload.get("patient_name") or "API Patient")[:120],
                patient_id=str(payload.get("patient_id") or "")[:60],
                result=label, probability=probability, risk_level=band,
                model_name=predictor.model_name,
                explanation_json=json.dumps(explanation),
                meld_score=scores["meld"]["value"],
                fib4_score=scores["fib4"]["value"],
                apri_score=scores["apri"]["value"],
                source="api", **cleaned,
            )
            db.session.add(record)
            db.session.commit()
            log_action(current_user.id, f"API prediction #{record.id}", client_ip(),
                       category="prediction")

        return jsonify({
            "id": record.id if record else None,
            "result": label,
            "probability": round(probability * 100, 2),
            "risk_level": band,
            "model": predictor.model_name,
            "explanation": explanation,
            "clinical_scores": {
                key: {"value": value["value"], "band": value["band"]}
                for key, value in scores.items() if isinstance(value, dict)
            },
            "abnormal_flags": {
                field: flag["status"] for field, flag in flag_abnormal(cleaned).items()
            },
            "disclaimer": "Decision support only. Not a medical diagnosis.",
        })

    @app.route("/api/patients")
    @login_required
    def api_patients():
        page = request.args.get("page", 1, type=int)
        per_page = min(request.args.get("per_page", 50, type=int), 200)
        pagination = db.paginate(
            Prediction.query.filter_by(user_id=current_user.id)
            .order_by(Prediction.created_at.desc()),
            page=page, per_page=per_page, error_out=False,
        )
        return jsonify({
            "items": [p.to_dict() for p in pagination.items],
            "page": pagination.page,
            "pages": pagination.pages,
            "total": pagination.total,
        })

    @app.route("/api/patients/<int:prediction_id>")
    @login_required
    def api_patient_detail(prediction_id):
        record = owned_prediction_or_404(prediction_id)
        payload = record.to_dict()
        payload["clinical_inputs"] = record.clinical_inputs()
        payload["explanation"] = record.explanation
        return jsonify(payload)

    @app.route("/api/history")
    @login_required
    def api_history():
        records = (
            Prediction.query.filter_by(user_id=current_user.id)
            .order_by(Prediction.created_at.desc()).limit(50).all()
        )
        return jsonify([r.to_dict() for r in records])

    @app.route("/api/dashboard")
    @login_required
    def api_dashboard():
        query = Prediction.query.filter_by(user_id=current_user.id)
        band_rows = (
            db.session.query(Prediction.risk_level, func.count(Prediction.id))
            .filter(Prediction.user_id == current_user.id)
            .group_by(Prediction.risk_level).all()
        )
        return jsonify({
            "total": query.count(),
            "high_risk": query.filter_by(result="High Risk").count(),
            "low_risk": query.filter_by(result="Low Risk").count(),
            "by_risk_band": {level: count for level, count in band_rows if level},
        })

    @app.route("/api/model")
    @login_required
    def api_model():
        try:
            predictor = get_predictor()
        except ModelNotTrainedError as e:
            return jsonify({"available": False, "error": str(e)}), 503
        return jsonify({
            "available": True,
            "name": predictor.model_name,
            "metrics": {
                key: predictor.metrics.get(key) for key in
                ("accuracy", "precision", "recall", "f1_score", "roc_auc",
                 "brier_score", "trained_at", "sklearn_version")
            },
            "feature_importance": dict(predictor.top_features(16)),
            "model_comparison": predictor.model_comparison,
        })

    @app.route("/api/schema")
    def api_schema():
        """Self-documenting field schema, so API clients don't guess."""
        from preprocessing import VALID_RANGES, REFERENCE_RANGES

        return jsonify({
            "required_fields": REQUIRED_FORM_FIELDS,
            "fields": {
                field: {
                    **meta,
                    "valid_range": VALID_RANGES.get(field),
                    "reference_range": REFERENCE_RANGES.get(field),
                }
                for field, meta in FIELD_META.items()
            },
        })

    @app.route("/api/recommendations/<int:prediction_id>")
    @login_required
    def api_recommendations(prediction_id):
        """Return specialists sorted by distance from the user's location."""
        record = owned_prediction_or_404(prediction_id)
        lat = request.args.get("lat", type=float)
        lng = request.args.get("lng", type=float)
        recs = get_recommendations(
            record.risk_level or "Low",
            record.result or "Low Risk",
            user_lat=lat,
            user_lng=lng,
        )
        return jsonify({
            "urgency": recs["urgency"],
            "message": recs["message"],
            "specialists": recs["specialists"],
            "actions": recs["actions"],
            "location_used": lat is not None and lng is not None,
        })


# ---------------------------------------------------------------------------
# Health Assistant chat API
# ---------------------------------------------------------------------------
def register_health_chat(app):
    @app.route("/api/health-chat", methods=["POST"])
    def api_health_chat():
        data = request.get_json(silent=True) or {}
        message = (data.get("message") or "").strip()
        if not message:
            return jsonify({"error": "Empty message."}), 400

        latest_prediction = None
        if current_user.is_authenticated:
            latest_prediction = (
                Prediction.query
                .filter_by(user_id=current_user.id)
                .order_by(Prediction.created_at.desc())
                .first()
            )

        history = data.get("history")
        reply, is_ai = get_health_response(message, prediction=latest_prediction, history=history)
        return jsonify({"reply": reply, "ai_powered": is_ai})


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------
def register_error_handlers(app):
    @app.errorhandler(400)
    def bad_request(e):
        if wants_json():
            return jsonify({"error": "Bad request."}), 400
        return render_template("error.html", code=400, title="Bad Request",
                               message="The request couldn't be understood."), 400

    @app.errorhandler(403)
    def forbidden(e):
        if wants_json():
            return jsonify({"error": "Forbidden."}), 403
        return render_template("error.html", code=403, title="Access Denied",
                               message="You don't have permission to view this page."), 403

    @app.errorhandler(404)
    def not_found(e):
        if wants_json():
            return jsonify({"error": "Not found."}), 404
        return render_template("error.html", code=404, title="Page Not Found",
                               message="We couldn't find the page you were looking for."), 404

    @app.errorhandler(413)
    def payload_too_large(e):
        limit = app.config["MAX_REPORT_SIZE_MB"]
        if wants_json():
            return jsonify({"error": f"File too large. Maximum size is {limit}MB."}), 413
        return render_template("error.html", code=413, title="File Too Large",
                               message=f"Uploads are limited to {limit}MB."), 413

    @app.errorhandler(CSRFError)
    def csrf_error(e):
        if wants_json():
            return jsonify({"error": "CSRF token missing or expired."}), 400
        flash("Your session expired. Please try that again.", "warning")
        return redirect(request.referrer or url_for("index")), 302

    @app.errorhandler(500)
    def server_error(e):
        db.session.rollback()
        app.logger.exception("Unhandled server error")
        if wants_json():
            return jsonify({"error": "Internal server error."}), 500
        return render_template("error.html", code=500, title="Something Went Wrong",
                               message="An unexpected error occurred. Please try again."), 500

    @app.errorhandler(ModelNotTrainedError)
    def model_missing(e):
        if wants_json():
            return jsonify({"error": str(e)}), 503
        return render_template("error.html", code=503, title="Model Unavailable",
                               message=str(e)), 503


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def bootstrap_database(app=None):
    """Create tables and seed the default administrator."""
    app = app or create_app()
    with app.app_context():
        db.create_all()

        admin_email = app.config["ADMIN_EMAIL"]
        if not User.query.filter_by(email=admin_email).first():
            admin_user = User(
                full_name="System Administrator", email=admin_email, role="admin",
                organisation="Hepato AI", specialisation="Administration",
            )
            admin_user.set_password(app.config["ADMIN_PASSWORD"])
            db.session.add(admin_user)
            db.session.commit()
            print(f"  Created default admin: {admin_email} / {app.config['ADMIN_PASSWORD']}")
            if app.config["ADMIN_PASSWORD"] == "Admin@123":
                print("  WARNING: change ADMIN_PASSWORD before deploying.")
    return app


# The module-level `app` keeps `flask run` and `gunicorn app:app` working.
app = create_app()


if __name__ == "__main__":
    print("Bootstrapping database...")
    bootstrap_database(app)
    print("Starting Hepato AI on http://127.0.0.1:5000")
    app.run(debug=os.environ.get("FLASK_DEBUG", "1") == "1", host="127.0.0.1", port=5000)
