"""
database.py — saxenashvam0321@gmail.com
------------
SQLAlchemy ORM models for the Liver Cirrhosis Prediction System.

Tables:
 - User           : registered users / admins (auth, roles, lockout state)
 - Prediction     : every prediction made, tied to the user who ran it
 - AuditLog       : security / activity trail
 - ContactMessage : messages submitted through the public contact form
"""

from datetime import datetime, timedelta, timezone

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()


# ---------------------------------------------------------------------------
# User
# ---------------------------------------------------------------------------
class User(db.Model, UserMixin):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), default="user", nullable=False)  # 'user' | 'admin'
    is_active_account = db.Column(db.Boolean, default=True, nullable=False)

    # Optional profile fields
    organisation = db.Column(db.String(140), default="")
    specialisation = db.Column(db.String(140), default="")
    phone = db.Column(db.String(40), default="")

    # Auth hardening / activity
    failed_login_attempts = db.Column(db.Integer, default=0, nullable=False)
    locked_until = db.Column(db.DateTime, nullable=True)
    last_login_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None), nullable=False)

    predictions = db.relationship(
        "Prediction", backref="user", lazy=True, cascade="all, delete-orphan"
    )
    # Keep the audit trail but detach it from the deleted user, so deleting a
    # user never leaves rows pointing at a missing FK.
    audit_logs = db.relationship("AuditLog", backref="user", lazy=True, passive_deletes=True)

    # --- password ---------------------------------------------------------
    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    # --- roles / state ----------------------------------------------------
    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    @property
    def is_active(self) -> bool:
        """Flask-Login calls this to decide whether a session may be created.

        Overriding UserMixin.is_active here is what actually stops deactivated
        accounts from logging in.
        """
        return bool(self.is_active_account)

    @property
    def is_locked(self) -> bool:
        return self.locked_until is not None and self.locked_until > datetime.now(timezone.utc).replace(tzinfo=None)

    def lock_for(self, minutes: int) -> None:
        self.locked_until = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(minutes=minutes)

    def register_failed_login(self, max_attempts: int, lockout_minutes: int) -> None:
        self.failed_login_attempts = (self.failed_login_attempts or 0) + 1
        if self.failed_login_attempts >= max_attempts:
            self.lock_for(lockout_minutes)
            self.failed_login_attempts = 0

    def register_successful_login(self) -> None:
        self.failed_login_attempts = 0
        self.locked_until = None
        self.last_login_at = datetime.now(timezone.utc).replace(tzinfo=None)

    # --- password reset tokens -------------------------------------------
    def generate_reset_token(self, secret_key: str) -> str:
        from itsdangerous import URLSafeTimedSerializer

        serializer = URLSafeTimedSerializer(secret_key, salt="liverai-password-reset")
        # Bind the token to the current hash so it becomes invalid once used.
        return serializer.dumps({"uid": self.id, "pw": self.password_hash[-16:]})

    @staticmethod
    def verify_reset_token(token: str, secret_key: str, max_age: int = 3600):
        from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired

        serializer = URLSafeTimedSerializer(secret_key, salt="liverai-password-reset")
        try:
            data = serializer.loads(token, max_age=max_age)
        except (BadSignature, SignatureExpired):
            return None

        user = db.session.get(User, data.get("uid"))
        if user is None:
            return None
        # Reject a token minted against a password that has since changed.
        if user.password_hash[-16:] != data.get("pw"):
            return None
        return user

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "full_name": self.full_name,
            "email": self.email,
            "role": self.role,
            "is_active": self.is_active_account,
            "organisation": self.organisation,
            "specialisation": self.specialisation,
            "prediction_count": len(self.predictions),
            "last_login_at": self.last_login_at.isoformat() if self.last_login_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self) -> str:
        return f"<User {self.email} ({self.role})>"


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
class Prediction(db.Model):
    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
        db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )

    patient_name = db.Column(db.String(120), default="Unnamed Patient")
    patient_id = db.Column(db.String(60), default="", index=True)

    age = db.Column(db.Float)
    gender = db.Column(db.String(10))
    bilirubin = db.Column(db.Float)
    cholesterol = db.Column(db.Float)
    albumin = db.Column(db.Float)
    copper = db.Column(db.Float)
    alk_phos = db.Column(db.Float)
    sgot = db.Column(db.Float)
    triglycerides = db.Column(db.Float)
    platelets = db.Column(db.Float)
    prothrombin = db.Column(db.Float)
    ascites = db.Column(db.String(5))
    hepatomegaly = db.Column(db.String(5))
    spiders = db.Column(db.String(5))
    edema = db.Column(db.String(5))
    stage = db.Column(db.Integer)

    result = db.Column(db.String(20), index=True)   # "High Risk" / "Low Risk"
    probability = db.Column(db.Float)                # 0-1
    risk_level = db.Column(db.String(20))
    model_name = db.Column(db.String(60), default="RandomForest")

    # Per-prediction explanation (JSON-encoded list of contribution records)
    explanation_json = db.Column(db.Text, default="")
    # Derived clinical scores stored for reporting
    meld_score = db.Column(db.Float)
    fib4_score = db.Column(db.Float)
    apri_score = db.Column(db.Float)

    doctor_notes = db.Column(db.Text, default="")
    source = db.Column(db.String(20), default="manual")  # manual | ocr | batch | api
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None), nullable=False, index=True)

    @property
    def probability_pct(self) -> float:
        return round((self.probability or 0.0) * 100, 1)

    @property
    def explanation(self) -> list:
        import json

        if not self.explanation_json:
            return []
        try:
            return json.loads(self.explanation_json)
        except (ValueError, TypeError):
            return []

    def clinical_inputs(self) -> dict:
        """The 16 model features in the form-field naming used by the app."""
        return {
            "age": self.age,
            "gender": self.gender,
            "bilirubin": self.bilirubin,
            "cholesterol": self.cholesterol,
            "albumin": self.albumin,
            "copper": self.copper,
            "alk_phos": self.alk_phos,
            "sgot": self.sgot,
            "triglycerides": self.triglycerides,
            "platelets": self.platelets,
            "prothrombin": self.prothrombin,
            "ascites": self.ascites,
            "hepatomegaly": self.hepatomegaly,
            "spiders": self.spiders,
            "edema": self.edema,
            "stage": self.stage,
        }

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "patient_name": self.patient_name,
            "patient_id": self.patient_id,
            "age": self.age,
            "gender": self.gender,
            "result": self.result,
            "probability": self.probability_pct,
            "risk_level": self.risk_level,
            "stage": self.stage,
            "model_name": self.model_name,
            "meld_score": self.meld_score,
            "fib4_score": self.fib4_score,
            "apri_score": self.apri_score,
            "source": self.source,
            "doctor_notes": self.doctor_notes,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M"),
        }

    def __repr__(self) -> str:
        return f"<Prediction #{self.id} {self.patient_name} {self.result}>"


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------
class AuditLog(db.Model):
    __tablename__ = "audit_logs"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
        db.Integer, db.ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    actor_email = db.Column(db.String(120), default="")  # snapshot, survives user deletion
    action = db.Column(db.String(255), nullable=False)
    category = db.Column(db.String(40), default="general")  # auth | prediction | admin | data
    ip_address = db.Column(db.String(64), default="")
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None), nullable=False, index=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "actor_email": self.actor_email,
            "action": self.action,
            "category": self.category,
            "ip_address": self.ip_address,
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        }


# ---------------------------------------------------------------------------
# Contact messages
# ---------------------------------------------------------------------------
class ContactMessage(db.Model):
    __tablename__ = "contact_messages"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    subject = db.Column(db.String(200), default="")
    message = db.Column(db.Text, nullable=False)
    is_read = db.Column(db.Boolean, default=False, nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None), nullable=False)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "subject": self.subject,
            "message": self.message,
            "is_read": self.is_read,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M"),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def log_action(user_id, action, ip_address="", category="general", actor_email=""):
    """Append an audit entry.

    Never raises: an audit failure must not take down the request that
    triggered it.
    """
    try:
        if user_id and not actor_email:
            user = db.session.get(User, user_id)
            actor_email = user.email if user else ""
        entry = AuditLog(
            user_id=user_id,
            actor_email=actor_email,
            action=action,
            category=category,
            ip_address=(ip_address or "")[:64],
        )
        db.session.add(entry)
        db.session.commit()
    except Exception:  # pragma: no cover - defensive
        db.session.rollback()
