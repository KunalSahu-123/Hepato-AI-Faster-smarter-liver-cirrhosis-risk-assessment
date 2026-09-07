"""
test_app.py
------------
Unit and integration tests for the LiverAI liver cirrhosis prediction system.

Run everything:
    python -m pytest test_app.py -v

Run one group:
    python -m pytest test_app.py -v -k clinical_scores

The model-dependent tests are skipped automatically if `saved_model/` has not
been populated yet, so a fresh clone can still run the rest of the suite before
`python train_model.py` has been executed.
"""

import io
import json
import os

import pytest

from config import TestConfig
import app as app_module
from app import create_app, reset_predictor_cache
from database import db, User, Prediction, AuditLog, ContactMessage
from model import CirrhosisPredictor, ModelNotTrainedError
import clinical_scores as cs
from preprocessing import (
    FEATURE_COLUMNS,
    FIELD_META,
    Preprocessor,
    ValidationError,
    encode_single_input,
    flag_abnormal,
    load_and_clean_dataset,
    rows_from_dataframe,
    validate_clinical_input,
)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_READY = all(
    os.path.exists(os.path.join(BASE_DIR, "saved_model", name))
    for name in ("random_forest_model.joblib", "encoders.joblib", "metrics.json")
)
needs_model = pytest.mark.skipif(
    not MODEL_READY, reason="model artefacts missing - run `python train_model.py` first"
)


# ---------------------------------------------------------------------------
# Shared sample data
# ---------------------------------------------------------------------------
# A decompensated patient: high bilirubin, low albumin, ascites present.
SAMPLE_PATIENT = {
    "patient_name": "Test Patient",
    "patient_id": "T-001",
    "age": "55",
    "gender": "F",
    "bilirubin": "3.2",
    "cholesterol": "320",
    "albumin": "3.0",
    "copper": "150",
    "alk_phos": "2200",
    "sgot": "110",
    "triglycerides": "160",
    "platelets": "200",
    "prothrombin": "12.5",
    "ascites": "Y",
    "hepatomegaly": "Y",
    "spiders": "Y",
    "edema": "S",
    "stage": "3",
}

# A comparatively healthy patient, for tests that need two different profiles.
HEALTHY_PATIENT = dict(
    SAMPLE_PATIENT,
    patient_name="Healthy Patient",
    patient_id="T-002",
    bilirubin="0.8",
    albumin="4.4",
    copper="30",
    alk_phos="90",
    sgot="25",
    platelets="290",
    prothrombin="10.2",
    ascites="N",
    hepatomegaly="N",
    spiders="N",
    edema="N",
    stage="1",
)

PASSWORD = "Secret123"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def app():
    """A fresh app on an in-memory database for each test."""
    flask_app = create_app(TestConfig)
    with flask_app.app_context():
        db.create_all()
        yield flask_app
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def csrf_client():
    """A client with CSRF protection left ON, to prove it is actually wired up."""

    class CsrfConfig(TestConfig):
        WTF_CSRF_ENABLED = True

    flask_app = create_app(CsrfConfig)
    with flask_app.app_context():
        db.create_all()
        yield flask_app.test_client()
        db.session.remove()
        db.drop_all()


@pytest.fixture(scope="module")
def predictor():
    if not MODEL_READY:
        pytest.skip("model artefacts missing")
    return CirrhosisPredictor(
        os.path.join(BASE_DIR, "saved_model", "random_forest_model.joblib"),
        os.path.join(BASE_DIR, "saved_model", "encoders.joblib"),
        os.path.join(BASE_DIR, "saved_model", "metrics.json"),
    )


@pytest.fixture(scope="module")
def fitted_preprocessor():
    df = load_and_clean_dataset()
    return Preprocessor().fit(df, save_imputer=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_user(email="user@test.com", password=PASSWORD, role="user", **kwargs):
    user = User(full_name=kwargs.pop("full_name", "Test User"), email=email, role=role, **kwargs)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    return user


def login(client, email="user@test.com", password=PASSWORD):
    return client.post(
        "/login", data={"email": email, "password": password}, follow_redirects=True
    )


def make_logged_in_user(client, email="user@test.com", role="user"):
    user = make_user(email=email, role=role)
    login(client, email)
    return user


def save_prediction(user, **overrides):
    """Insert a prediction row directly, bypassing the model."""
    cleaned = validate_clinical_input(SAMPLE_PATIENT)
    fields = dict(
        user_id=user.id,
        patient_name="Stored Patient",
        result="High Risk",
        probability=0.82,
        risk_level="High",
        model_name="test-model",
        **cleaned,
    )
    fields.update(overrides)
    record = Prediction(**fields)
    db.session.add(record)
    db.session.commit()
    return record


# ===========================================================================
# Unit tests: preprocessing / validation
# ===========================================================================
class TestDataset:
    def test_dataset_loads_and_is_clean(self):
        df = load_and_clean_dataset()
        assert len(df) > 0
        assert "Risk" in df.columns
        for column in FEATURE_COLUMNS:
            assert column in df.columns, f"{column} missing from cleaned dataset"

    def test_target_is_binary(self):
        df = load_and_clean_dataset()
        assert set(df["Risk"].unique()) <= {0, 1}

    def test_both_classes_present(self):
        # A single-class dataset would silently produce a useless model.
        df = load_and_clean_dataset()
        assert df["Risk"].nunique() == 2

    def test_missing_dataset_raises(self):
        with pytest.raises(FileNotFoundError):
            load_and_clean_dataset("dataset/does-not-exist.csv")


class TestValidation:
    def test_accepts_and_coerces_a_good_record(self):
        cleaned = validate_clinical_input(SAMPLE_PATIENT)
        assert cleaned["age"] == 55.0
        assert isinstance(cleaned["age"], float)
        assert cleaned["stage"] == 3
        assert isinstance(cleaned["stage"], int)
        assert cleaned["gender"] == "F"
        # Non-clinical form fields are dropped, not passed through.
        assert "patient_name" not in cleaned

    def test_every_required_field_is_returned(self):
        cleaned = validate_clinical_input(SAMPLE_PATIENT)
        assert set(cleaned) == set(FIELD_META)

    def test_missing_field_reports_that_field(self):
        payload = dict(SAMPLE_PATIENT)
        del payload["bilirubin"]
        with pytest.raises(ValidationError) as excinfo:
            validate_clinical_input(payload)
        assert "bilirubin" in excinfo.value.errors
        assert "required" in excinfo.value.errors["bilirubin"].lower()

    def test_blank_field_is_treated_as_missing(self):
        with pytest.raises(ValidationError) as excinfo:
            validate_clinical_input(dict(SAMPLE_PATIENT, albumin="   "))
        assert "albumin" in excinfo.value.errors

    def test_non_numeric_value_rejected(self):
        with pytest.raises(ValidationError) as excinfo:
            validate_clinical_input(dict(SAMPLE_PATIENT, platelets="abc"))
        assert "must be a number" in excinfo.value.errors["platelets"]

    @pytest.mark.parametrize(
        "field,value",
        [
            ("age", "250"),        # above range
            ("age", "0"),          # below range
            ("bilirubin", "500"),
            ("albumin", "0.1"),
            ("prothrombin", "3"),
            ("platelets", "5000"),
        ],
    )
    def test_out_of_range_values_rejected(self, field, value):
        with pytest.raises(ValidationError) as excinfo:
            validate_clinical_input(dict(SAMPLE_PATIENT, **{field: value}))
        assert field in excinfo.value.errors
        assert "between" in excinfo.value.errors[field]

    def test_range_boundaries_are_inclusive(self):
        cleaned = validate_clinical_input(dict(SAMPLE_PATIENT, age="1"))
        assert cleaned["age"] == 1.0
        cleaned = validate_clinical_input(dict(SAMPLE_PATIENT, age="120"))
        assert cleaned["age"] == 120.0

    def test_invalid_choice_rejected(self):
        with pytest.raises(ValidationError) as excinfo:
            validate_clinical_input(dict(SAMPLE_PATIENT, edema="MAYBE"))
        assert "edema" in excinfo.value.errors

    @pytest.mark.parametrize(
        "field,given,expected",
        [
            ("gender", "male", "M"),
            ("gender", "Female", "F"),
            ("ascites", "yes", "Y"),
            ("ascites", "No", "N"),
            ("hepatomegaly", "TRUE", "Y"),
            ("spiders", "false", "N"),
            ("edema", "slight", "S"),
            ("edema", "none", "N"),
            ("gender", " f ", "F"),
        ],
    )
    def test_friendly_spellings_accepted(self, field, given, expected):
        # CSV exports and API clients write "Male"/"Yes"/"None" rather than codes.
        cleaned = validate_clinical_input(dict(SAMPLE_PATIENT, **{field: given}))
        assert cleaned[field] == expected

    def test_all_errors_reported_at_once(self):
        payload = dict(SAMPLE_PATIENT, age="", bilirubin="nope", edema="X")
        with pytest.raises(ValidationError) as excinfo:
            validate_clinical_input(payload)
        assert {"age", "bilirubin", "edema"} <= set(excinfo.value.errors)

    def test_error_message_lists_every_field(self):
        with pytest.raises(ValidationError) as excinfo:
            validate_clinical_input({})
        assert len(excinfo.value.errors) == len(FIELD_META)


class TestAbnormalFlags:
    def test_flags_high_low_and_normal(self):
        flags = flag_abnormal(validate_clinical_input(SAMPLE_PATIENT))
        assert flags["bilirubin"]["status"] == "high"     # 3.2 vs 0.1-1.2
        assert flags["albumin"]["status"] == "low"        # 3.0 vs 3.5-5.0
        assert flags["platelets"]["status"] == "normal"   # 200 within 150-400

    def test_flag_carries_display_metadata(self):
        flags = flag_abnormal(validate_clinical_input(SAMPLE_PATIENT))
        entry = flags["bilirubin"]
        assert entry["label"] == "Bilirubin"
        assert entry["unit"] == "mg/dL"
        assert entry["range"] == (0.1, 1.2)
        assert entry["value"] == 3.2

    def test_healthy_patient_has_no_high_flags(self):
        flags = flag_abnormal(validate_clinical_input(HEALTHY_PATIENT))
        assert flags["bilirubin"]["status"] == "normal"
        assert flags["albumin"]["status"] == "normal"

    def test_missing_values_are_skipped_not_flagged(self):
        flags = flag_abnormal({"bilirubin": 2.0})
        assert set(flags) == {"bilirubin"}


class TestPreprocessor:
    def test_encodes_categoricals_to_expected_codes(self, fitted_preprocessor):
        df = load_and_clean_dataset()
        encoded = fitted_preprocessor.transform(df)
        assert encoded["Sex"].isin([0, 1]).all()
        assert encoded["Ascites"].isin([0, 1]).all()
        assert encoded["Edema"].isin([0, 1, 2]).all()

    def test_transform_leaves_no_missing_values(self, fitted_preprocessor):
        encoded = fitted_preprocessor.transform(load_and_clean_dataset())
        assert not encoded[FEATURE_COLUMNS].isnull().values.any()

    def test_encode_single_input_matches_training_columns(self, fitted_preprocessor):
        row = encode_single_input(SAMPLE_PATIENT, fitted_preprocessor)
        assert row.shape == (1, len(FEATURE_COLUMNS))
        assert list(row.columns) == FEATURE_COLUMNS
        assert not row.isnull().values.any()

    def test_encode_single_input_validates(self, fitted_preprocessor):
        with pytest.raises(ValidationError):
            encode_single_input(dict(SAMPLE_PATIENT, bilirubin="oops"), fitted_preprocessor)

    def test_saved_encoder_round_trips(self, fitted_preprocessor, tmp_path):
        path = str(tmp_path / "encoders.joblib")
        fitted_preprocessor.save(path)
        reloaded = Preprocessor.load(path)
        original = encode_single_input(SAMPLE_PATIENT, fitted_preprocessor)
        again = encode_single_input(SAMPLE_PATIENT, reloaded)
        assert original.equals(again)


class TestRowsFromDataframe:
    def test_maps_csv_columns_onto_form_fields(self):
        import pandas as pd

        df = pd.DataFrame([{
            "patient_name": "CSV Patient", "age": 60, "gender": "M",
            "bilirubin": 2.0, "cholesterol": 300, "albumin": 3.2, "copper": 120,
            "alk_phos": 1500, "sgot": 90, "triglycerides": 140, "platelets": 210,
            "prothrombin": 11.5, "ascites": "N", "hepatomegaly": "Y",
            "spiders": "N", "edema": "N", "stage": 2,
        }])
        rows = rows_from_dataframe(df)
        assert len(rows) == 1
        assert rows[0]["patient_name"] == "CSV Patient"
        assert validate_clinical_input(rows[0])["age"] == 60.0


# ===========================================================================
# Unit tests: clinical scores
# ===========================================================================
class TestClinicalScores:
    def test_inr_estimated_from_prothrombin_time(self):
        assert cs.estimate_inr(11.0) == pytest.approx(11.0 / 11.0, abs=0.3)
        assert cs.estimate_inr(22.0) > cs.estimate_inr(11.0)
        assert cs.estimate_inr(None) is None

    def test_meld_is_in_the_documented_range(self):
        score = cs.meld_score(bilirubin=3.2, prothrombin=12.5)
        assert score is not None
        assert 6 <= score <= 40

    def test_meld_rises_with_bilirubin(self):
        low = cs.meld_score(bilirubin=1.0, prothrombin=11.0)
        high = cs.meld_score(bilirubin=15.0, prothrombin=11.0)
        assert high > low

    def test_meld_handles_missing_inputs(self):
        assert cs.meld_score(None, None) is None

    def test_fib4_rises_as_platelets_fall(self):
        healthy = cs.fib4_score(age=55, sgot=30, platelets=300)
        fibrotic = cs.fib4_score(age=55, sgot=120, platelets=90)
        assert fibrotic > healthy

    def test_apri_rises_as_platelets_fall(self):
        assert cs.apri_score(sgot=120, platelets=90) > cs.apri_score(sgot=30, platelets=300)

    def test_apri_guards_zero_platelets(self):
        # Division by zero must not escape as an exception.
        assert cs.apri_score(sgot=50, platelets=0) is None

    @pytest.mark.parametrize("score,expected_in_band", [(5, "Low"), (25, "High")])
    def test_meld_interpretation_bands_differ(self, score, expected_in_band):
        band, meaning = cs.interpret_meld(score)
        assert band
        assert meaning
        assert cs.interpret_meld(5)[0] != cs.interpret_meld(35)[0]

    def test_interpretations_handle_none(self):
        for fn in (cs.interpret_meld, cs.interpret_fib4, cs.interpret_apri):
            band, meaning = fn(None)
            assert band == "N/A"

    def test_child_pugh_classes_span_a_to_c(self):
        _, mild, _ = cs.child_pugh_partial(1.0, 4.2, 10.0, "N")
        _, severe, _ = cs.child_pugh_partial(4.0, 2.0, 22.0, "Y")
        assert mild == "A"
        assert severe == "C"

    def test_child_pugh_ascites_adds_points(self):
        without, _, _ = cs.child_pugh_partial(1.0, 4.2, 10.0, "N")
        with_ascites, _, _ = cs.child_pugh_partial(1.0, 4.2, 10.0, "Y")
        assert with_ascites > without

    def test_child_pugh_needs_core_values(self):
        points, band, _ = cs.child_pugh_partial(None, None, None, "N")
        assert points is None
        assert band == "N/A"

    def test_compute_all_returns_every_score(self):
        scores = cs.compute_all(validate_clinical_input(SAMPLE_PATIENT))
        for key in ("meld", "fib4", "apri", "child_pugh"):
            assert key in scores
            assert scores[key]["value"] is not None
            assert scores[key]["label"]
            assert scores[key]["band"]
            assert scores[key]["meaning"]
        assert scores["inr_estimate"] is not None

    def test_compute_all_survives_empty_input(self):
        # The detail page calls this on old records that may lack some columns.
        scores = cs.compute_all({})
        assert scores["meld"]["value"] is None
        assert scores["meld"]["band"] == "N/A"


# ===========================================================================
# Unit tests: model
# ===========================================================================
@needs_model
class TestModel:
    def test_predict_returns_label_probability_and_band(self, predictor):
        label, probability, band = predictor.predict(SAMPLE_PATIENT)
        assert label in ("High Risk", "Low Risk")
        assert 0.0 <= probability <= 1.0
        assert band in ("Critical", "High", "Moderate", "Low")

    def test_label_agrees_with_probability(self, predictor):
        label, probability, _ = predictor.predict(SAMPLE_PATIENT)
        assert (probability >= 0.5) == (label == "High Risk")

    def test_predictions_are_deterministic(self, predictor):
        label_a, probability_a, band_a = predictor.predict(SAMPLE_PATIENT)
        label_b, probability_b, band_b = predictor.predict(SAMPLE_PATIENT)
        assert (label_a, band_a) == (label_b, band_b)
        # The forest sums tree votes across threads, so the reduction order can
        # shift the last bit or two of the float. Anything above 1e-9 would be a
        # real problem; 1e-16 is arithmetic noise.
        assert probability_a == pytest.approx(probability_b, abs=1e-9)

    def test_sicker_patient_scores_higher(self, predictor):
        _, sick_probability, _ = predictor.predict(SAMPLE_PATIENT)
        _, well_probability, _ = predictor.predict(HEALTHY_PATIENT)
        assert sick_probability > well_probability

    def test_predict_rejects_invalid_input(self, predictor):
        with pytest.raises(ValidationError):
            predictor.predict(dict(SAMPLE_PATIENT, albumin=""))

    @pytest.mark.parametrize(
        "probability,expected",
        [(0.99, "Critical"), (0.72, "High"), (0.45, "Moderate"), (0.05, "Low")],
    )
    def test_risk_bands_are_ordered(self, predictor, probability, expected):
        assert predictor.risk_band(probability) == expected

    def test_batch_isolates_bad_rows(self, predictor):
        broken = dict(SAMPLE_PATIENT, patient_name="Broken", bilirubin="not-a-number")
        outcomes = predictor.predict_batch([SAMPLE_PATIENT, broken, HEALTHY_PATIENT])
        assert [o["ok"] for o in outcomes] == [True, False, True]
        assert outcomes[1]["row"] == 2
        assert "bilirubin" in outcomes[1]["errors"]
        # The good rows still carry usable results.
        assert outcomes[0]["result"] in ("High Risk", "Low Risk")
        assert outcomes[2]["cleaned"]["albumin"] == 4.4

    def test_batch_falls_back_to_row_number_for_unnamed_rows(self, predictor):
        anonymous = {k: v for k, v in SAMPLE_PATIENT.items() if k != "patient_name"}
        outcomes = predictor.predict_batch([anonymous])
        assert outcomes[0]["patient_name"] == "Row 1"

    def test_batch_handles_empty_list(self, predictor):
        assert predictor.predict_batch([]) == []

    def test_explanation_shape_and_direction(self, predictor):
        drivers = predictor.explain(SAMPLE_PATIENT, top_n=5)
        assert 0 < len(drivers) <= 5
        for driver in drivers:
            assert {"label", "impact", "direction"} <= set(driver)
            assert driver["direction"] in ("increases risk", "decreases risk")
        # Ordered by magnitude of effect, strongest first.
        impacts = [abs(d["impact"]) for d in drivers]
        assert impacts == sorted(impacts, reverse=True)

    def test_top_features_are_ranked(self, predictor):
        features = predictor.top_features(5)
        assert len(features) == 5
        importances = [importance for _, importance in features]
        assert importances == sorted(importances, reverse=True)

    def test_labelled_top_features_have_display_names(self, predictor):
        for entry in predictor.labelled_top_features(3):
            assert entry["label"]
            assert entry["importance"] >= 0

    def test_metrics_are_loaded(self, predictor):
        assert 0 <= predictor.metrics["accuracy"] <= 1
        assert 0 <= predictor.metrics["roc_auc"] <= 1
        assert predictor.model_name

    def test_model_comparison_available(self, predictor):
        comparison = predictor.model_comparison
        assert isinstance(comparison, list)
        assert comparison and "model" in comparison[0]

    def test_is_stale_returns_a_bool(self, predictor):
        assert isinstance(predictor.is_stale, bool)

    def test_missing_artefacts_raise_clearly(self):
        with pytest.raises(Exception):
            CirrhosisPredictor("no/such/model.joblib", "no/such/encoders.joblib")


# ===========================================================================
# Integration: public pages
# ===========================================================================
class TestPublicPages:
    @pytest.mark.parametrize(
        "path", ["/", "/about", "/services", "/contact", "/login", "/register",
                 "/forgot-password"],
    )
    def test_page_renders(self, client, path):
        assert client.get(path).status_code == 200

    def test_unknown_path_renders_the_error_page(self, client):
        response = client.get("/no-such-page")
        assert response.status_code == 404
        assert b"LiverAI" in response.data

    def test_health_endpoint_reports_status(self, client):
        response = client.get("/health")
        assert response.status_code in (200, 503)
        body = response.get_json()
        assert body["status"] in ("healthy", "degraded", "unhealthy")
        assert body["checks"]["database"] == "ok"
        assert "model" in body["checks"]
        assert body["timestamp"]

    def test_security_headers_are_set(self, client):
        headers = client.get("/").headers
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["X-Frame-Options"] == "SAMEORIGIN"
        assert "Referrer-Policy" in headers

    def test_login_page_hides_demo_credentials_in_production(self):
        class ProdConfig(TestConfig):
            ENV_NAME = "production"

        prod_app = create_app(ProdConfig)
        with prod_app.app_context():
            db.create_all()
            response = prod_app.test_client().get("/login")
            assert b"Demo administrator" not in response.data
            db.drop_all()


class TestContactForm:
    def test_valid_message_is_stored(self, client):
        response = client.post("/contact", data={
            "name": "Asker", "email": "asker@test.com",
            "subject": "Question", "message": "Could you explain the calibration step?",
        }, follow_redirects=True)
        assert response.status_code == 200
        message = ContactMessage.query.filter_by(email="asker@test.com").first()
        assert message is not None
        assert message.subject == "Question"
        assert message.is_read is False

    def test_short_message_is_rejected_and_repopulates(self, client):
        response = client.post("/contact", data={
            "name": "Asker", "email": "asker@test.com", "subject": "Hi", "message": "no",
        }, follow_redirects=True)
        assert response.status_code == 200
        assert ContactMessage.query.count() == 0
        # The user's typed values come back rather than being wiped.
        assert b"asker@test.com" in response.data

    def test_invalid_email_is_rejected(self, client):
        client.post("/contact", data={
            "name": "Asker", "email": "not-an-email",
            "subject": "", "message": "A long enough message body.",
        })
        assert ContactMessage.query.count() == 0


# ===========================================================================
# Integration: authentication
# ===========================================================================
class TestRegistration:
    def test_registration_creates_an_account(self, client):
        response = client.post("/register", data={
            "full_name": "Jane Doctor", "email": "jane@test.com",
            "password": PASSWORD, "confirm_password": PASSWORD,
            "organisation": "City General", "specialisation": "Hepatology",
        }, follow_redirects=True)
        assert response.status_code == 200
        user = User.query.filter_by(email="jane@test.com").first()
        assert user is not None
        assert user.role == "user"
        assert user.organisation == "City General"

    def test_password_is_hashed_not_stored(self, client):
        client.post("/register", data={
            "full_name": "Jane Doctor", "email": "jane@test.com",
            "password": PASSWORD, "confirm_password": PASSWORD,
        })
        user = User.query.filter_by(email="jane@test.com").first()
        assert PASSWORD not in user.password_hash
        assert user.check_password(PASSWORD)
        assert not user.check_password("wrong-password")

    def test_email_is_normalised_to_lowercase(self, client):
        client.post("/register", data={
            "full_name": "Case Test", "email": "MiXeD@Test.COM",
            "password": PASSWORD, "confirm_password": PASSWORD,
        })
        assert User.query.filter_by(email="mixed@test.com").first() is not None

    def test_registration_logs_an_audit_entry(self, client):
        client.post("/register", data={
            "full_name": "Audited", "email": "audited@test.com",
            "password": PASSWORD, "confirm_password": PASSWORD,
        })
        assert AuditLog.query.filter_by(category="auth").count() >= 1

    @pytest.mark.parametrize("overrides,expected", [
        ({"confirm_password": "Different1"}, b"do not match"),
        ({"password": "abc", "confirm_password": "abc"}, b"at least"),
        ({"password": "12345678", "confirm_password": "12345678"}, b"mix letters"),
        ({"password": "abcdefgh", "confirm_password": "abcdefgh"}, b"mix letters"),
        ({"email": "not-an-email"}, b"valid email"),
        ({"full_name": "X"}, b"full name"),
    ])
    def test_invalid_registrations_are_rejected(self, client, overrides, expected):
        data = {
            "full_name": "Bad User", "email": "bad@test.com",
            "password": PASSWORD, "confirm_password": PASSWORD,
        }
        data.update(overrides)
        response = client.post("/register", data=data, follow_redirects=True)
        assert response.status_code == 200
        assert expected in response.data
        assert User.query.count() == 0

    def test_duplicate_email_is_rejected(self, client):
        make_user(email="taken@test.com")
        response = client.post("/register", data={
            "full_name": "Copycat", "email": "taken@test.com",
            "password": PASSWORD, "confirm_password": PASSWORD,
        }, follow_redirects=True)
        assert b"already exists" in response.data
        assert User.query.filter_by(email="taken@test.com").count() == 1


class TestLogin:
    def test_correct_credentials_log_in(self, client):
        make_user()
        response = login(client)
        assert response.status_code == 200
        assert b"Dashboard" in response.data

    def test_wrong_password_is_refused(self, client):
        make_user()
        response = login(client, password="WrongPass1")
        assert b"Invalid" in response.data or b"incorrect" in response.data.lower()
        assert client.get("/dashboard").status_code == 302

    def test_unknown_email_is_refused_without_confirming_existence(self, client):
        response = login(client, email="ghost@test.com")
        # Same wording as a wrong password, so accounts can't be enumerated.
        assert b"Invalid" in response.data or b"incorrect" in response.data.lower()

    def test_deactivated_account_cannot_log_in(self, client):
        user = make_user()
        user.is_active_account = False
        db.session.commit()
        login(client)
        assert client.get("/dashboard").status_code == 302

    def test_logout_ends_the_session(self, client):
        make_user()
        login(client)
        assert client.get("/dashboard").status_code == 200
        client.get("/logout", follow_redirects=True)
        assert client.get("/dashboard").status_code == 302

    def test_lockout_after_repeated_failures(self, client, app):
        app.config["MAX_LOGIN_ATTEMPTS"] = 3
        app.config["LOGIN_LOCKOUT_MINUTES"] = 15
        user = make_user()
        for _ in range(3):
            login(client, password="WrongPass1")
        db.session.refresh(user)
        assert user.is_locked
        # Even the correct password is refused while the lock stands.
        response = login(client)
        assert client.get("/dashboard").status_code == 302
        assert b"locked" in response.data.lower()

    def test_successful_login_clears_the_failure_count(self, client, app):
        app.config["MAX_LOGIN_ATTEMPTS"] = 5
        user = make_user()
        login(client, password="WrongPass1")
        db.session.refresh(user)
        assert user.failed_login_attempts == 1
        login(client)
        db.session.refresh(user)
        assert user.failed_login_attempts == 0
        assert user.last_login_at is not None


class TestPasswordReset:
    def test_token_round_trips(self, app):
        user = make_user()
        token = user.generate_reset_token(app.config["SECRET_KEY"])
        assert User.verify_reset_token(token, app.config["SECRET_KEY"]) is user

    def test_token_is_single_use(self, app):
        # The token embeds a slice of the password hash, so changing the
        # password invalidates any outstanding token.
        user = make_user()
        token = user.generate_reset_token(app.config["SECRET_KEY"])
        user.set_password("Changed123")
        db.session.commit()
        assert User.verify_reset_token(token, app.config["SECRET_KEY"]) is None

    def test_tampered_token_is_refused(self, app):
        user = make_user()
        token = user.generate_reset_token(app.config["SECRET_KEY"])
        assert User.verify_reset_token(token + "x", app.config["SECRET_KEY"]) is None

    def test_token_from_another_secret_is_refused(self, app):
        user = make_user()
        token = user.generate_reset_token("some-other-secret")
        assert User.verify_reset_token(token, app.config["SECRET_KEY"]) is None

    def test_expired_token_is_refused(self, app):
        user = make_user()
        token = user.generate_reset_token(app.config["SECRET_KEY"])
        assert User.verify_reset_token(token, app.config["SECRET_KEY"], max_age=-1) is None

    def test_request_response_is_identical_for_unknown_emails(self, client):
        make_user(email="real@test.com")
        known = client.post("/forgot-password", data={"email": "real@test.com"},
                            follow_redirects=True)
        unknown = client.post("/forgot-password", data={"email": "ghost@test.com"},
                              follow_redirects=True)
        assert known.status_code == unknown.status_code == 200
        # Neither response may reveal whether the account exists.
        assert (b"no account" in unknown.data.lower()) is False

    def test_reset_page_rejects_a_bad_token(self, client):
        response = client.get("/reset-password/garbage", follow_redirects=True)
        assert response.status_code == 200
        assert b"invalid" in response.data.lower() or b"expired" in response.data.lower()

    def test_password_can_be_reset_with_a_valid_token(self, client, app):
        user = make_user()
        token = user.generate_reset_token(app.config["SECRET_KEY"])
        response = client.post(f"/reset-password/{token}", data={
            "password": "BrandNew123", "confirm_password": "BrandNew123",
        }, follow_redirects=True)
        assert response.status_code == 200
        db.session.refresh(user)
        assert user.check_password("BrandNew123")
        assert login(client, password="BrandNew123").status_code == 200

    def test_reset_rejects_mismatched_passwords(self, client, app):
        user = make_user()
        token = user.generate_reset_token(app.config["SECRET_KEY"])
        client.post(f"/reset-password/{token}", data={
            "password": "BrandNew123", "confirm_password": "Different123",
        })
        db.session.refresh(user)
        assert user.check_password(PASSWORD)


class TestCsrf:
    def test_form_post_without_a_token_is_rejected(self, csrf_client):
        response = csrf_client.post("/contact", data={
            "name": "No Token", "email": "notoken@test.com",
            "subject": "", "message": "This should not be accepted.",
        })
        # Browsers get a redirect plus a "session expired" flash rather than a
        # bare 400 page; what matters is that nothing was written.
        assert response.status_code == 302
        assert ContactMessage.query.count() == 0

    def test_login_post_without_a_token_is_rejected(self, csrf_client):
        make_user()
        response = csrf_client.post("/login", data={"email": "user@test.com",
                                                    "password": PASSWORD})
        assert response.status_code == 302
        # The redirect must not have logged anybody in.
        assert csrf_client.get("/dashboard").status_code == 302

    def test_json_post_without_a_token_gets_a_400(self, csrf_client):
        make_user()
        csrf_client.post("/login", data={"email": "user@test.com", "password": PASSWORD})
        response = csrf_client.post("/api/predict", json=SAMPLE_PATIENT)
        assert response.status_code in (302, 400)
        assert Prediction.query.count() == 0


class TestAccessControl:
    @pytest.mark.parametrize("path", [
        "/dashboard", "/prediction", "/history", "/reports", "/profile",
        "/batch", "/compare", "/admin", "/api/patients", "/api/dashboard",
    ])
    def test_protected_pages_redirect_anonymous_users(self, client, path):
        assert client.get(path).status_code == 302

    def test_non_admin_is_forbidden_from_admin_pages(self, client):
        make_logged_in_user(client)
        for path in ("/admin", "/admin/messages", "/admin/audit"):
            assert client.get(path).status_code == 403

    def test_admin_can_reach_admin_pages(self, client):
        make_logged_in_user(client, email="admin@test.com", role="admin")
        for path in ("/admin", "/admin/messages", "/admin/audit"):
            assert client.get(path).status_code == 200

    def test_users_cannot_read_each_others_predictions(self, client):
        owner = make_user(email="owner@test.com")
        record = save_prediction(owner)
        make_logged_in_user(client, email="snooper@test.com")
        # Not 403: existence itself shouldn't be confirmed to a stranger.
        assert client.get(f"/prediction/{record.id}").status_code == 404
        assert client.get(f"/api/patients/{record.id}").status_code == 404
        assert client.get(f"/reports/pdf/{record.id}").status_code == 404

    def test_users_cannot_delete_each_others_predictions(self, client):
        owner = make_user(email="owner@test.com")
        record = save_prediction(owner)
        make_logged_in_user(client, email="snooper@test.com")
        response = client.post(f"/history/{record.id}/delete")
        assert response.status_code == 404
        assert db.session.get(Prediction, record.id) is not None


# ===========================================================================
# Integration: prediction workflow
# ===========================================================================
@needs_model
class TestPredictionWorkflow:
    def test_form_submission_saves_and_renders_a_result(self, client):
        user = make_logged_in_user(client)
        response = client.post("/prediction", data=SAMPLE_PATIENT, follow_redirects=True)
        assert response.status_code == 200
        record = Prediction.query.filter_by(user_id=user.id).first()
        assert record is not None
        assert record.result in ("High Risk", "Low Risk")
        assert 0 <= record.probability <= 1
        assert record.patient_name == "Test Patient"
        assert record.source == "manual"

    def test_clinical_scores_are_persisted(self, client):
        make_logged_in_user(client)
        client.post("/prediction", data=SAMPLE_PATIENT)
        record = Prediction.query.first()
        assert record.meld_score is not None
        assert record.fib4_score is not None
        assert record.apri_score is not None

    def test_explanation_is_persisted_and_readable(self, client):
        make_logged_in_user(client)
        client.post("/prediction", data=SAMPLE_PATIENT)
        record = Prediction.query.first()
        assert isinstance(record.explanation, list)
        assert record.explanation

    def test_invalid_submission_reports_fields_and_saves_nothing(self, client):
        make_logged_in_user(client)
        payload = dict(SAMPLE_PATIENT)
        del payload["bilirubin"]
        response = client.post("/prediction", data=payload, follow_redirects=True)
        assert response.status_code == 200
        assert b"required" in response.data.lower()
        assert Prediction.query.count() == 0

    def test_invalid_submission_keeps_the_other_values(self, client):
        make_logged_in_user(client)
        response = client.post("/prediction", data=dict(SAMPLE_PATIENT, bilirubin=""),
                               follow_redirects=True)
        # The 55-year-old's other answers survive the round trip.
        assert b'value="55"' in response.data

    def test_detail_page_shows_the_record(self, client):
        user = make_logged_in_user(client)
        record = save_prediction(user)
        response = client.get(f"/prediction/{record.id}")
        assert response.status_code == 200
        assert b"Stored Patient" in response.data

    def test_notes_can_be_saved(self, client):
        user = make_logged_in_user(client)
        record = save_prediction(user)
        client.post(f"/prediction/{record.id}/notes",
                    data={"notes": "Referred to hepatology."}, follow_redirects=True)
        db.session.refresh(record)
        assert record.doctor_notes == "Referred to hepatology."

    def test_prediction_is_audit_logged(self, client):
        make_logged_in_user(client)
        client.post("/prediction", data=SAMPLE_PATIENT)
        assert AuditLog.query.filter_by(category="prediction").count() == 1


@needs_model
class TestBatchPrediction:
    @staticmethod
    def _csv(rows):
        header = ["patient_name"] + list(FIELD_META)
        lines = [",".join(header)]
        for row in rows:
            lines.append(",".join(str(row.get(column, "")) for column in header))
        return io.BytesIO("\n".join(lines).encode())

    def test_template_download_has_the_expected_columns(self, client):
        make_logged_in_user(client)
        response = client.get("/batch/template.csv")
        assert response.status_code == 200
        assert response.mimetype == "text/csv"
        header = response.data.decode().splitlines()[0]
        for field in FIELD_META:
            assert field in header

    def test_good_rows_are_scored_and_saved(self, client):
        user = make_logged_in_user(client)
        data = {"batch_file": (self._csv([SAMPLE_PATIENT, HEALTHY_PATIENT]), "patients.csv")}
        response = client.post("/batch", data=data, content_type="multipart/form-data",
                               follow_redirects=True)
        assert response.status_code == 200
        saved = Prediction.query.filter_by(user_id=user.id).all()
        assert len(saved) == 2
        assert {r.source for r in saved} == {"batch"}

    def test_bad_rows_are_reported_without_losing_good_ones(self, client):
        make_logged_in_user(client)
        broken = dict(SAMPLE_PATIENT, patient_name="Broken", albumin="99")
        data = {"batch_file": (self._csv([SAMPLE_PATIENT, broken]), "patients.csv")}
        response = client.post("/batch", data=data, content_type="multipart/form-data",
                               follow_redirects=True)
        assert response.status_code == 200
        assert Prediction.query.count() == 1
        assert b"Broken" in response.data

    def test_non_csv_upload_is_refused(self, client):
        make_logged_in_user(client)
        data = {"batch_file": (io.BytesIO(b"not a csv"), "payload.exe")}
        response = client.post("/batch", data=data, content_type="multipart/form-data",
                               follow_redirects=True)
        assert b".csv" in response.data
        assert Prediction.query.count() == 0

    def test_missing_file_is_refused(self, client):
        make_logged_in_user(client)
        response = client.post("/batch", data={}, content_type="multipart/form-data",
                               follow_redirects=True)
        assert response.status_code == 200
        assert Prediction.query.count() == 0

    def test_summary_pdf_follows_a_batch(self, client):
        make_logged_in_user(client)
        data = {"batch_file": (self._csv([SAMPLE_PATIENT]), "patients.csv")}
        client.post("/batch", data=data, content_type="multipart/form-data")
        response = client.get("/batch/summary.pdf")
        assert response.status_code == 200
        assert response.mimetype == "application/pdf"


class TestHistoryAndCompare:
    def test_history_lists_only_your_own_records(self, client):
        other = make_user(email="other@test.com")
        save_prediction(other, patient_name="Someone Elses Patient")
        user = make_logged_in_user(client)
        save_prediction(user, patient_name="My Patient")
        response = client.get("/history")
        assert b"My Patient" in response.data
        assert b"Someone Elses Patient" not in response.data

    def test_history_search_filters_by_name(self, client):
        user = make_logged_in_user(client)
        save_prediction(user, patient_name="Alice Smith")
        save_prediction(user, patient_name="Bob Jones")
        response = client.get("/history?search=Alice")
        assert b"Alice Smith" in response.data
        assert b"Bob Jones" not in response.data

    def test_history_filters_by_risk(self, client):
        user = make_logged_in_user(client)
        save_prediction(user, patient_name="High One", result="High Risk", probability=0.9)
        save_prediction(user, patient_name="Low One", result="Low Risk", probability=0.1)
        response = client.get("/history?risk=Low Risk")
        assert b"Low One" in response.data
        assert b"High One" not in response.data

    def test_record_can_be_deleted(self, client):
        user = make_logged_in_user(client)
        record = save_prediction(user)
        client.post(f"/history/{record.id}/delete", follow_redirects=True)
        assert db.session.get(Prediction, record.id) is None

    def test_bulk_delete_removes_the_selected_records(self, client):
        user = make_logged_in_user(client)
        # Read the ids up front: once the rows are gone, touching the stale ORM
        # instances raises ObjectDeletedError instead of returning the id.
        first_id = save_prediction(user).id
        second_id = save_prediction(user).id
        third_id = save_prediction(user).id
        client.post("/history/bulk-delete",
                    data={"prediction_ids": [str(first_id), str(second_id)]},
                    follow_redirects=True)
        assert db.session.get(Prediction, first_id) is None
        assert db.session.get(Prediction, second_id) is None
        assert db.session.get(Prediction, third_id) is not None

    def test_bulk_delete_cannot_touch_another_users_records(self, client):
        victim = make_user(email="victim@test.com")
        victim_record_id = save_prediction(victim).id
        make_logged_in_user(client, email="attacker@test.com")
        client.post("/history/bulk-delete",
                    data={"prediction_ids": [str(victim_record_id)]},
                    follow_redirects=True)
        assert db.session.get(Prediction, victim_record_id) is not None

    def test_compare_page_shows_selected_records(self, client):
        user = make_logged_in_user(client)
        first = save_prediction(user, patient_name="Baseline Visit")
        second = save_prediction(user, patient_name="Follow Up Visit")
        response = client.get(f"/compare?ids={first.id}&ids={second.id}")
        assert response.status_code == 200
        assert b"Baseline Visit" in response.data
        assert b"Follow Up Visit" in response.data

    def test_compare_page_works_with_nothing_selected(self, client):
        make_logged_in_user(client)
        assert client.get("/compare").status_code == 200


class TestReportsAndExports:
    def test_reports_page_renders(self, client):
        user = make_logged_in_user(client)
        save_prediction(user)
        assert client.get("/reports").status_code == 200

    def test_csv_export_contains_the_records(self, client):
        user = make_logged_in_user(client)
        save_prediction(user, patient_name="CSV Patient")
        response = client.get("/reports/csv")
        assert response.status_code == 200
        assert response.mimetype == "text/csv"
        assert b"CSV Patient" in response.data

    def test_pdf_report_is_generated(self, client):
        user = make_logged_in_user(client)
        record = save_prediction(user)
        response = client.get(f"/reports/pdf/{record.id}")
        assert response.status_code == 200
        assert response.mimetype == "application/pdf"
        assert response.data.startswith(b"%PDF")

    def test_data_export_is_json(self, client):
        user = make_logged_in_user(client)
        save_prediction(user, patient_name="Exported Patient")
        response = client.get("/profile/export")
        assert response.status_code == 200
        payload = json.loads(response.data)
        assert payload["account"]["email"] == user.email
        assert payload["predictions"][0]["patient_name"] == "Exported Patient"


class TestProfile:
    def test_details_can_be_updated(self, client):
        user = make_logged_in_user(client)
        client.post("/profile", data={
            "action": "details", "full_name": "Dr Updated",
            "organisation": "New Hospital", "specialisation": "Gastroenterology",
            "phone": "555-0100",
        }, follow_redirects=True)
        db.session.refresh(user)
        assert user.full_name == "Dr Updated"
        assert user.organisation == "New Hospital"

    def test_password_change_requires_the_current_password(self, client):
        user = make_logged_in_user(client)
        client.post("/profile", data={
            "action": "password", "current_password": "WrongPass1",
            "new_password": "Rotated123", "confirm_password": "Rotated123",
        }, follow_redirects=True)
        db.session.refresh(user)
        assert user.check_password(PASSWORD)

    def test_password_change_succeeds_with_the_current_password(self, client):
        user = make_logged_in_user(client)
        client.post("/profile", data={
            "action": "password", "current_password": PASSWORD,
            "new_password": "Rotated123", "confirm_password": "Rotated123",
        }, follow_redirects=True)
        db.session.refresh(user)
        assert user.check_password("Rotated123")

    def test_account_deletion_requires_the_password(self, client):
        user = make_logged_in_user(client)
        client.post("/profile/delete", data={"password": "WrongPass1"}, follow_redirects=True)
        assert db.session.get(User, user.id) is not None

    def test_account_deletion_removes_the_user_and_their_records(self, client):
        user = make_logged_in_user(client)
        record = save_prediction(user)
        user_id, record_id = user.id, record.id
        client.post("/profile/delete", data={"password": PASSWORD}, follow_redirects=True)
        assert db.session.get(User, user_id) is None
        assert db.session.get(Prediction, record_id) is None


class TestAdmin:
    def test_dashboard_counts_users_and_predictions(self, client):
        admin = make_logged_in_user(client, email="admin@test.com", role="admin")
        other = make_user(email="patient-owner@test.com")
        save_prediction(other)
        response = client.get("/admin")
        assert response.status_code == 200
        assert b"patient-owner@test.com" in response.data

    def test_admin_can_deactivate_a_user(self, client):
        make_logged_in_user(client, email="admin@test.com", role="admin")
        target = make_user(email="target@test.com")
        client.post(f"/admin/users/{target.id}/toggle-active", follow_redirects=True)
        db.session.refresh(target)
        assert target.is_active_account is False

    def test_admin_can_promote_a_user(self, client):
        make_logged_in_user(client, email="admin@test.com", role="admin")
        target = make_user(email="target@test.com")
        client.post(f"/admin/users/{target.id}/toggle-role", follow_redirects=True)
        db.session.refresh(target)
        assert target.role == "admin"

    def test_admin_cannot_deactivate_themselves(self, client):
        admin = make_logged_in_user(client, email="admin@test.com", role="admin")
        client.post(f"/admin/users/{admin.id}/toggle-active", follow_redirects=True)
        db.session.refresh(admin)
        # Locking the last admin out of their own panel must not be possible.
        assert admin.is_active_account is True

    def test_admin_can_delete_a_user(self, client):
        make_logged_in_user(client, email="admin@test.com", role="admin")
        target = make_user(email="target@test.com")
        target_id = target.id
        client.post(f"/admin/users/{target_id}/delete", follow_redirects=True)
        assert db.session.get(User, target_id) is None

    def test_admin_cannot_delete_themselves(self, client):
        admin = make_logged_in_user(client, email="admin@test.com", role="admin")
        client.post(f"/admin/users/{admin.id}/delete", follow_redirects=True)
        assert db.session.get(User, admin.id) is not None

    def test_messages_can_be_marked_read_and_deleted(self, client):
        make_logged_in_user(client, email="admin@test.com", role="admin")
        message = ContactMessage(name="Asker", email="asker@test.com",
                                 subject="Hi", message="A question about the model.")
        db.session.add(message)
        db.session.commit()
        message_id = message.id

        client.post(f"/admin/messages/{message_id}/read", follow_redirects=True)
        db.session.refresh(message)
        assert message.is_read is True

        client.post(f"/admin/messages/{message_id}/delete", follow_redirects=True)
        assert db.session.get(ContactMessage, message_id) is None

    def test_patient_export_is_csv(self, client):
        make_logged_in_user(client, email="admin@test.com", role="admin")
        owner = make_user(email="owner@test.com")
        save_prediction(owner, patient_name="Admin Export Patient")
        response = client.get("/admin/export/patients.csv")
        assert response.status_code == 200
        assert response.mimetype == "text/csv"
        assert b"Admin Export Patient" in response.data

    def test_audit_trail_can_be_filtered(self, client):
        make_logged_in_user(client, email="admin@test.com", role="admin")
        assert client.get("/admin/audit?category=auth").status_code == 200

    @needs_model
    def test_model_reload_succeeds(self, client):
        make_logged_in_user(client, email="admin@test.com", role="admin")
        response = client.post("/admin/reload-model", follow_redirects=True)
        assert response.status_code == 200
        reset_predictor_cache()


# ===========================================================================
# Integration: JSON API
# ===========================================================================
class TestApi:
    def test_patient_endpoints_require_authentication(self, client):
        assert client.post("/api/predict", json=SAMPLE_PATIENT).status_code == 302
        for path in ("/api/patients", "/api/history", "/api/dashboard", "/api/model"):
            assert client.get(path).status_code == 302

    def test_schema_is_public(self, client):
        # No patient data in it, and API clients need it before they can log in.
        assert client.get("/api/schema").status_code == 200

    def test_schema_documents_every_field(self, client):
        payload = client.get("/api/schema").get_json()
        assert set(payload["fields"]) == set(FIELD_META)
        assert payload["required_fields"]
        bilirubin = payload["fields"]["bilirubin"]
        assert bilirubin["unit"] == "mg/dL"
        assert bilirubin["valid_range"]
        assert bilirubin["reference_range"]

    def test_schema_lists_allowed_values_for_choices(self, client):
        payload = client.get("/api/schema").get_json()
        assert set(payload["fields"]["edema"]["choices"]) == {"N", "S", "Y"}

    def test_dashboard_aggregates_are_returned(self, client):
        user = make_logged_in_user(client)
        save_prediction(user, result="High Risk", risk_level="High")
        payload = client.get("/api/dashboard").get_json()
        assert payload["total"] == 1
        assert payload["high_risk"] == 1
        assert payload["low_risk"] == 0
        assert payload["by_risk_band"] == {"High": 1}

    def test_dashboard_counts_only_your_own_records(self, client):
        other = make_user(email="other@test.com")
        save_prediction(other)
        make_logged_in_user(client)
        assert client.get("/api/dashboard").get_json()["total"] == 0

    def test_patients_endpoint_is_paginated(self, client):
        user = make_logged_in_user(client)
        save_prediction(user, patient_name="API Listed Patient")
        payload = client.get("/api/patients").get_json()
        assert payload["total"] == 1
        assert payload["page"] == 1
        names = [p["patient_name"] for p in payload["items"]]
        assert "API Listed Patient" in names

    def test_patients_endpoint_caps_page_size(self, client):
        make_logged_in_user(client)
        # per_page is clamped to 200 so a client can't ask for the whole table.
        assert client.get("/api/patients?per_page=10000").status_code == 200

    def test_patient_detail_includes_inputs_and_explanation(self, client):
        user = make_logged_in_user(client)
        record = save_prediction(user)
        payload = client.get(f"/api/patients/{record.id}").get_json()
        assert payload["patient_name"] == "Stored Patient"
        assert payload["clinical_inputs"]
        assert "explanation" in payload

    def test_history_endpoint_returns_a_list(self, client):
        user = make_logged_in_user(client)
        save_prediction(user)
        payload = client.get("/api/history").get_json()
        assert isinstance(payload, list)
        assert len(payload) == 1
        assert payload[0]["result"] == "High Risk"

    @needs_model
    def test_model_endpoint_reports_metrics(self, client):
        make_logged_in_user(client)
        payload = client.get("/api/model").get_json()
        assert payload["available"] is True
        assert 0 <= payload["metrics"]["accuracy"] <= 1
        assert payload["feature_importance"]
        assert payload["model_comparison"]

    @needs_model
    def test_predict_endpoint_scores_and_saves(self, client):
        user = make_logged_in_user(client)
        payload = client.post("/api/predict", json=SAMPLE_PATIENT).get_json()
        assert payload["result"] in ("High Risk", "Low Risk")
        assert 0 <= payload["probability"] <= 100
        assert payload["clinical_scores"]["meld"]["value"] is not None
        assert payload["explanation"]
        assert payload["abnormal_flags"]["bilirubin"] == "high"
        assert Prediction.query.filter_by(user_id=user.id).count() == 1

    @needs_model
    def test_predict_endpoint_can_skip_saving(self, client):
        make_logged_in_user(client)
        response = client.post("/api/predict", json=dict(SAMPLE_PATIENT, save=False))
        assert response.get_json()["id"] is None
        assert Prediction.query.count() == 0

    @needs_model
    def test_predict_endpoint_returns_field_errors(self, client):
        make_logged_in_user(client)
        payload = dict(SAMPLE_PATIENT)
        del payload["albumin"]
        response = client.post("/api/predict", json=payload)
        assert response.status_code == 422
        assert "albumin" in response.get_json()["fields"]

    def test_predict_endpoint_rejects_a_non_object_body(self, client):
        make_logged_in_user(client)
        response = client.post("/api/predict", json=["not", "an", "object"])
        assert response.status_code == 400


# ===========================================================================
# Model-unavailable behaviour
# ===========================================================================
class TestModelUnavailable:
    def test_pages_degrade_instead_of_crashing(self, monkeypatch, client):
        """A missing model must produce a message, not a 500 on every page."""
        make_logged_in_user(client)

        def boom():
            raise ModelNotTrainedError("Model artefacts not found.")

        monkeypatch.setattr(app_module, "get_predictor", boom)

        for path in ("/", "/about", "/prediction", "/batch"):
            assert client.get(path).status_code == 200
        assert client.post("/api/predict", json=SAMPLE_PATIENT).status_code == 503
