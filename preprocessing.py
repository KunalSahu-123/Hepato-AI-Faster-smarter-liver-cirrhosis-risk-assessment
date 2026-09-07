"""
preprocessing.py
-----------------
Shared preprocessing utilities for the Liver Cirrhosis Prediction System.

Used identically at training time and at inference time so the model always
sees data in the same shape and encoding it was trained on. Also holds the
clinical validation ranges the web form and the batch importer both check
against.
"""

import os

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
CATEGORICAL_BINARY = ["Sex", "Ascites", "Hepatomegaly", "Spiders"]
CATEGORICAL_ORDINAL = {"Edema": {"N": 0, "S": 1, "Y": 2}}
NUMERIC_COLUMNS = [
    "Age", "Bilirubin", "Cholesterol", "Albumin", "Copper",
    "Alk_Phos", "SGOT", "Tryglicerides", "Platelets", "Prothrombin", "Stage",
]
FEATURE_COLUMNS = [
    "Age", "Sex", "Ascites", "Hepatomegaly", "Spiders", "Edema",
    "Bilirubin", "Cholesterol", "Albumin", "Copper", "Alk_Phos",
    "SGOT", "Tryglicerides", "Platelets", "Prothrombin", "Stage",
]
TARGET_COLUMN = "Risk"

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
ENCODER_PATH = os.path.join(BASE_DIR, "saved_model", "encoders.joblib")
IMPUTER_PATH = os.path.join(BASE_DIR, "saved_model", "imputer_values.joblib")

# Maps the app's form-field names onto dataset column names.
FORM_TO_COLUMN = {
    "age": "Age",
    "gender": "Sex",
    "ascites": "Ascites",
    "hepatomegaly": "Hepatomegaly",
    "spiders": "Spiders",
    "edema": "Edema",
    "bilirubin": "Bilirubin",
    "cholesterol": "Cholesterol",
    "albumin": "Albumin",
    "copper": "Copper",
    "alk_phos": "Alk_Phos",
    "sgot": "SGOT",
    "triglycerides": "Tryglicerides",
    "platelets": "Platelets",
    "prothrombin": "Prothrombin",
    "stage": "Stage",
}

# Human-readable labels + units, shared by the form, the PDF and the API docs.
FIELD_META = {
    "age":           {"label": "Age",                   "unit": "years",   "type": "number"},
    "gender":        {"label": "Gender",                "unit": "",        "type": "choice", "choices": ["M", "F"]},
    "bilirubin":     {"label": "Bilirubin",             "unit": "mg/dL",   "type": "number"},
    "cholesterol":   {"label": "Cholesterol",           "unit": "mg/dL",   "type": "number"},
    "albumin":       {"label": "Albumin",               "unit": "g/dL",    "type": "number"},
    "copper":        {"label": "Urine Copper",          "unit": "µg/day",  "type": "number"},
    "alk_phos":      {"label": "Alkaline Phosphatase",  "unit": "U/L",     "type": "number"},
    "sgot":          {"label": "SGOT (AST)",            "unit": "U/mL",    "type": "number"},
    "triglycerides": {"label": "Triglycerides",         "unit": "mg/dL",   "type": "number"},
    "platelets":     {"label": "Platelets",             "unit": "10⁹/L",   "type": "number"},
    "prothrombin":   {"label": "Prothrombin Time",      "unit": "seconds", "type": "number"},
    "ascites":       {"label": "Ascites",               "unit": "",        "type": "choice", "choices": ["Y", "N"]},
    "hepatomegaly":  {"label": "Hepatomegaly",          "unit": "",        "type": "choice", "choices": ["Y", "N"]},
    "spiders":       {"label": "Spider Angiomas",       "unit": "",        "type": "choice", "choices": ["Y", "N"]},
    "edema":         {"label": "Edema",                 "unit": "",        "type": "choice", "choices": ["N", "S", "Y"]},
    "stage":         {"label": "Histologic Stage",      "unit": "1-4",     "type": "choice", "choices": ["1", "2", "3", "4"]},
}

# Physiologically acceptable input ranges. Anything outside these is rejected
# with a clear message rather than silently fed to the model, where it would
# produce a confident but meaningless answer.
VALID_RANGES = {
    "age": (1, 120),
    "bilirubin": (0.1, 50),
    "cholesterol": (50, 2000),
    "albumin": (0.5, 7.0),
    "copper": (1, 700),
    "alk_phos": (20, 25000),
    "sgot": (5, 1000),
    "triglycerides": (10, 1500),
    "platelets": (5, 1200),
    "prothrombin": (7, 40),
    "stage": (1, 4),
}

# Typical reference ranges, shown next to each result so the clinician can see
# at a glance which of the patient's values are abnormal.
REFERENCE_RANGES = {
    "bilirubin": (0.1, 1.2),
    "cholesterol": (125, 200),
    "albumin": (3.5, 5.0),
    "copper": (15, 60),
    "alk_phos": (44, 147),
    "sgot": (8, 45),
    "triglycerides": (0, 150),
    "platelets": (150, 400),
    "prothrombin": (9.5, 13.5),
}

NUMERIC_FORM_FIELDS = [
    "age", "bilirubin", "cholesterol", "albumin", "copper",
    "alk_phos", "sgot", "triglycerides", "platelets", "prothrombin",
]
CHOICE_FORM_FIELDS = {
    "gender": {"M", "F"},
    "ascites": {"Y", "N"},
    "hepatomegaly": {"Y", "N"},
    "spiders": {"Y", "N"},
    "edema": {"N", "S", "Y"},
    "stage": {"1", "2", "3", "4"},
}
REQUIRED_FORM_FIELDS = NUMERIC_FORM_FIELDS + list(CHOICE_FORM_FIELDS)


class ValidationError(ValueError):
    """Raised when submitted clinical data can't be used for a prediction.

    Carries a per-field error map so the UI can highlight exactly what's wrong.
    """

    def __init__(self, errors: dict):
        self.errors = errors
        super().__init__("; ".join(f"{k}: {v}" for k, v in errors.items()))


def validate_clinical_input(raw: dict) -> dict:
    """Validate and coerce a submitted patient record.

    Returns a dict of cleaned values (floats/ints for numerics, canonical
    single-letter codes for categoricals). Raises ValidationError with a
    per-field message map if anything is missing, unparseable, or outside the
    physiologically plausible range.
    """
    errors: dict[str, str] = {}
    cleaned: dict = {}

    for field in NUMERIC_FORM_FIELDS:
        raw_value = raw.get(field)
        if raw_value is None or str(raw_value).strip() == "":
            errors[field] = f"{FIELD_META[field]['label']} is required."
            continue
        try:
            value = float(str(raw_value).strip())
        except (TypeError, ValueError):
            errors[field] = f"{FIELD_META[field]['label']} must be a number."
            continue
        lo, hi = VALID_RANGES[field]
        if not (lo <= value <= hi):
            unit = FIELD_META[field]["unit"]
            errors[field] = (
                f"{FIELD_META[field]['label']} must be between {lo} and {hi} {unit}".strip() + "."
            )
            continue
        cleaned[field] = value

    for field, allowed in CHOICE_FORM_FIELDS.items():
        raw_value = raw.get(field)
        if raw_value is None or str(raw_value).strip() == "":
            errors[field] = f"{FIELD_META[field]['label']} is required."
            continue
        value = str(raw_value).strip().upper()
        # Accept a few friendly spellings from CSV imports / API clients.
        aliases = {
            "MALE": "M", "FEMALE": "F",
            "YES": "Y", "NO": "N", "TRUE": "Y", "FALSE": "N",
            "NONE": "N", "SLIGHT": "S",
        }
        value = aliases.get(value, value)
        if value not in allowed:
            errors[field] = (
                f"{FIELD_META[field]['label']} must be one of: {', '.join(sorted(allowed))}."
            )
            continue
        cleaned[field] = int(value) if field == "stage" else value

    if errors:
        raise ValidationError(errors)

    return cleaned


def flag_abnormal(cleaned: dict) -> dict:
    """Compare each value against its reference range.

    Returns {field: {"status": "low"|"normal"|"high", "range": (lo, hi)}} so
    the UI can colour-code the patient's panel.
    """
    flags = {}
    for field, (lo, hi) in REFERENCE_RANGES.items():
        value = cleaned.get(field)
        if value is None:
            continue
        if value < lo:
            status = "low"
        elif value > hi:
            status = "high"
        else:
            status = "normal"
        flags[field] = {
            "status": status,
            "range": (lo, hi),
            "value": value,
            "label": FIELD_META[field]["label"],
            "unit": FIELD_META[field]["unit"],
        }
    return flags


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------
class Preprocessor:
    """Handles cleaning, missing-value imputation, and categorical encoding."""

    def __init__(self):
        self.sex_map = {"M": 1, "F": 0}
        self.binary_map = {"Y": 1, "N": 0}
        self.edema_map = CATEGORICAL_ORDINAL["Edema"]
        self.impute_values = {}

    def fit(self, df: pd.DataFrame, save_imputer: bool = True):
        df = df.copy()
        for col in NUMERIC_COLUMNS:
            if col in df.columns:
                median = df[col].median()
                # A fully-empty column yields NaN; fall back to 0 so transform
                # can never reintroduce NaN into the feature matrix.
                self.impute_values[col] = 0.0 if pd.isna(median) else float(median)
        if save_imputer:
            os.makedirs(os.path.dirname(IMPUTER_PATH), exist_ok=True)
            joblib.dump(self.impute_values, IMPUTER_PATH)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Input is missing required columns: {', '.join(missing)}")

        # Impute numeric missing values with training-time medians.
        for col in NUMERIC_COLUMNS:
            if col in df.columns:
                fill_value = self.impute_values.get(col)
                if fill_value is None:
                    median = df[col].median()
                    fill_value = 0.0 if pd.isna(median) else float(median)
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(fill_value)

        # Encode categoricals. Unrecognised labels fall back to the reference
        # level (0) rather than becoming NaN.
        if "Sex" in df.columns:
            df["Sex"] = df["Sex"].map(self.sex_map).fillna(0).astype(int)
        for col in ["Ascites", "Hepatomegaly", "Spiders"]:
            if col in df.columns:
                df[col] = df[col].map(self.binary_map).fillna(0).astype(int)
        if "Edema" in df.columns:
            df["Edema"] = df["Edema"].map(self.edema_map).fillna(0).astype(int)

        return df[FEATURE_COLUMNS].astype(float)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def save(self, path=ENCODER_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path=ENCODER_PATH):
        return joblib.load(path)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_and_clean_dataset(path=None) -> pd.DataFrame:
    """Load the training dataset and apply schema-level cleaning.

    Tolerates the raw UCI/Kaggle `cirrhosis.csv` layout as well as our own
    generated file: it drops the ID/administrative columns the raw file has,
    converts Age from days to years when needed, and derives the binary `Risk`
    target from the `Status`/`Stage` columns if it isn't already present.
    """
    if path is None:
        path = os.path.join(BASE_DIR, "dataset", "cirrhosis.csv")

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # Harmonise the common alternative spellings found in public copies.
    renames = {
        "Triglycerides": "Tryglicerides",
        "Alk_phos": "Alk_Phos",
        "Sex ": "Sex",
        "Platelet": "Platelets",
    }
    df = df.rename(columns={k: v for k, v in renames.items() if k in df.columns})

    # Drop administrative columns present in the raw UCI file.
    for col in ["ID", "N_Days", "Drug", "Unnamed: 0"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # The raw dataset stores Age in days; ours already stores years.
    if "Age" in df.columns and df["Age"].dropna().gt(200).any():
        df["Age"] = (df["Age"] / 365.25).round(1)

    # Derive the target if the file doesn't carry one.
    if TARGET_COLUMN not in df.columns:
        if "Status" in df.columns:
            # 'D' = died, 'CL' = censored due to transplant -> adverse outcome
            df[TARGET_COLUMN] = df["Status"].isin(["D", "CL"]).astype(int)
        elif "Stage" in df.columns:
            df[TARGET_COLUMN] = (pd.to_numeric(df["Stage"], errors="coerce") >= 3).astype(int)
        else:
            raise ValueError(
                "Dataset has no 'Risk' column and no 'Status'/'Stage' column to derive it from."
            )
    if "Status" in df.columns:
        df = df.drop(columns=["Status"])

    df = df.drop_duplicates()
    # A row with no target is unusable for supervised training.
    df = df.dropna(subset=[TARGET_COLUMN])
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    return df.reset_index(drop=True)


def encode_single_input(form_dict: dict, preprocessor: Preprocessor) -> pd.DataFrame:
    """Turn one patient's submitted values into a model-ready single-row frame.

    Runs full validation first, so a caller that skips `validate_clinical_input`
    still can't push out-of-range data into the model.
    """
    cleaned = validate_clinical_input(form_dict)
    row = {column: cleaned[form_field] for form_field, column in FORM_TO_COLUMN.items()}
    return preprocessor.transform(pd.DataFrame([row]))


def rows_from_dataframe(df: pd.DataFrame) -> list[dict]:
    """Convert an uploaded batch CSV into a list of form-style dicts.

    Accepts either the app's form-field names (age, sgot, ...) or the dataset's
    column names (Age, SGOT, ...), case-insensitively.
    """
    column_lookup = {}
    for form_field, dataset_col in FORM_TO_COLUMN.items():
        column_lookup[form_field.lower()] = form_field
        column_lookup[dataset_col.lower()] = form_field
    # A couple of friendly aliases people actually put in spreadsheets.
    column_lookup.update({
        "sex": "gender",
        "triglycerides": "triglycerides",
        "ast": "sgot",
        "alp": "alk_phos",
        "patient": "patient_name",
        "patient_name": "patient_name",
        "name": "patient_name",
        "patient_id": "patient_id",
        "mrn": "patient_id",
    })

    records = []
    for _, raw_row in df.iterrows():
        record = {}
        for column, value in raw_row.items():
            key = column_lookup.get(str(column).strip().lower())
            if key is None:
                continue
            if pd.isna(value):
                continue
            record[key] = value
        records.append(record)
    return records
