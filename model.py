"""
model.py
---------
Serving-side wrapper around the trained classifier and its preprocessor.

Responsibilities:
  * load the model + preprocessor + metrics once and reuse them
  * validate and encode a single patient record
  * return a label, calibrated probability, and risk band
  * produce a per-prediction explanation (which of this patient's values pushed
    the risk up or down, and by how much)
  * expose global feature importances and training metrics for the reports page
"""

import json
import os
import threading

import numpy as np
import pandas as pd

from preprocessing import (
    FEATURE_COLUMNS,
    FORM_TO_COLUMN,
    FIELD_META,
    Preprocessor,
    encode_single_input,
    validate_clinical_input,
)

# Risk band cut-offs on the predicted probability of the positive class.
RISK_BANDS = [
    (0.75, "Critical"),
    (0.50, "High"),
    (0.25, "Moderate"),
    (0.00, "Low"),
]

# Column name -> the form field it came from, for mapping explanations back to
# labels the user recognises.
COLUMN_TO_FORM = {column: form_field for form_field, column in FORM_TO_COLUMN.items()}


class ModelNotTrainedError(RuntimeError):
    """Raised when the app is started before `train_model.py` has been run."""


class CirrhosisPredictor:
    """Thread-safe, lazily-loaded predictor."""

    def __init__(self, model_path, encoder_path, metrics_path=None):
        self.model_path = model_path
        self.encoder_path = encoder_path
        self.metrics_path = metrics_path

        missing = [p for p in (model_path, encoder_path) if not os.path.exists(p)]
        if missing:
            raise ModelNotTrainedError(
                "Trained model artefacts not found: "
                + ", ".join(os.path.basename(p) for p in missing)
                + ". Run `python train_model.py` first."
            )

        import joblib

        self.model = joblib.load(model_path)
        self.preprocessor = Preprocessor.load(encoder_path)
        self._lock = threading.Lock()

        self.metrics = {}
        if metrics_path and os.path.exists(metrics_path):
            try:
                with open(metrics_path, encoding="utf-8") as f:
                    self.metrics = json.load(f)
            except (ValueError, OSError):
                self.metrics = {}

        self.model_name = self.metrics.get("model_name", type(self.model).__name__)

        # Population medians from training, used as the comparison baseline in
        # explanations ("this patient's bilirubin vs a typical patient's").
        self.baseline = self.metrics.get("feature_medians", {})

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, form_dict: dict):
        """Returns (result_label, probability_of_high_risk, risk_level).

        Raises `preprocessing.ValidationError` if the input is unusable.
        """
        X = encode_single_input(form_dict, self.preprocessor)
        with self._lock:
            proba = self.model.predict_proba(X)[0]

        # Locate the positive class robustly rather than assuming index 1.
        classes = list(getattr(self.model, "classes_", [0, 1]))
        positive_index = classes.index(1) if 1 in classes else len(classes) - 1
        prob_high_risk = float(proba[positive_index])

        # Clamp: no clinical model should report exactly 0% or 100%.
        prob_high_risk = float(np.clip(prob_high_risk, 0.01, 0.99))

        result_label = "High Risk" if prob_high_risk >= 0.5 else "Low Risk"
        risk_level = self.risk_band(prob_high_risk)

        return result_label, prob_high_risk, risk_level

    @staticmethod
    def risk_band(probability: float) -> str:
        for threshold, label in RISK_BANDS:
            if probability >= threshold:
                return label
        return "Low"

    def predict_batch(self, records: list[dict]) -> list[dict]:
        """Predict for many records, isolating per-row validation failures.

        One bad row in an uploaded CSV must not abort the whole file, so each
        row's outcome is reported individually.
        """
        from preprocessing import ValidationError

        results = []
        for index, record in enumerate(records):
            try:
                label, probability, band = self.predict(record)
                results.append({
                    "row": index + 1,
                    "ok": True,
                    "cleaned": validate_clinical_input(record),
                    "patient_name": str(record.get("patient_name") or f"Row {index + 1}"),
                    "patient_id": str(record.get("patient_id") or ""),
                    "result": label,
                    "probability": probability,
                    "risk_level": band,
                })
            except ValidationError as e:
                results.append({
                    "row": index + 1,
                    "ok": False,
                    "patient_name": str(record.get("patient_name") or f"Row {index + 1}"),
                    "errors": e.errors,
                })
            except Exception as e:  # pragma: no cover - defensive
                results.append({
                    "row": index + 1,
                    "ok": False,
                    "patient_name": str(record.get("patient_name") or f"Row {index + 1}"),
                    "errors": {"_": str(e)},
                })
        return results

    # ------------------------------------------------------------------
    # Explanation
    # ------------------------------------------------------------------
    def explain(self, form_dict: dict, top_n: int = 6) -> list[dict]:
        """Explain one prediction by measuring each feature's actual effect.

        Method: re-score the patient with one feature at a time replaced by the
        training-set median, and record how much the predicted probability
        moves. A feature whose removal drops the risk was pushing the risk up;
        one whose removal raises it was holding the risk down. This is a
        leave-one-out occlusion analysis — model-agnostic, needs no extra
        dependency, and unlike global feature importance it reflects *this*
        patient's values.
        """
        cleaned = validate_clinical_input(form_dict)
        row = {column: cleaned[field] for field, column in FORM_TO_COLUMN.items()}
        base_frame = pd.DataFrame([row])
        X_base = self.preprocessor.transform(base_frame)

        classes = list(getattr(self.model, "classes_", [0, 1]))
        positive_index = classes.index(1) if 1 in classes else len(classes) - 1

        with self._lock:
            base_probability = float(self.model.predict_proba(X_base)[0][positive_index])

            # Build every counterfactual first, then score in one batched call.
            counterfactuals = []
            considered = []
            for column in FEATURE_COLUMNS:
                baseline_value = self.baseline.get(column)
                if baseline_value is None:
                    baseline_value = float(self.preprocessor.impute_values.get(column, 0.0))
                variant = X_base.copy()
                variant.iloc[0, X_base.columns.get_loc(column)] = float(baseline_value)
                counterfactuals.append(variant)
                considered.append((column, float(baseline_value)))

            if not counterfactuals:
                return []

            stacked = pd.concat(counterfactuals, ignore_index=True)
            probabilities = self.model.predict_proba(stacked)[:, positive_index]

        contributions = []
        for (column, baseline_value), counterfactual_probability in zip(considered, probabilities):
            delta = base_probability - float(counterfactual_probability)
            if abs(delta) < 1e-6:
                continue
            form_field = COLUMN_TO_FORM.get(column, column.lower())
            meta = FIELD_META.get(form_field, {"label": column, "unit": ""})
            contributions.append({
                "feature": column,
                "label": meta["label"],
                "unit": meta.get("unit", ""),
                "patient_value": cleaned.get(form_field),
                "typical_value": round(baseline_value, 2),
                "impact": round(delta, 4),
                "impact_pct": round(delta * 100, 1),
                "direction": "increases risk" if delta > 0 else "decreases risk",
            })

        contributions.sort(key=lambda c: abs(c["impact"]), reverse=True)
        return contributions[:top_n]

    # ------------------------------------------------------------------
    # Metrics / global importance
    # ------------------------------------------------------------------
    def top_features(self, n: int = 5) -> list[tuple[str, float]]:
        """Global feature importance from training, most important first."""
        importance = self.metrics.get("feature_importance", {})
        if not importance:
            raw = getattr(self.model, "feature_importances_", None)
            if raw is None:
                # Ensemble: average across sub-estimators that expose importances.
                estimators = getattr(self.model, "estimators_", None)
                if estimators is None:
                    base = getattr(self.model, "estimator", None)
                    estimators = [base] if base is not None else []
                arrays = [getattr(e, "feature_importances_", None) for e in estimators]
                arrays = [a for a in arrays if a is not None]
                if not arrays:
                    return []
                raw = np.mean(arrays, axis=0)
            importance = dict(zip(FEATURE_COLUMNS, [float(v) for v in raw]))
        ordered = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
        return ordered[:n]

    def labelled_top_features(self, n: int = 5) -> list[dict]:
        """Same as `top_features` but with display labels for the templates."""
        output = []
        for column, importance in self.top_features(n):
            form_field = COLUMN_TO_FORM.get(column, column.lower())
            meta = FIELD_META.get(form_field, {"label": column})
            output.append({
                "feature": column,
                "label": meta["label"],
                "importance": importance,
                "importance_pct": round(importance * 100, 1),
            })
        return output

    @property
    def model_comparison(self) -> list[dict]:
        """Per-algorithm scores recorded by the training script, if present."""
        return self.metrics.get("model_comparison", [])

    @property
    def is_stale(self) -> bool:
        """True when the artefacts were written by a different sklearn version.

        A version mismatch is the usual cause of subtly wrong predictions after
        an environment change, so the admin page surfaces it.
        """
        recorded = self.metrics.get("sklearn_version")
        if not recorded:
            return False
        import sklearn

        return recorded != sklearn.__version__
