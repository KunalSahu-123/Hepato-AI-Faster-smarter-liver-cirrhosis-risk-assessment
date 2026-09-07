"""
train_model.py
----------------
End-to-end training pipeline for the Liver Cirrhosis Prediction System.

Steps:
 1. Load & clean the dataset
 2. EDA charts
 3. Preprocess (impute + encode)
 4. Stratified train/test split
 5. Train and compare several algorithms with cross-validation
 6. Hyperparameter-tune the best family with GridSearchCV
 7. Calibrate probabilities so the confidence score is meaningful
 8. Evaluate (accuracy, precision, recall, F1, ROC-AUC, confusion matrix)
 9. Generate evaluation plots
10. Serialize model + preprocessor + metrics

Run:  python train_model.py
      python train_model.py --quick     (skip the grid search, for a fast rerun)
"""

import argparse
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    average_precision_score, brier_score_loss,
)
from sklearn.model_selection import (
    train_test_split, GridSearchCV, cross_val_score, learning_curve, StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from preprocessing import (
    Preprocessor, load_and_clean_dataset, FEATURE_COLUMNS, TARGET_COLUMN,
)

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_model")
STATIC_CHARTS_DIR = os.path.join(BASE_DIR, "static", "images", "charts")
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "cirrhosis.csv")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(STATIC_CHARTS_DIR, exist_ok=True)

sns.set_style("whitegrid")
PALETTE = ["#1a73e8", "#e8534f", "#34a853", "#fbbc04", "#8e44ad", "#16a085"]
RANDOM_STATE = 42


def _save(fig_name: str) -> None:
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_CHARTS_DIR, fig_name), dpi=110)
    plt.close()


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------
def run_eda(df: pd.DataFrame) -> dict:
    """Generate exploratory charts and return a small summary for the UI."""
    plt.figure(figsize=(7, 4))
    sns.histplot(df["Age"], kde=True, color=PALETTE[0])
    plt.title("Age Distribution")
    plt.xlabel("Age (years)")
    _save("age_distribution.png")

    plt.figure(figsize=(5, 4))
    counts = df["Sex"].value_counts()
    plt.bar([{"M": "Male", "F": "Female"}.get(k, k) for k in counts.index],
            counts.values, color=PALETTE[:len(counts)])
    plt.title("Gender Distribution")
    plt.ylabel("Patients")
    _save("gender_distribution.png")

    plt.figure(figsize=(5, 4))
    stage_counts = df["Stage"].value_counts().sort_index()
    plt.bar([f"Stage {int(s)}" for s in stage_counts.index], stage_counts.values, color=PALETTE[2])
    plt.title("Histologic Stage Distribution")
    plt.ylabel("Patients")
    _save("stage_distribution.png")

    plt.figure(figsize=(5, 4))
    risk_counts = df[TARGET_COLUMN].value_counts().sort_index()
    plt.bar(["Low Risk", "High Risk"][: len(risk_counts)], risk_counts.values,
            color=[PALETTE[2], PALETTE[1]][: len(risk_counts)])
    plt.title("Target Class Balance")
    plt.ylabel("Patients")
    _save("class_balance.png")

    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(9, 7))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", annot=False, center=0)
    plt.title("Correlation Heatmap")
    _save("correlation_heatmap.png")

    # Which markers separate the two classes most visibly
    key_markers = ["Bilirubin", "Albumin", "Platelets", "Prothrombin", "Copper", "SGOT"]
    available = [m for m in key_markers if m in df.columns]
    if available:
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        for ax, marker in zip(axes.ravel(), available):
            sns.boxplot(data=df, x=TARGET_COLUMN, y=marker, ax=ax,
                        hue=TARGET_COLUMN, palette=[PALETTE[2], PALETTE[1]],
                        legend=False)
            ax.set_xticklabels(["Low Risk", "High Risk"])
            ax.set_xlabel("")
            ax.set_title(marker)
        for ax in axes.ravel()[len(available):]:
            ax.axis("off")
        plt.suptitle("Key Marker Distributions by Risk Class")
        _save("marker_boxplots.png")

    return {
        "rows": int(len(df)),
        "features": len(FEATURE_COLUMNS),
        "positive_rate": round(float(df[TARGET_COLUMN].mean()), 4),
        "missing_cells": int(df.isna().sum().sum()),
    }


# ---------------------------------------------------------------------------
# Evaluation plots
# ---------------------------------------------------------------------------
def plot_confusion_matrix(cm) -> None:
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Low Risk", "High Risk"],
                yticklabels=["Low Risk", "High Risk"])
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    _save("confusion_matrix.png")


def plot_roc_curve(y_test, y_proba) -> None:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, color=PALETTE[0], lw=2, label=f"ROC AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    _save("roc_curve.png")


def plot_pr_curve(y_test, y_proba) -> None:
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision, color=PALETTE[1], lw=2, label=f"Avg precision = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    _save("precision_recall_curve.png")


def plot_feature_importance(importances, feature_names) -> dict:
    order = np.argsort(importances)[::-1]
    plt.figure(figsize=(7, 5))
    sns.barplot(x=np.asarray(importances)[order],
                y=np.asarray(feature_names)[order], color=PALETTE[0])
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    _save("feature_importance.png")
    return dict(zip(np.asarray(feature_names)[order].tolist(),
                    np.asarray(importances)[order].astype(float).tolist()))


def plot_learning_curve(model, X, y) -> None:
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 6), n_jobs=-1, random_state=RANDOM_STATE,
    )
    plt.figure(figsize=(6, 4))
    plt.plot(train_sizes, train_scores.mean(axis=1), "o-", color=PALETTE[0], label="Training")
    plt.fill_between(train_sizes, train_scores.mean(1) - train_scores.std(1),
                     train_scores.mean(1) + train_scores.std(1), alpha=0.12, color=PALETTE[0])
    plt.plot(train_sizes, val_scores.mean(axis=1), "o-", color=PALETTE[1], label="Validation")
    plt.fill_between(train_sizes, val_scores.mean(1) - val_scores.std(1),
                     val_scores.mean(1) + val_scores.std(1), alpha=0.12, color=PALETTE[1])
    plt.xlabel("Training Examples")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve")
    plt.legend(loc="lower right")
    _save("learning_curve.png")


def plot_model_comparison(comparison: list[dict]) -> None:
    names = [row["model"] for row in comparison]
    scores = [row["cv_f1"] for row in comparison]
    plt.figure(figsize=(7, 4))
    bars = plt.barh(names, scores, color=PALETTE[0])
    bars[int(np.argmax(scores))].set_color(PALETTE[2])
    plt.xlabel("Cross-validated F1 score")
    plt.title("Algorithm Comparison (5-fold CV)")
    plt.xlim(0, 1)
    for name, score in zip(names, scores):
        plt.text(score + 0.01, name, f"{score:.3f}", va="center", fontsize=9)
    _save("model_comparison.png")


def plot_calibration(y_test, y_proba) -> None:
    fraction_positive, mean_predicted = calibration_curve(y_test, y_proba, n_bins=10, strategy="quantile")
    plt.figure(figsize=(5, 4))
    plt.plot(mean_predicted, fraction_positive, "o-", color=PALETTE[0], label="Model")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfectly calibrated")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Probability Calibration")
    plt.legend(loc="upper left")
    _save("calibration_curve.png")


# ---------------------------------------------------------------------------
# Candidate models
# ---------------------------------------------------------------------------
def candidate_models() -> dict:
    """Algorithms compared before committing to one.

    Scale-sensitive learners (SVM, logistic regression) are wrapped in a
    pipeline with a scaler; the tree ensembles don't need one.
    """
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=250, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=250, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=6, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                       random_state=RANDOM_STATE)),
        ]),
        # SVC has no native predict_proba; wrapping it in a calibrator both
        # supplies one and avoids the deprecated `probability=True` path.
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(
                SVC(class_weight="balanced", random_state=RANDOM_STATE), cv=3
            )),
        ]),
    }


def compare_models(X_train, y_train) -> list[dict]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, estimator in candidate_models().items():
        f1 = cross_val_score(estimator, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
        auc = cross_val_score(estimator, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        accuracy = cross_val_score(estimator, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        rows.append({
            "model": name,
            "cv_f1": round(float(f1.mean()), 4),
            "cv_f1_std": round(float(f1.std()), 4),
            "cv_roc_auc": round(float(auc.mean()), 4),
            "cv_accuracy": round(float(accuracy.mean()), 4),
        })
        print(f"  {name:<20} F1={f1.mean():.4f}  ROC-AUC={auc.mean():.4f}  Acc={accuracy.mean():.4f}")

    rows.sort(key=lambda r: r["cv_f1"], reverse=True)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(quick: bool = False) -> dict:
    print("Step 1: Loading dataset...")
    df = load_and_clean_dataset(DATASET_PATH)
    print(f"  Rows after cleaning: {len(df)}  |  positive rate: {df[TARGET_COLUMN].mean():.3f}")

    print("Step 2: EDA charts...")
    dataset_summary = run_eda(df)

    print("Step 3: Preprocessing (impute + encode)...")
    pre = Preprocessor().fit(df)
    X = pre.transform(df)
    y = df[TARGET_COLUMN].astype(int)

    print("Step 4: Train/test split (80/20, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train: {len(X_train)}   Test: {len(X_test)}")

    print("Step 5: Comparing candidate algorithms (5-fold CV)...")
    comparison = compare_models(X_train, y_train)
    best_family = comparison[0]["model"]
    print(f"  Best by CV F1: {best_family}")

    print("Step 6: Hyperparameter tuning & building ensemble...")
    if quick:
        rf = RandomForestClassifier(
            n_estimators=250, min_samples_leaf=2, min_samples_split=5,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
        )
        best_params = {"n_estimators": 250, "min_samples_leaf": 2, "min_samples_split": 5}
    else:
        param_grid = {
            "n_estimators": [200, 300, 500],
            "max_depth": [8, 12, None],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt", "log2"],
        }
        grid = GridSearchCV(
            RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1),
            param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=0,
        )
        grid.fit(X_train, y_train)
        rf = grid.best_estimator_
        best_params = grid.best_params_
    print(f"  RF best params: {best_params}")

    # Build a soft-voting ensemble of the top three tree families for
    # smoother probability estimates and better generalisation.
    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        min_samples_leaf=4, random_state=RANDOM_STATE,
    )
    et = ExtraTreesClassifier(
        n_estimators=300, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    model = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb), ("et", et)],
        voting="soft", n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("  Voting ensemble (RF + GradientBoosting + ExtraTrees) trained")

    print("Step 7: Calibrating probabilities...")
    # Sigmoid calibration produces smooth, continuous probabilities (never
    # exact 0 or 1) and generalises better on small datasets than isotonic.
    calibrated = CalibratedClassifierCV(model, method="sigmoid", cv=5)
    calibrated.fit(X_train, y_train)

    uncalibrated_brier = brier_score_loss(y_test, model.predict_proba(X_test)[:, 1])
    calibrated_brier = brier_score_loss(y_test, calibrated.predict_proba(X_test)[:, 1])
    print(f"  Brier score: {uncalibrated_brier:.4f} (raw) -> {calibrated_brier:.4f} (calibrated)")

    # Keep calibration only when it genuinely helps on held-out data.
    if calibrated_brier <= uncalibrated_brier:
        final_model = calibrated
        calibration_applied = True
    else:
        final_model = model
        calibration_applied = False
    print(f"  Calibration {'applied' if calibration_applied else 'skipped (raw model scored better)'}")

    print("Step 8: Cross-validation of the final model...")
    cv_scores = cross_val_score(final_model, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)
    print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    print("Step 9: Evaluating on the held-out test set...")
    y_pred = final_model.predict(X_test)
    y_proba = final_model.predict_proba(X_test)[:, 1]

    metrics = {
        "model_name": "Ensemble (RF+GB+ET)" + (" (sigmoid-calibrated)" if calibration_applied else ""),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "average_precision": float(average_precision_score(y_test, y_proba)),
        "brier_score": float(calibrated_brier if calibration_applied else uncalibrated_brier),
        "brier_score_uncalibrated": float(uncalibrated_brier),
        "cv_mean_accuracy": float(cv_scores.mean()),
        "cv_std_accuracy": float(cv_scores.std()),
        "best_params": {k: (None if v is None else v) for k, v in best_params.items()},
        "calibration_applied": calibration_applied,
        "best_family_by_cv": best_family,
        "test_size": int(len(X_test)),
        "train_size": int(len(X_train)),
    }
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Low Risk", "High Risk"])
    print(report)

    print("Step 10: Generating evaluation plots...")
    plot_confusion_matrix(cm)
    plot_roc_curve(y_test, y_proba)
    plot_pr_curve(y_test, y_proba)
    plot_learning_curve(model, X_train, y_train)
    plot_model_comparison(comparison)
    plot_calibration(y_test, y_proba)
    # Average importances across the ensemble members that expose them.
    fi_arrays = []
    for name, est in model.named_estimators_.items():
        raw = getattr(est, "feature_importances_", None)
        if raw is not None:
            fi_arrays.append(raw)
    avg_importances = np.mean(fi_arrays, axis=0) if fi_arrays else np.zeros(len(FEATURE_COLUMNS))
    feature_importance = plot_feature_importance(avg_importances, FEATURE_COLUMNS)

    print("Step 11: Saving model + preprocessor + metrics...")
    joblib.dump(final_model, os.path.join(MODEL_DIR, "random_forest_model.joblib"))
    pre.save(os.path.join(MODEL_DIR, "encoders.joblib"))

    metrics_out = dict(metrics)
    metrics_out["classification_report"] = report
    metrics_out["confusion_matrix"] = cm.tolist()
    metrics_out["feature_importance"] = feature_importance
    metrics_out["model_comparison"] = comparison
    metrics_out["dataset_summary"] = dataset_summary
    # Medians of the encoded training features: the baseline the per-prediction
    # explanation compares each patient against.
    metrics_out["feature_medians"] = {
        column: float(X_train[column].median()) for column in FEATURE_COLUMNS
    }
    metrics_out["sklearn_version"] = sklearn.__version__
    metrics_out["trained_at"] = pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M:%S UTC")

    with open(os.path.join(MODEL_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)

    print(f"\nDone. Accuracy {metrics['accuracy']:.2%} | ROC-AUC {metrics['roc_auc']:.3f} | "
          f"F1 {metrics['f1_score']:.3f}")
    print(f"Model saved to {os.path.join(MODEL_DIR, 'random_forest_model.joblib')}")
    return metrics_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the cirrhosis risk model.")
    parser.add_argument("--quick", action="store_true",
                        help="Skip the grid search for a faster run.")
    args = parser.parse_args()
    main(quick=args.quick)
