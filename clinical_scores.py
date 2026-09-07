"""
clinical_scores.py
-------------------
Established, literature-defined liver scoring formulas computed alongside the
ML prediction. These are deterministic clinical calculators, not learned
models, so they give the clinician a familiar reference point to sanity-check
the Random Forest output against.

Implemented:
  * MELD   - Model For End-Stage Liver Disease (UNOS/OPTN 2016 variant)
  * FIB-4  - Fibrosis-4 index
  * APRI   - AST to Platelet Ratio Index
  * Child-Pugh (partial) - the subset computable from the fields we collect

Caveats that matter for interpretation:
  - Our dataset uses the Mayo PBC schema, which has no creatinine, INR, or
    sodium. MELD needs creatinine + INR, so we approximate: INR is estimated
    from prothrombin time and creatinine falls back to a normal value. The
    result is therefore labelled an *estimate* everywhere it is displayed.
  - Platelets in the PBC dataset are recorded in units of 10^9/L (e.g. 250),
    which is what FIB-4/APRI expect, so no conversion is applied.
"""

import math

# Reference upper limit of normal for AST/SGOT, needed by APRI (U/L).
AST_UPPER_LIMIT_NORMAL = 40.0

# Prothrombin time (seconds) considered normal, used to estimate INR.
NORMAL_PROTHROMBIN_SECONDS = 11.0
# International Sensitivity Index typical of hospital thromboplastin reagents.
ISI = 1.0


def _safe_float(value, default=None):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def estimate_inr(prothrombin_seconds) -> float | None:
    """Approximate INR from prothrombin time.

    INR = (patient PT / normal PT) ** ISI. With ISI = 1 this reduces to a
    simple ratio, which is the standard approximation when the reagent ISI
    isn't recorded.
    """
    pt = _safe_float(prothrombin_seconds)
    if pt is None or pt <= 0:
        return None
    return round((pt / NORMAL_PROTHROMBIN_SECONDS) ** ISI, 2)


def meld_score(bilirubin, prothrombin, creatinine=1.0, sodium=None) -> float | None:
    """MELD score (UNOS variant), clamped to the official 6-40 range.

    MELD = 3.78*ln(bilirubin) + 11.2*ln(INR) + 9.57*ln(creatinine) + 6.43

    Each lab value is floored at 1.0 before the log, per the published rules,
    so a normal value contributes ln(1) = 0. Creatinine defaults to 1.0 (a
    normal value) because the PBC schema doesn't record it.
    """
    bili = _safe_float(bilirubin)
    inr = estimate_inr(prothrombin)
    creat = _safe_float(creatinine, 1.0)
    if bili is None or inr is None:
        return None

    bili = max(bili, 1.0)
    inr = max(inr, 1.0)
    creat = min(max(creat, 1.0), 4.0)  # capped at 4.0 per UNOS

    score = 3.78 * math.log(bili) + 11.2 * math.log(inr) + 9.57 * math.log(creat) + 6.43

    # MELD-Na adjustment when sodium is available (125-137 mmol/L window)
    na = _safe_float(sodium)
    if na is not None and 10 <= score:
        na = min(max(na, 125.0), 137.0)
        score = score + 1.32 * (137 - na) - (0.033 * score * (137 - na))

    return round(min(max(score, 6.0), 40.0), 1)


def fib4_score(age, sgot, platelets, alt=None) -> float | None:
    """FIB-4 = (Age * AST) / (Platelets * sqrt(ALT)).

    ALT isn't in the PBC schema. AST is substituted for ALT, which is the
    common fallback when only one transaminase is recorded; it makes the index
    an approximation rather than an exact FIB-4.
    """
    age_v = _safe_float(age)
    ast = _safe_float(sgot)
    plt = _safe_float(platelets)
    alt_v = _safe_float(alt, ast)

    if None in (age_v, ast, plt, alt_v) or plt <= 0 or alt_v <= 0:
        return None

    return round((age_v * ast) / (plt * math.sqrt(alt_v)), 2)


def apri_score(sgot, platelets) -> float | None:
    """APRI = ((AST / AST_ULN) * 100) / Platelet count (10^9/L)."""
    ast = _safe_float(sgot)
    plt = _safe_float(platelets)
    if ast is None or plt is None or plt <= 0:
        return None
    return round(((ast / AST_UPPER_LIMIT_NORMAL) * 100) / plt, 2)


# ---------------------------------------------------------------------------
# Interpretation bands (from the source literature)
# ---------------------------------------------------------------------------
def interpret_meld(score) -> tuple[str, str]:
    """Returns (band, plain-language meaning)."""
    if score is None:
        return "N/A", "Not enough data to estimate."
    if score <= 9:
        return "Low", "Approximately 1.9% three-month mortality; routine follow-up."
    if score <= 19:
        return "Moderate", "Approximately 6% three-month mortality; hepatology follow-up advised."
    if score <= 29:
        return "High", "Approximately 20% three-month mortality; transplant evaluation considered."
    return "Very High", "Above 50% three-month mortality; urgent specialist care."


def interpret_fib4(score) -> tuple[str, str]:
    if score is None:
        return "N/A", "Not enough data to calculate."
    if score < 1.45:
        return "Low", "Advanced fibrosis unlikely (high negative predictive value)."
    if score <= 3.25:
        return "Indeterminate", "Inconclusive zone; consider elastography or biopsy."
    return "High", "Advanced fibrosis (F3-F4) likely; specialist assessment indicated."


def interpret_apri(score) -> tuple[str, str]:
    if score is None:
        return "N/A", "Not enough data to calculate."
    if score < 0.5:
        return "Low", "Significant fibrosis unlikely."
    if score <= 1.5:
        return "Indeterminate", "Inconclusive zone; correlate with other markers."
    return "High", "Significant fibrosis / cirrhosis likely."


def child_pugh_partial(bilirubin, albumin, prothrombin, ascites, encephalopathy_grade=0):
    """Child-Pugh points from the components we actually collect.

    The full score also needs hepatic encephalopathy grading, which is a
    bedside neurological assessment and isn't in our feature set; it is
    assumed absent (0) unless supplied. The result is therefore a *partial*
    Child-Pugh and is labelled as such in the UI.
    """
    bili = _safe_float(bilirubin)
    alb = _safe_float(albumin)
    inr = estimate_inr(prothrombin)
    if None in (bili, alb, inr):
        return None, "N/A", "Not enough data."

    points = 0
    # Bilirubin (mg/dL)
    points += 1 if bili < 2 else (2 if bili <= 3 else 3)
    # Albumin (g/dL)
    points += 1 if alb > 3.5 else (2 if alb >= 2.8 else 3)
    # INR
    points += 1 if inr < 1.7 else (2 if inr <= 2.3 else 3)
    # Ascites
    ascites_flag = str(ascites or "N").upper()
    points += 3 if ascites_flag == "Y" else 1
    # Encephalopathy (assumed none unless provided)
    grade = int(_safe_float(encephalopathy_grade, 0) or 0)
    points += 1 if grade == 0 else (2 if grade <= 2 else 3)

    if points <= 6:
        return points, "A", "Well-compensated disease (1-year survival ~100%)."
    if points <= 9:
        return points, "B", "Significant functional compromise (1-year survival ~80%)."
    return points, "C", "Decompensated disease (1-year survival ~45%)."


def compute_all(inputs: dict) -> dict:
    """Compute every score from a form/dict of the 16 clinical fields.

    Accepts the app's form-field naming (age, sgot, platelets, ...).
    """
    meld = meld_score(inputs.get("bilirubin"), inputs.get("prothrombin"))
    fib4 = fib4_score(inputs.get("age"), inputs.get("sgot"), inputs.get("platelets"))
    apri = apri_score(inputs.get("sgot"), inputs.get("platelets"))
    cp_points, cp_class, cp_meaning = child_pugh_partial(
        inputs.get("bilirubin"), inputs.get("albumin"),
        inputs.get("prothrombin"), inputs.get("ascites"),
    )

    meld_band, meld_meaning = interpret_meld(meld)
    fib4_band, fib4_meaning = interpret_fib4(fib4)
    apri_band, apri_meaning = interpret_apri(apri)

    return {
        "meld": {
            "value": meld, "band": meld_band, "meaning": meld_meaning,
            "label": "MELD (estimated)",
            "note": "Estimated: creatinine unavailable in this dataset, INR derived from prothrombin time.",
        },
        "fib4": {
            "value": fib4, "band": fib4_band, "meaning": fib4_meaning,
            "label": "FIB-4 Index",
            "note": "AST substituted for ALT (ALT not recorded in this dataset).",
        },
        "apri": {
            "value": apri, "band": apri_band, "meaning": apri_meaning,
            "label": "APRI",
            "note": "AST upper limit of normal assumed to be 40 U/L.",
        },
        "child_pugh": {
            "value": cp_points, "band": cp_class, "meaning": cp_meaning,
            "label": "Child-Pugh (partial)",
            "note": "Encephalopathy grade assumed 0 (requires bedside assessment).",
        },
        "inr_estimate": estimate_inr(inputs.get("prothrombin")),
    }
