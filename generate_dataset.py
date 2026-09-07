"""
generate_dataset.py
--------------------
Generates a synthetic liver cirrhosis dataset that follows the same schema and
realistic value ranges as the well-known Mayo Clinic Primary Biliary Cirrhosis
(PBC) study dataset (the dataset commonly used for this kind of project on
Kaggle/UCI).

NOTE: This environment has no internet access to Kaggle/UCI, so a live
download is not possible. This generator produces a statistically realistic
stand-in dataset (same columns, realistic distributions, correlated risk
factors) so the full pipeline (EDA -> preprocessing -> training -> evaluation
-> serving) is genuinely functional end-to-end.

To use a REAL dataset instead:
1. Download "cirrhosis.csv" (Mayo Clinic PBC dataset) from Kaggle/UCI.
2. Place it at dataset/cirrhosis.csv with the same column names used below.
3. Re-run train_model.py -- no other code changes are required.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

N = 1200

def generate():
    age_days = np.random.normal(50 * 365, 10 * 365, N).clip(18 * 365, 85 * 365)
    sex = np.random.choice(["M", "F"], size=N, p=[0.11, 0.89])

    stage = np.random.choice([1, 2, 3, 4], size=N, p=[0.18, 0.27, 0.30, 0.25])

    # Correlate clinical markers with disease stage to make the data learnable
    bilirubin = np.random.gamma(2 + stage * 0.8, 1.0, N)
    cholesterol = np.random.normal(280 + stage * 25, 80, N).clip(120, 1200)
    albumin = np.random.normal(3.9 - stage * 0.15, 0.4, N).clip(1.9, 4.8)
    copper = np.random.normal(60 + stage * 35, 40, N).clip(4, 600)
    alk_phos = np.random.normal(1500 + stage * 400, 900, N).clip(280, 14000)
    sgot = np.random.normal(90 + stage * 15, 35, N).clip(25, 350)
    triglycerides = np.random.normal(120 + stage * 10, 45, N).clip(30, 500)
    platelets = np.random.normal(280 - stage * 25, 70, N).clip(60, 550)
    prothrombin = np.random.normal(10.5 + stage * 0.35, 1.0, N).clip(9, 18)

    def prob_from_stage(base_p, per_stage):
        p = np.clip(base_p + per_stage * (stage - 1), 0.02, 0.95)
        return np.random.binomial(1, p)

    ascites = prob_from_stage(0.02, 0.10)
    hepatomegaly = prob_from_stage(0.15, 0.12)
    spiders = prob_from_stage(0.10, 0.10)
    edema = np.random.choice(["N", "S", "Y"], size=N,
                              p=None) if False else None

    edema_list = []
    for s in stage:
        r = np.random.rand()
        if s <= 1:
            edema_list.append("N" if r < 0.85 else "S")
        elif s == 2:
            edema_list.append("N" if r < 0.7 else ("S" if r < 0.9 else "Y"))
        else:
            edema_list.append("N" if r < 0.45 else ("S" if r < 0.75 else "Y"))

    # Target: risk of cirrhosis progression / decompensation (binary)
    risk_score = (
        0.55 * (stage >= 3).astype(int)
        + 0.15 * ascites
        + 0.10 * hepatomegaly
        + 0.10 * spiders
        + 0.10 * (bilirubin > np.percentile(bilirubin, 65)).astype(int)
        + 0.08 * (albumin < 3.3).astype(int)
        + 0.07 * (prothrombin > 12).astype(int)
        + np.random.normal(0, 0.12, N)
    )
    cirrhosis_risk = (risk_score > np.percentile(risk_score, 55)).astype(int)

    df = pd.DataFrame({
        "Age": (age_days / 365).round(1),
        "Sex": sex,
        "Ascites": np.where(ascites == 1, "Y", "N"),
        "Hepatomegaly": np.where(hepatomegaly == 1, "Y", "N"),
        "Spiders": np.where(spiders == 1, "Y", "N"),
        "Edema": edema_list,
        "Bilirubin": bilirubin.round(2),
        "Cholesterol": cholesterol.round(1),
        "Albumin": albumin.round(2),
        "Copper": copper.round(1),
        "Alk_Phos": alk_phos.round(1),
        "SGOT": sgot.round(1),
        "Tryglicerides": triglycerides.round(1),
        "Platelets": platelets.round(1),
        "Prothrombin": prothrombin.round(2),
        "Stage": stage,
        "Risk": cirrhosis_risk
    })

    # Inject a small % of missing values, like real clinical data
    for col in ["Cholesterol", "Copper", "Tryglicerides", "Platelets"]:
        mask = np.random.rand(N) < 0.04
        df.loc[mask, col] = np.nan

    return df


if __name__ == "__main__":
    os.makedirs("dataset", exist_ok=True)
    df = generate()
    df.to_csv("dataset/cirrhosis.csv", index=False)
    print(f"Generated dataset/cirrhosis.csv with {len(df)} rows")
    print(df["Risk"].value_counts(normalize=True))
