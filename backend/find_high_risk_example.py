"""Utility: find high-probability examples from the dataset for demo.

Run:
  D:/anti_project/.venv/Scripts/python.exe backend/find_high_risk_example.py

It prints quantiles of predicted probabilities and the top-scoring rows (a subset
of columns that match the frontend form).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    artifact = joblib.load(root / "model" / "xgboost_model.pkl")
    pipeline = artifact["pipeline"]
    features = list(artifact["feature_columns"])

    df = pd.read_csv(root / "data" / "diabetic_data.csv").replace("?", np.nan)

    X = df.drop(columns=[c for c in ["readmitted", "encounter_id", "patient_nbr"] if c in df.columns])
    X = X.reindex(columns=features)

    # Use a sample for speed
    X_sub = X.sample(n=min(20000, len(X)), random_state=42)
    proba = pipeline.predict_proba(X_sub)[:, 1]

    qs = np.quantile(proba, [0, 0.5, 0.9, 0.95, 0.99, 0.999, 1])
    print("n=", len(proba))
    print(
        "quantiles=",
        dict(zip(["min", "p50", "p90", "p95", "p99", "p99.9", "max"], map(float, qs))),
    )

    keys = [
        "race",
        "gender",
        "age",
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses",
        "diag_1",
        "diag_2",
        "diag_3",
        "A1Cresult",
        "insulin",
        "change",
        "diabetesMed",
        "admission_type_id",
        "discharge_disposition_id",
        "admission_source_id",
    ]

    idx = np.argsort(-proba)[:5]
    for rank, i in enumerate(idx, start=1):
        row = X_sub.iloc[i]
        p = float(proba[i])
        print("\nTop", rank, "prob=", p)
        out = {k: (None if k not in row.index else row[k]) for k in keys}
        print(out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
