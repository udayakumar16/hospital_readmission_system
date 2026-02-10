"""Training script for Hospital Readmission Prediction (XGBoost).

Dataset: UCI Diabetes 130-US hospitals dataset (already present in this repo).
Target: readmitted within 30 days (readmitted == '<30')

This script trains an XGBoost classifier inside a scikit-learn Pipeline that includes
all preprocessing steps (imputation + one-hot encoding). The saved artifact can be
loaded directly by the Flask API for inference.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "model"

DEFAULT_DATASET_PATH = DATA_DIR / "diabetic_data.csv"
DEFAULT_MODEL_PATH = MODEL_DIR / "xgboost_model.pkl"
DEFAULT_METADATA_PATH = MODEL_DIR / "model_metadata.json"

TARGET_COL = "readmitted"
DROP_COLS = ["encounter_id", "patient_nbr"]


@dataclass
class TrainResult:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float


def _risk_category(prob: float) -> str:
    if prob < 0.30:
        return "LOW"
    if prob <= 0.60:
        return "MEDIUM"
    return "HIGH"


def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # In this dataset, missing values are often represented as '?'
    df = df.replace("?", np.nan)

    return df


def build_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset")

    # Binary label: 1 if readmitted within 30 days, else 0
    y = (df[TARGET_COL].astype(str) == "<30").astype(int)

    drop_cols = [c for c in [TARGET_COL, *DROP_COLS] if c in df.columns]
    X = df.drop(columns=drop_cols)

    return X, y


def train_xgboost_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
) -> tuple[Pipeline, TrainResult, dict[str, Any]]:
    numeric_features = [
        "admission_type_id",
        "discharge_disposition_id",
        "admission_source_id",
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses",
    ]
    numeric_features = [c for c in numeric_features if c in X.columns]
    categorical_features = [c for c in X.columns if c not in numeric_features]

    # Convert numeric columns to numeric dtype (coerce errors to NaN)
    X = X.copy()
    for c in numeric_features:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        min_child_weight=1,
        gamma=0.0,
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="logloss",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    proba = pipeline.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    result = TrainResult(
        accuracy=float(accuracy_score(y_test, preds)),
        precision=float(precision_score(y_test, preds, zero_division=0)),
        recall=float(recall_score(y_test, preds, zero_division=0)),
        f1=float(f1_score(y_test, preds, zero_division=0)),
        roc_auc=float(roc_auc_score(y_test, proba)),
    )

    # Some extra info that is useful at inference time
    example_probs = proba[:5].tolist()
    example_risks = [_risk_category(p) for p in example_probs]

    metadata: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(DEFAULT_DATASET_PATH.name),
        "target_definition": "1 if readmitted == '<30' else 0",
        "feature_columns": list(X.columns),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "metrics": asdict(result),
        "example_probabilities": example_probs,
        "example_risks": example_risks,
        "risk_thresholds": {
            "LOW": "p < 0.30",
            "MEDIUM": "0.30 <= p <= 0.60",
            "HIGH": "p > 0.60",
        },
    }

    return pipeline, result, metadata


def save_artifacts(
    pipeline: Pipeline,
    metadata: dict[str, Any],
    model_path: Path,
    metadata_path: Path,
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Save a single artifact so inference always uses the exact same preprocessing.
    artifact = {
        "pipeline": pipeline,
        "feature_columns": metadata["feature_columns"],
        "numeric_features": metadata["numeric_features"],
    }
    joblib.dump(artifact, model_path)

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_DATASET_PATH),
        help="Path to CSV dataset",
    )
    parser.add_argument(
        "--model-out",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Where to save the trained model artifact (joblib)",
    )
    args = parser.parse_args()

    csv_path = Path(args.data)
    model_out = Path(args.model_out)

    df = load_dataset(csv_path)
    X, y = build_xy(df)

    pipeline, result, metadata = train_xgboost_pipeline(X, y)

    save_artifacts(
        pipeline=pipeline,
        metadata=metadata,
        model_path=model_out,
        metadata_path=DEFAULT_METADATA_PATH,
    )

    print("=== Training Complete ===")
    print(f"Accuracy : {result.accuracy:.4f}")
    print(f"Precision: {result.precision:.4f}")
    print(f"Recall   : {result.recall:.4f}")
    print(f"F1-score : {result.f1:.4f}")
    print(f"ROC-AUC  : {result.roc_auc:.4f}")
    print(f"Saved model: {model_out}")
    print(f"Saved metadata: {DEFAULT_METADATA_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
