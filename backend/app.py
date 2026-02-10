"""Flask API + UI server for Hospital Readmission Prediction.

- Loads a saved XGBoost model artifact (preprocessing + model) from ../model
- If the model file does not exist, trains a model using backend/train_model.py
- Serves the frontend from ../frontend
- Provides /api/predict endpoint for probability + risk category + recommendation
- (Optional) Stores prediction history in SQLite
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from db import init_db, insert_prediction


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "model"
FRONTEND_DIR = ROOT_DIR / "frontend"
DATASET_PATH = ROOT_DIR / "data" / "diabetic_data.csv"

MODEL_PATH = MODEL_DIR / "xgboost_model.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
DB_PATH = Path(__file__).resolve().parent / "predictions.db"


def risk_category(prob: float) -> str:
    if prob < 0.30:
        return "LOW"
    if prob <= 0.60:
        return "MEDIUM"
    return "HIGH"


def recommendation_for_risk(risk: str) -> str:
    if risk == "LOW":
        return "Standard discharge procedure"
    if risk == "MEDIUM":
        return "Follow-up appointment and medication reminder"
    return "Home nurse visit and frequent monitoring"


def load_or_train_model() -> dict[str, Any]:
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    # Train if missing
    from train_model import build_xy, load_dataset, save_artifacts, train_xgboost_pipeline

    df = load_dataset(DATASET_PATH)
    X, y = build_xy(df)
    pipeline, _result, metadata = train_xgboost_pipeline(X, y)

    save_artifacts(
        pipeline=pipeline,
        metadata=metadata,
        model_path=MODEL_PATH,
        metadata_path=METADATA_PATH,
    )

    return joblib.load(MODEL_PATH)


artifact = load_or_train_model()
PIPELINE = artifact["pipeline"]
FEATURE_COLUMNS: list[str] = list(artifact["feature_columns"])
NUMERIC_FEATURES: set[str] = set(artifact.get("numeric_features", []))

init_db(DB_PATH)

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
CORS(app)


@app.get("/")
def index() -> Any:
    return send_from_directory(str(FRONTEND_DIR), "index.html")


@app.get("/api/health")
def health() -> Any:
    return jsonify(
        {
            "status": "ok",
            "model_loaded": True,
            "model_path": str(MODEL_PATH),
            "features": len(FEATURE_COLUMNS),
        }
    )


@app.post("/api/predict")
def predict() -> Any:
    payload = request.get_json(silent=True) or {}

    # Create a full feature dict so the pipeline sees the expected columns.
    record: dict[str, Any] = {col: None for col in FEATURE_COLUMNS}

    # Accept either flat JSON or {"features": {...}}
    features = payload.get("features") if isinstance(payload, dict) else None
    if isinstance(features, dict):
        incoming = features
    else:
        incoming = payload if isinstance(payload, dict) else {}

    for k, v in incoming.items():
        if k in record:
            record[k] = v

    df = pd.DataFrame([record], columns=FEATURE_COLUMNS)

    # Coerce numeric fields (empty strings -> NaN)
    for c in FEATURE_COLUMNS:
        if c in NUMERIC_FEATURES:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    prob = float(PIPELINE.predict_proba(df)[0, 1])
    risk = risk_category(prob)
    rec = recommendation_for_risk(risk)

    created_at = datetime.now(timezone.utc).isoformat()
    try:
        insert_prediction(DB_PATH, created_at, prob, risk, record)
    except Exception:
        # Prediction should still work even if DB write fails.
        pass

    return jsonify(
        {
            "probability": prob,
            "probability_percent": round(prob * 100, 2),
            "risk": risk,
            "recommendation": rec,
            "created_at": created_at,
        }
    )


# Serve other frontend assets (style.css, script.js)
@app.get("/<path:path>")
def static_proxy(path: str) -> Any:
    return send_from_directory(str(FRONTEND_DIR), path)


if __name__ == "__main__":
    # For Windows: use built-in dev server. Production can use waitress.
    app.run(host="127.0.0.1", port=5000, debug=False)
