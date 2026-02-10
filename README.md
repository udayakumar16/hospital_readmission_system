# Hospital Readmission Prediction and Prevention System (XGBoost)

A complete, runnable mini-project that predicts **30-day hospital readmission risk** using an **XGBoost** classifier trained on an EHR-style dataset.

- **Model**: XGBoost (`xgboost.XGBClassifier`)
- **Preprocessing**: Missing value handling + categorical encoding using a scikit-learn `Pipeline`
- **API**: Flask (`/api/predict`) returns probability + risk tier + care recommendation
- **UI**: Simple HTML/CSS/JS form-based frontend
- **Optional storage**: SQLite prediction history

## Folder Structure

```
anti_project/
├── data/
│   └── diabetic_data.csv
├── model/
│   ├── xgboost_model.pkl
│   └── model_metadata.json
├── backend/
│   ├── app.py
│   ├── db.py
│   ├── train_model.py
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
└── README.md
```

## Target Definition

This project uses the dataset column `readmitted`:
- `"<30"` → **Positive class (1)**: readmitted within 30 days
- `">30"` and `"NO"` → **Negative class (0)**

## Risk Classification Logic

- Probability `< 0.30` → **LOW**
- Probability `0.30 – 0.60` → **MEDIUM**
- Probability `> 0.60` → **HIGH**

## Care Recommendations

- **LOW**: Standard discharge procedure
- **MEDIUM**: Follow-up appointment and medication reminder
- **HIGH**: Home nurse visit and frequent monitoring

## Setup (Windows)

From the repo root:

```powershell
cd d:\anti_project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
```

## Train the XGBoost Model

This saves the model pipeline to `model/xgboost_model.pkl` (and metadata to `model/model_metadata.json`).

```powershell
python backend\train_model.py
```

## Run the Web App (Backend + Frontend)

The Flask server also serves the frontend from the `frontend/` folder.

```powershell
python backend\app.py
```

Open:
- http://127.0.0.1:5000

## API Usage

### Health

`GET /api/health`

### Predict

`POST /api/predict` with JSON body. You can send a subset of fields (others will be treated as missing and handled by the pipeline):

```json
{
  "race": "Caucasian",
  "gender": "Female",
  "age": "[70-80)",
  "time_in_hospital": 3,
  "num_lab_procedures": 40,
  "num_procedures": 1,
  "num_medications": 16,
  "number_outpatient": 0,
  "number_emergency": 0,
  "number_inpatient": 0,
  "number_diagnoses": 7,
  "diag_1": "250.83",
  "diag_2": "403",
  "diag_3": "V27",
  "A1Cresult": "None",
  "insulin": "Up",
  "change": "Ch",
  "diabetesMed": "Yes",
  "admission_type_id": 1,
  "discharge_disposition_id": 1,
  "admission_source_id": 7
}
```

Response:
- `probability` (0–1)
- `probability_percent`
- `risk`
- `recommendation`

## Notes

- Accuracy can be high because the positive class (readmitted within 30 days) is typically **imbalanced**. This project also reports ROC-AUC, precision, recall, and F1-score in `model/model_metadata.json`.
- This is an educational project; do not use it for real clinical decision-making without proper validation and governance.
