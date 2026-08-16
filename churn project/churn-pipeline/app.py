from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Optional
import numpy as np
import pandas as pd
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0",
    description="Returns real-time churn probability for a given customer"
)

# ─────────────────────────────────────────────
#  Load artifacts ONCE at startup (not per request)
# ─────────────────────────────────────────────
try:
    MODEL  = joblib.load("models/rf_model.pkl")
    SCALER = joblib.load("models/scaler.pkl")
    TARGET_ENC = joblib.load("models/target_encoder.pkl")
    logger.info("Model artifacts loaded ✓")
except FileNotFoundError as e:
    raise RuntimeError(f"Model file not found: {e}")

# ─────────────────────────────────────────────
#  REQUEST SCHEMA — Pydantic enforces types & validation
# ─────────────────────────────────────────────
class CustomerFeatures(BaseModel):
    age:                  int   = Field(..., ge=16, le=85)
    tenure_days:          int   = Field(..., ge=0)
    recency:              int   = Field(..., ge=0,  description="Days since last purchase")
    frequency:            int   = Field(..., ge=1)
    monetary:             float = Field(..., ge=0.0)
    avg_ticket_size:      float = Field(..., ge=0.0)
    num_support_tickets:  int   = Field(0, ge=0)
    avg_satisfaction:     float = Field(3.0, ge=1.0, le=5.0)
    pct_unresolved:       float = Field(0.0, ge=0.0, le=1.0)
    city:                 str   = "Unknown"
    gender:               str   = "Unknown"
    subscription_plan:    str   = "Free"

class ChurnResponse(BaseModel):
    customer_churn_probability: float
    churn_risk_level:           str
    recommendation:             str

# ─────────────────────────────────────────────
#  PREPROCESSING — mirror the training pipeline exactly
# ─────────────────────────────────────────────
SCALE_COLS = [
    'age', 'recency', 'frequency', 'monetary',
    'avg_ticket_size', 'tenure_days', 'num_support_tickets',
    'avg_satisfaction', 'pct_unresolved'
]

def preprocess(data: CustomerFeatures) -> pd.DataFrame:
    row = data.dict()
    df  = pd.DataFrame([row])

    # Target encode city
    df['city'] = TARGET_ENC.transform(df[['city']])

    # One-hot encode (align to training columns)
    df = pd.get_dummies(df, columns=['gender', 'subscription_plan'],
                         drop_first=True)

    # Scale numerical features
    df[SCALE_COLS] = SCALER.transform(df[SCALE_COLS])

    # Ensure column alignment with training (add missing OHE columns as 0)
    for col in MODEL.named_steps['clf'].feature_names_in_:
        if col not in df.columns:
            df[col] = 0
    df = df[MODEL.named_steps['clf'].feature_names_in_]
    return df

# ─────────────────────────────────────────────
#  ENDPOINT
# ─────────────────────────────────────────────
@app.post("/predict", response_model=ChurnResponse)
async def predict_churn(customer: CustomerFeatures):
    """
    Accepts a customer's behavioral metrics and returns
    a churn probability between 0.0 and 100.0.
    """
    try:
        X = preprocess(customer)
        prob = MODEL.predict_proba(X)[0][1]  # probability of class=1 (churned)
        pct  = round(prob * 100, 2)

        if   pct >= 70: risk, rec = "HIGH",   "Immediate retention offer"
        elif pct >= 40: risk, rec = "MEDIUM", "Send re-engagement email"
        else:            risk, rec = "LOW",    "No action needed"

        logger.info(f"Prediction: {pct}% churn probability [{risk}]")
        return ChurnResponse(
            customer_churn_probability=pct,
            churn_risk_level=risk,
            recommendation=rec
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health")
def health(): return {"status": "ok", "model_loaded": MODEL is not None}