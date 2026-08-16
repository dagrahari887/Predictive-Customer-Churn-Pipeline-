"""
feature_engineering.py
---------------------------------
Transforms cleaned raw data (from etl_pipeline.py) into model-ready
RFM + behavioral features. Saves the final feature table to
data/features.parquet, and saves the fitted scaler/encoder to models/.
 
Usage:
    python feature_engineering.py
"""
 
import pandas as pd
import numpy as np
import joblib
import os
 
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder
 
REFERENCE_DATE = pd.Timestamp("today")
CHURN_DAYS = 60
 
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)
 
 
def build_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates transaction-level rows into one row per customer
    with RFM + behavioral features.
    """
 
    # ── RFM Aggregation ────────────────────────
    rfm = df.groupby("customer_id").agg(
        recency=("txn_date", lambda x: (REFERENCE_DATE - x.max()).days
                 if x.notna().any() else 9999),
        frequency=("txn_date", "count"),
        monetary=("amount", "sum"),
    ).reset_index()
 
    # ── Derived Metrics ────────────────────────
    rfm["avg_ticket_size"] = (
        rfm["monetary"] / rfm["frequency"].replace(0, np.nan)
    ).fillna(0).round(2)
 
    # ── Support Behavior Features ──────────────
    support = df.groupby("customer_id").agg(
        num_support_tickets=("issue_type", "count"),
        avg_satisfaction=("satisfaction_score", "mean"),
        pct_unresolved=("resolution_status",
                         lambda x: (x == "Pending").mean() if len(x) > 0 else 0)
    ).reset_index()
 
    # ── Customer-level Demographics ────────────
    demo = df.drop_duplicates("customer_id")[[
        "customer_id", "age", "gender", "city",
        "subscription_plan", "signup_date"
    ]].copy()
    demo["tenure_days"] = (
        REFERENCE_DATE - pd.to_datetime(demo["signup_date"])
    ).dt.days
 
    # ── Merge all feature sets ──────────────────
    master = (
        demo
        .merge(rfm, on="customer_id", how="left")
        .merge(support, on="customer_id", how="left")
    )
 
    # Fill customers with zero support tickets / zero transactions
    master["num_support_tickets"] = master["num_support_tickets"].fillna(0)
    master["avg_satisfaction"] = master["avg_satisfaction"].fillna(3.0)
    master["pct_unresolved"] = master["pct_unresolved"].fillna(0)
    master["recency"] = master["recency"].fillna(9999)
    master["frequency"] = master["frequency"].fillna(0)
    master["monetary"] = master["monetary"].fillna(0)
    master["avg_ticket_size"] = master["avg_ticket_size"].fillna(0)
 
    # ── Churn Label: 60-day inactivity rule ─────
    master["is_churned"] = (master["recency"] > CHURN_DAYS).astype("int8")
 
    print(f"Built features for {len(master)} customers.")
    print(f"Churn rate: {master.is_churned.mean():.1%}")
    return master
 
 
def encode_and_scale(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes categoricals and scales numerical features.
    Saves the fitted scaler + target encoder for reuse at inference time.
    """
    df = df.copy()
 
    # ── One-Hot Encoding (low-cardinality) ─────
    df = pd.get_dummies(
        df, columns=["gender", "subscription_plan"], drop_first=True
    )
 
    # ── Target Encoding for city (high-cardinality) ─
    te = TargetEncoder(cols=["city"], smoothing=2.0)
    df["city"] = te.fit_transform(df[["city"]], df["is_churned"])
 
    # ── Feature Scaling ─────────────────────────
    scale_cols = [
        "age", "recency", "frequency", "monetary",
        "avg_ticket_size", "tenure_days", "num_support_tickets",
        "avg_satisfaction", "pct_unresolved"
    ]
    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])
 
    # Save scaler + encoder for API inference later
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(te, "models/target_encoder.pkl")
    print("Saved scaler.pkl and target_encoder.pkl to models/")
 
    return df
 
 
if __name__ == "__main__":
    print("Loading cleaned data...")
    raw_df = pd.read_parquet("data/cleaned_raw.parquet")
 
    print("\nBuilding RFM features...")
    rfm_df = build_rfm_features(raw_df)
 
    print("\nEncoding and scaling...")
    final_df = encode_and_scale(rfm_df)
 
    final_df.to_parquet("data/features.parquet", index=False)
    print(f"\nSaved {len(final_df)} rows to data/features.parquet ✓")