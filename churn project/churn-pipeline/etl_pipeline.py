import pandas as pd
import numpy as np
import mysql.connector
from sqlalchemy import create_engine

# ─────────────────────────────────────────────
#  1. CONNECTION — using SQLAlchemy for clean ORM-style access
# ─────────────────────────────────────────────
def get_engine():
    return create_engine(
        "mysql+mysqlconnector://root:1234@localhost/churn_db",
        pool_pre_ping=True   # auto-reconnect on stale connections
    )

# ─────────────────────────────────────────────
#  2. EXTRACT — denormalized JOIN across all 3 tables
# ─────────────────────────────────────────────
EXTRACT_QUERY = """
    SELECT
        c.customer_id,
        c.age,
        c.gender,
        c.city,
        c.signup_date,
        c.subscription_plan,
        c.is_churned,
        t.txn_date,
        t.amount,
        t.product_category,
        t.payment_method,
        s.issue_type,
        s.resolution_status,
        s.satisfaction_score
    FROM customers c
    LEFT JOIN transactions t ON c.customer_id = t.customer_id
    LEFT JOIN support_logs s ON c.customer_id = s.customer_id
"""

def extract_data(engine) -> pd.DataFrame:
    with engine.connect() as conn:
        df = pd.read_sql(EXTRACT_QUERY, conn)
    print(f"Extracted {len(df):,} rows | {df.customer_id.nunique():,} unique customers")
    return df

# ─────────────────────────────────────────────
#  3. CLEAN — handle missing values & outliers
# ─────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── Date parsing ──────────────────────────
    df['txn_date']    = pd.to_datetime(df['txn_date'],    errors='coerce')
    df['signup_date'] = pd.to_datetime(df['signup_date'], errors='coerce')

    # ── Missing value strategy ─────────────────
    #    amount: median imputation (robust to skew)
    #    satisfaction_score: mode (ordinal)
    #    categorical: 'Unknown' sentinel
    df['amount'] = df['amount'].fillna(df['amount'].median())
    df['satisfaction_score'] = (
        df['satisfaction_score']
        .fillna(df['satisfaction_score'].mode()[0])
    )
    for col in ['gender', 'city', 'subscription_plan',
                 'product_category', 'payment_method',
                 'issue_type', 'resolution_status']:
        df[col] = df[col].fillna('Unknown')

    # ── Outlier handling (IQR clipping on amount) ──
    Q1, Q3 = df['amount'].quantile([0.01, 0.99])
    df['amount'] = df['amount'].clip(lower=Q1, upper=Q3)

    # ── Type enforcement ───────────────────────
    df['age'] = df['age'].clip(16, 85).astype('Int64')
    df['is_churned'] = df['is_churned'].astype('int8')

    print(f"Cleaned. Remaining nulls: {df.isnull().sum().sum()}")
    return df

# ── Entry point ───────────────────────────────
if __name__ == "__main__":
    engine = get_engine()
    raw_df = extract_data(engine)
    clean_df = clean_data(raw_df)
    clean_df.to_parquet("data/cleaned_raw.parquet", index=False)
    print("Saved to Parquet ✓")