"""
predict_from_excel.py
---------------------------------
Takes ANY store's transaction-level Excel file (one row per order/sale)
and predicts churn probability for every customer in it.

Designed for files like the Global Superstore dataset, but works with
any sheet that has: a customer identifier, an order date, and a sale
amount. Column names are auto-detected (case-insensitive, flexible).

HOW IT WORKS:
  1. Reads the Excel file
  2. Auto-detects customer_id / order_date / sales_amount columns
  3. Builds RFM features per customer (Recency, Frequency, Monetary)
  4. Reference date = the LATEST order date in the file (not today's date) —
     this matters for historical datasets like Global Superstore (2011-2014)
  5. Trains a fresh Random Forest on this data's own churn pattern
     (since we don't have a pre-trained model for arbitrary stores)
  6. Outputs an Excel file with a Churn_Probability_% column per customer

Usage:
    python predict_from_excel.py "path/to/your_store_data.xlsx"
"""

import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

CHURN_DAYS = 60  # business rule: no purchase in 60 days => churned


# ─────────────────────────────────────────────
#  STEP 1 — Auto-detect relevant columns
# ─────────────────────────────────────────────
def detect_columns(df: pd.DataFrame) -> dict:
    """
    Scans column names (case-insensitive, punctuation-insensitive) for
    likely matches to customer_id, order_date, and sales_amount.

    Two-pass strategy:
      Pass 1: match against a wide list of common naming keywords
      Pass 2 (fallback): if a column is still missing, guess by DATA TYPE
        - order_date: first column that parses mostly as dates
        - sales: first remaining numeric column with positive values
        - customer_id: first remaining text/object column with many
          repeated values (customers appear in multiple rows)

    Raises a clear error only if both passes fail.
    """
    # Normalize: lowercase, strip, remove underscores/spaces/punctuation
    def norm(s):
        return "".join(ch for ch in str(s).lower() if ch.isalnum())

    cols_norm = {norm(c): c for c in df.columns}

    def find(*keywords):
        for kw in keywords:
            kw_n = norm(kw)
            for norm_name, original_name in cols_norm.items():
                if kw_n in norm_name:
                    return original_name
        return None

    mapping = {
        "customer_id": find(
            "customer id", "customerid", "cust id", "custid",
            "client id", "clientid", "user id", "userid",
            "customer no", "custno", "customer number", "member id"
        ),
        "order_date": find(
            "order date", "orderdate", "purchase date", "purchasedate",
            "txn date", "transaction date", "sale date", "saledate",
            "invoice date", "date of purchase", "timestamp", "date"
        ),
        "sales": find(
            "sales", "amount", "total", "revenue", "price",
            "order value", "ordervalue", "net sales", "grand total",
            "invoice amount", "spend"
        ),
        "customer_name": find("customer name", "client name", "cust name"),
        "category": find(
            "category", "product category", "item category",
            "product type", "department"
        ),
    }

    # ── Pass 2: fallback by data type, for whatever is still missing ──
    used_cols = {v for v in mapping.values() if v}

    if mapping["order_date"] is None:
        import warnings
        for col in df.columns:
            if col in used_cols:
                continue
            # Skip plain numeric columns entirely — pandas will happily
            # "parse" integers like 1001 as epoch nanosecond timestamps
            # (giving bogus 1970-01-01-ish dates), which is a false
            # positive, not a real date column.
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
            if parsed.notna().mean() > 0.7:  # >70% of values look like dates
                mapping["order_date"] = col
                used_cols.add(col)
                break

    if mapping["sales"] is None:
        numeric_cols = df.select_dtypes(include="number").columns
        for col in numeric_cols:
            if col in used_cols:
                continue
            if (df[col] >= 0).mean() > 0.9:  # mostly non-negative = likely an amount
                mapping["sales"] = col
                used_cols.add(col)
                break

    if mapping["customer_id"] is None:
        # A real customer ID column should repeat (each customer places
        # multiple orders), so it shouldn't be 100% unique, but it should
        # still have MANY distinct values — far more than a low-cardinality
        # column like "Region" or "Category". Instead of taking the first
        # vaguely-plausible match, score every remaining column by
        # cardinality and pick the one that looks most ID-like.
        best_col, best_unique = None, 0
        n_rows = len(df)
        for col in df.columns:
            if col in used_cols:
                continue
            n_unique = df[col].nunique()
            if 1 < n_unique < n_rows * 0.95 and n_unique > best_unique:
                best_col, best_unique = col, n_unique
        if best_col is not None:
            mapping["customer_id"] = best_col
            used_cols.add(best_col)

    missing = [k for k in ("customer_id", "order_date", "sales") if mapping[k] is None]
    if missing:
        raise ValueError(
            f"Could not detect required columns: {missing}. "
            f"Available columns: {list(df.columns)}. "
            "Try renaming columns to something like "
            "'Customer ID', 'Order Date', 'Sales'."
        )

    print("Detected columns:")
    for k, v in mapping.items():
        if v:
            print(f"  {k:15s} -> '{v}'")
    return mapping


# ─────────────────────────────────────────────
#  STEP 2 — Build RFM features (with category-aware churn threshold)
# ─────────────────────────────────────────────
def compute_dynamic_threshold(df: pd.DataFrame, cid_col: str, date_col: str,
                               cat_col: str = None, multiplier: float = 2.5,
                               min_threshold: int = 30, max_threshold: int = 365) -> dict:
    """
    Instead of one fixed 60-day rule for every business, this calculates
    a CHURN THRESHOLD PER CATEGORY based on actual buying behavior in
    the data itself:

        threshold = median(days_between_repeat_purchases) * multiplier

    Logic: if customers in "Groceries" typically buy every 10 days, a
    25-day gap (2.5x) is a meaningful churn signal. If "Furniture"
    customers typically buy every 70 days, the same 25-day gap would be
    completely normal — so furniture needs a much longer threshold.

    Returns a dict: {category_name: threshold_in_days}.
    If no category column exists, returns {"__global__": default 60}.
    """
    if cat_col is None or cat_col not in df.columns:
        return {"__global__": CHURN_DAYS}

    thresholds = {}
    for cat, sub in df.groupby(cat_col):
        sub = sub.sort_values([cid_col, date_col])
        gaps = (
            sub.groupby(cid_col)[date_col]
            .apply(lambda x: x.diff().dt.days.dropna())
            .explode()
            .dropna()
        )
        if len(gaps) < 5:
            # Not enough repeat-purchase data for this category — use
            # the global default rather than an unreliable estimate.
            thresholds[cat] = CHURN_DAYS
            continue

        median_gap = gaps.median()
        threshold = median_gap * multiplier
        # Clamp to sane bounds so one noisy category doesn't produce a
        # threshold of 2 days or 3 years.
        threshold = max(min_threshold, min(max_threshold, threshold))
        thresholds[cat] = round(threshold)

    return thresholds


def build_features(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    cid_col   = col_map["customer_id"]
    date_col  = col_map["order_date"]
    sales_col = col_map["sales"]
    name_col  = col_map.get("customer_name")
    cat_col   = col_map.get("category")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    df = df.dropna(subset=[cid_col, date_col, sales_col])

    # Reference date = most recent transaction in the WHOLE dataset.
    # (Using real "today" would mark everyone churned on historical data.)
    reference_date = df[date_col].max()
    print(f"\nReference date (latest order in file): {reference_date.date()}")

    # ── Dynamic, category-aware churn threshold ────────────────────
    thresholds = compute_dynamic_threshold(df, cid_col, date_col, cat_col)
    if cat_col:
        print(f"\nCategory-aware churn thresholds (data-driven, not fixed):")
        for cat, days in sorted(thresholds.items(), key=lambda x: x[1]):
            print(f"  {cat:20s} -> churned if inactive > {days} days")
    else:
        print(f"\nNo category column detected — using a single global "
              f"threshold of {CHURN_DAYS} days for all customers.")

    # ── Per-customer RFM (and, if available, their dominant category) ──
    rfm = df.groupby(cid_col).agg(
        recency=(date_col, lambda x: (reference_date - x.max()).days),
        frequency=(date_col, "count"),
        monetary=(sales_col, "sum"),
    ).reset_index()
    rfm["avg_ticket_size"] = (rfm["monetary"] / rfm["frequency"]).round(2)

    if cat_col:
        # IMPORTANT: use the category of each customer's MOST RECENT
        # purchase (not their all-time "favorite" category). A customer
        # who buys across multiple categories should be judged by what
        # they bought last — that's what their recency gap is actually
        # measuring. Using an all-time dominant category here was tested
        # and found to misclassify customers who buy broadly across
        # categories, inflating the churn rate artificially.
        last_purchase_idx = df.groupby(cid_col)[date_col].idxmax()
        last_category = df.loc[last_purchase_idx].set_index(cid_col)[cat_col]
        rfm["last_purchase_category"] = rfm[cid_col].map(last_category)
        rfm["churn_threshold_days"] = rfm["last_purchase_category"].map(thresholds).fillna(CHURN_DAYS)
    else:
        rfm["churn_threshold_days"] = CHURN_DAYS

    rfm["is_churned"] = (rfm["recency"] > rfm["churn_threshold_days"]).astype(int)

    if name_col:
        names = df.drop_duplicates(cid_col)[[cid_col, name_col]]
        rfm = rfm.merge(names, on=cid_col, how="left")

    rfm = rfm.rename(columns={cid_col: "customer_id"})

    print(f"\nBuilt features for {len(rfm)} customers.")
    print(f"Overall churn rate (category-aware thresholds): {rfm['is_churned'].mean():.1%}")
    if cat_col:
        print("\nChurn rate by dominant category:")
        cat_summary = rfm.groupby("last_purchase_category")["is_churned"].agg(["mean", "count"])
        cat_summary.columns = ["churn_rate", "num_customers"]
        cat_summary["churn_rate"] = (cat_summary["churn_rate"] * 100).round(1)
        print(cat_summary.sort_values("churn_rate", ascending=False).to_string())

    return rfm


# ─────────────────────────────────────────────
#  STEP 3 — Train a model on THIS dataset's pattern
#  (no recency in the feature set — it created the label, so it's excluded
#   to avoid leakage, same fix we applied earlier)
# ─────────────────────────────────────────────
def train_and_predict(rfm: pd.DataFrame) -> pd.DataFrame:
    feature_cols = ["frequency", "monetary", "avg_ticket_size"]
    X = rfm[feature_cols]
    y = rfm["is_churned"]

    # Edge case: if every customer is in the same class (all churned or
    # all active), there is nothing for a classifier to learn, and
    # predict_proba() would crash trying to return a 2nd column that
    # does not exist. Fall back to a recency-based risk score instead.
    if y.nunique() < 2:
        only_class = "churned" if y.iloc[0] == 1 else "active"
        print("WARNING: every customer in this file is '" + only_class +
              "' -- there is no variation for a classifier to learn from. "
              "Falling back to a recency-based risk score instead.")
        max_recency = max(rfm["recency"].max(), 1)
        rfm["Churn_Probability_%"] = (
            (rfm["recency"] / max_recency) * 100
        ).round(2)
        rfm["Churn_Risk"] = rfm["Churn_Probability_%"].apply(
            lambda p: "HIGH" if p >= 70 else "MEDIUM" if p >= 40 else "LOW"
        )
        return rfm

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

    # If churn class is too small for a clean split, skip SMOTE and just
    # use class_weight balancing — small datasets break SMOTE's k_neighbors.
    min_class_count = y.value_counts().min()
    use_smote = min_class_count >= 6

    if use_smote:
        pipeline = ImbPipeline([
            ("smote", SMOTE(random_state=42, k_neighbors=min(5, min_class_count - 1))),
            ("clf", RandomForestClassifier(
                n_estimators=300, class_weight="balanced",
                random_state=42, n_jobs=-1
            ))
        ])
    else:
        pipeline = ImbPipeline([
            ("clf", RandomForestClassifier(
                n_estimators=300, class_weight="balanced",
                random_state=42, n_jobs=-1
            ))
        ])

    pipeline.fit(X_scaled, y)

    # Predict probability for EVERY customer (in-sample, since the goal
    # here is to score the full customer base, not hold out a test set)
    probs = pipeline.predict_proba(X_scaled)[:, 1]
    rfm["Churn_Probability_%"] = (probs * 100).round(2)

    def risk_label(p):
        if p >= 70:
            return "HIGH"
        elif p >= 40:
            return "MEDIUM"
        return "LOW"

    rfm["Churn_Risk"] = rfm["Churn_Probability_%"].apply(risk_label)
    return rfm


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main(input_path: str):
    print(f"Reading: {input_path}")
    df = pd.read_excel(input_path)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns.\n")

    col_map = detect_columns(df)
    rfm = build_features(df, col_map)
    result = train_and_predict(rfm)

    # Reorder columns for a clean output report
    display_cols = ["customer_id"]
    if "Customer Name" in result.columns:
        display_cols.append("Customer Name")
    if "last_purchase_category" in result.columns:
        display_cols += ["last_purchase_category", "churn_threshold_days"]
    display_cols += ["recency", "frequency", "monetary",
                      "avg_ticket_size", "Churn_Probability_%", "Churn_Risk"]
    result = result[display_cols].sort_values("Churn_Probability_%", ascending=False)

    output_path = input_path.rsplit(".", 1)[0] + "_churn_predictions.xlsx"
    result.to_excel(output_path, index=False)

    print(f"\nSaved predictions to: {output_path}")
    print(f"\nTop 10 highest-risk customers:")
    print(result.head(10).to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_from_excel.py <path_to_excel_file>")
        sys.exit(1)
    main(sys.argv[1])