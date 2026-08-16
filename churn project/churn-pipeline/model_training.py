"""
model_training.py
---------------------------------
Loads engineered features, trains Random Forest + Logistic Regression
with SMOTE / class_weight balancing and hyperparameter tuning via
GridSearchCV. Saves the best models to models/.
 
Usage:
    python model_training.py
"""
 
import pandas as pd
import numpy as np
import joblib
import os
 
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
 
os.makedirs("models", exist_ok=True)
 
 
def main():
    print("Loading engineered features...")
    df = pd.read_parquet("data/features.parquet")
 
    # Build the feature column list dynamically (handles OHE columns)
    base_cols = [
        "age", "tenure_days","frequency",
        "monetary", "avg_ticket_size", "num_support_tickets",
        "avg_satisfaction", "pct_unresolved", "city"
    ]
    ohe_cols = [c for c in df.columns
                if c.startswith(("gender_", "subscription_plan_"))]
    feature_cols = base_cols + ohe_cols
 
    X = df[feature_cols]
    y = df["is_churned"]
 
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")
    print(f"Overall churn rate: {y.mean():.1%}")
 
    # ── Train / Test Split ──────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train churn rate: {y_train.mean():.1%} | Test: {y_test.mean():.1%}")
 
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
 
    # ── MODEL A: Random Forest with SMOTE ───────
    print("\nTraining Random Forest (with GridSearchCV)...")
    rf_pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=42, k_neighbors=min(5, y_train.sum() - 1))),
        ("clf", RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=42, n_jobs=-1
        ))
    ])
    rf_param_grid = {
        "clf__max_depth": [10, 20, None],
        "clf__min_samples_split": [2, 5, 10],
        "clf__max_features": ["sqrt", "log2"],
    }
    rf_search = GridSearchCV(
        rf_pipeline, rf_param_grid, cv=cv,
        scoring="f1", n_jobs=-1, verbose=1
    )
    rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_
    print(f"Best RF params: {rf_search.best_params_}")
    print(f"Best RF CV F1:  {rf_search.best_score_:.4f}")
 
    # ── MODEL B: Logistic Regression ────────────
    print("\nTraining Logistic Regression (with GridSearchCV)...")
    lr_param_grid = {
        "C": [0.01, 0.1, 1.0, 10.0],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"],
    }
    lr_model = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42
    )
    lr_search = GridSearchCV(
        lr_model, lr_param_grid, cv=cv, scoring="f1", n_jobs=-1
    )
    lr_search.fit(X_train, y_train)
    best_lr = lr_search.best_estimator_
    print(f"Best LR params: {lr_search.best_params_}")
    print(f"Best LR CV F1:  {lr_search.best_score_:.4f}")
 
    # ── Save models + test split (for evaluate.py) ──
    joblib.dump(best_rf, "models/rf_model.pkl")
    joblib.dump(best_lr, "models/lr_model.pkl")
    joblib.dump(feature_cols, "models/feature_cols.pkl")
    X_test.to_parquet("data/X_test.parquet", index=False)
    y_test.to_frame().to_parquet("data/y_test.parquet", index=False)
 
    print("\nModels saved to models/ ✓")
    print("Test split saved to data/ for evaluate.py ✓")
 
 
if __name__ == "__main__":
    main()
