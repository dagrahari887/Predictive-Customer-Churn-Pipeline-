"""
evaluate.py
---------------------------------
Loads trained models + held-out test set, runs full evaluation:
classification report, confusion matrix, ROC-AUC, feature importance.
Saves plots to outputs/.
 
Usage:
    python evaluate.py
"""
 
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # avoids GUI backend issues when run headless
import matplotlib.pyplot as plt
import joblib
import os
 
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)
 
os.makedirs("outputs", exist_ok=True)
 
 
def evaluate_model(model, X_test, y_test, model_name: str):
    """Full evaluation suite for a trained classifier."""
 
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
 
    # ── 1. Classification Report ───────────────
    print(f"\n{'='*50}")
    print(f" {model_name} — Classification Report")
    print(f"{'='*50}")
    print(classification_report(
        y_test, y_pred, target_names=["Active", "Churned"], digits=4
    ))
 
    # ── 2. ROC-AUC Score ───────────────────────
    auc = roc_auc_score(y_test, y_pred_prob)
    print(f"ROC-AUC Score: {auc:.4f}")
 
    # ── 3. Confusion Matrix + ROC plots ────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{model_name} Evaluation", fontsize=14, fontweight="bold")
 
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Active", "Churned"])
    disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title("Confusion Matrix")
 
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    axes[1].plot(fpr, tpr, color="#6c63ff", lw=2, label=f"AUC = {auc:.3f}")
    axes[1].plot([0, 1], [0, 1], "--", color="gray", label="Random Baseline")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve")
    axes[1].legend()
 
    plt.tight_layout()
    safe_name = model_name.replace(" ", "_")
    plt.savefig(f"outputs/{safe_name}_eval.png", dpi=150)
    plt.close()
    print(f"Saved plot: outputs/{safe_name}_eval.png")
 
    return {"model": model_name, "auc": auc, "confusion_matrix": cm}
 
 
def feature_importance_report(rf_model, feature_cols: list, top_n: int = 10):
    """Extract and plot feature importances from Random Forest."""
    clf = rf_model.named_steps["clf"]
    importances = clf.feature_importances_
 
    fi_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances
    }).sort_values("importance", ascending=False)
 
    print("\n── Top Feature Importances ──────────────")
    print(fi_df.head(top_n).to_string(index=False))
 
    plt.figure(figsize=(8, 5))
    plt.barh(
        fi_df["feature"].head(top_n)[::-1],
        fi_df["importance"].head(top_n)[::-1],
        color="#6c63ff", alpha=0.85
    )
    plt.title("Feature Importances (Random Forest)")
    plt.xlabel("Mean Decrease in Impurity")
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png", dpi=150)
    plt.close()
    print("Saved plot: outputs/feature_importance.png")
 
    return fi_df
 
 
if __name__ == "__main__":
    print("Loading models and test data...")
    best_rf = joblib.load("models/rf_model.pkl")
    best_lr = joblib.load("models/lr_model.pkl")
    feature_cols = joblib.load("models/feature_cols.pkl")
 
    X_test = pd.read_parquet("data/X_test.parquet")
    y_test = pd.read_parquet("data/y_test.parquet")["is_churned"]
 
    rf_results = evaluate_model(best_rf, X_test, y_test, "Random Forest")
    lr_results = evaluate_model(best_lr, X_test, y_test, "Logistic Regression")
 
    fi_df = feature_importance_report(best_rf, feature_cols)