# Predictive Customer Churn Pipeline

A production-style machine learning system and standalone analytical tool designed to predict retail customer churn based on transaction history, support logs, and demographic data.

---

## Overview

This repository contains two main entry points for analyzing and predicting customer churn[cite: 1]:
1. **Production Pipeline (MySQL + FastAPI):** A relational database pipeline that extracts raw customer logs, cleans and transforms features, trains/evaluates machine learning models, and exposes predictions via a live REST API[cite: 1].
2. **Standalone Excel Analysis Tool (`predict_from_excel.py`):** An ad-hoc pipeline that auto-detects date, customer ID, and transaction amount columns from unformatted retail spreadsheets, dynamically calculates category-aware churn thresholds, and trains a fresh model on the fly[cite: 1].

---

## Features & Highlights

* **Automated Data Cleaning & Outlier Management:** Features median imputation for missing values and IQR clipping (1st–99th percentiles) to neutralize extreme numerical outliers[cite: 1].
* **Feature Engineering & Leakage Avoidance:** Implements Recency, Frequency, Monetary (RFM), and Average Ticket Size aggregation[cite: 1]. To prevent data leakage, the direct target generator (`recency`) is isolated strictly to label creation and excluded from feature input sets[cite: 1].
* **Class Imbalance Mitigation:** Combines Synthetic Minority Over-sampling Technique (**SMOTE**) within cross-validation folds and `class_weight='balanced'` cost adjustment to target high recall on minority churners[cite: 1].
* **Flexible Column Auto-Detection:** Built-in two-pass heuristic parser (header keyword matching and data-type content inspection) to digest disparate client spreadsheets without pre-formatting[cite: 1].
* **Production Deployment:** Lightweight FastAPI microservice serving real-time risk classification (`HIGH`, `MEDIUM`, `LOW`) and recommended interventions with validation via Pydantic[cite: 1].

---

## Project Structure

```text
.
├── schema.sql                # MySQL relational database setup script
├── etl_pipeline.py           # SQL extraction and automated data cleaning
├── feature_engineering.py    # RFM metrics, target creation, and dataset export
├── model_training.py         # Pipeline setup, SMOTE, Hyperparameter tuning (GridSearchCV)
├── evaluate.py               # Model evaluation, matrix generation, and feature importance
├── app.py                    # FastAPI application serving live endpoint predictions
├── predict_from_excel.py     # Standalone auto-detect pipeline for raw Excel files
├── models/                   # Serialized model artifacts (.pkl files and scalers)
└── README.md
```[cite: 1]

---+--------------------+       +--------------------+       +----------------------+
|     customers      |       |    transactions    |       |     support_logs     |
+--------------------+       +--------------------+       +----------------------+
| customer_id (PK)   |<-----| txn_id (PK)        |       | log_id (PK)          |
| age                |       | customer_id (FK)   |   /---| customer_id (FK)     |
| gender             |       | txn_date           |  /    | ticket_date          |
| city               |       | amount             | /     | issue_type           |
| signup_date        |       | product_category   |/      | resolution_status    |
| subscription_plan  |       | payment_method     |       | satisfaction_score   |
| is_churned         |       +--------------------+       +----------------------+
+--------------------+---
+-------------------+
|  MySQL Database   |
+-------------------+
|
v (LEFT JOIN Extraction)
+-------------------+
|  etl_pipeline.py  |  --> [ Cleans outliers & imputes missing data ]
+-------------------+
|
v (Output: cleaned_raw.parquet)
+------------------------+
| feature_engineering.py |  --> [ Calculates RFM & flags churn (recency > 60 days) ]
+------------------------+
|
v (Output: features.parquet)
+--------------------+
| model_training.py  |  --> [ Applies SMOTE & trains Random Forest via GridSearchCV ]
+--------------------+
|
v (Saves: models/rf_model.pkl & models/scaler.pkl)
+--------------------+         +--------------------+
|    evaluate.py     |         |   app.py (FastAPI) |
| (Performance metrics)       | (Live Predictions) |
+--------------------+         +--------------------+---

## Model Evaluation Results

Models were tuned using 5-fold cross-validation with an emphasis on **Recall** (minimizing False Negatives to prevent undetected churn)[cite: 1]:

| Metric | Random Forest (Selected) | Logistic Regression |
| :--- | :--- | :--- |
| **Recall (Churned)** | **90.9%** | 97.0% |
| **Precision (Churned)** | **85.7%** | 51.2% |
| **F1-Score (Churned)** | **88.2%** | 67.0% |
| **Overall Accuracy** | **96.0%** | 84.3% |
| **ROC-AUC** | **0.9933** | 0.9606 |

*Table data sourced from model evaluation metrics[cite: 1].*

### Feature Importance (Top Churn Drivers)
1. `num_support_tickets` (**25.9%**): High frequency of logged customer service complaints[cite: 1].
2. `frequency` (**23.7%**): Overall volume of purchase interactions over lifetime[cite: 1].
3. `monetary` (**16.0%**): Total aggregate spending value[cite: 1].

---

## Setup & Running Instructions

### Prerequisites
* Python 3.9+
* MySQL Server (optional for API, required for full DB ETL)[cite: 1]

### Installation
1. Clone the repository and install required packages:
   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn fastapi uvicorn joblib openpyxl pyarrow

Code snippet
## System Architecture & Data Flow

## Database Architecture

The MySQL data layer (`churn_db`) normalizes transaction and support behavior across three core tables linked via Foreign Keys[cite: 1]:
