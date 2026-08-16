import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

print("=== PHASES 1 & 2: DATA INGESTION & FEATURE ENGINEERING ===")

# 1. Simulate Raw Data extraction from MySQL
np.random.seed(42)
n_customers = 2000

raw_data = {
    'customer_id': range(1001, 1001 + n_customers),
    'age': np.random.randint(18, 75, size=n_customers),
    'gender': np.random.choice(['M', 'F'], size=n_customers),
    'total_orders': np.random.randint(1, 50, size=n_customers),
    'total_spend': np.random.uniform(20, 2000, size=n_customers),
    'days_since_last_purchase': np.random.randint(1, 120, size=n_customers),
    'support_tickets_opened': np.random.randint(0, 10, size=n_customers),
    'membership_tier': np.random.choice(['Bronze', 'Silver', 'Gold'], size=n_customers, p=[0.5, 0.3, 0.2])
}

df_raw = pd.DataFrame(raw_data)

# 2. Inject Real-World Dependencies (Making the data logical)
# High support tickets and high recency (days since last purchase) = much higher chance of churn
churn_probability = (
    (df_raw['days_since_last_purchase'] * 0.4) + 
    (df_raw['support_tickets_opened'] * 5.0) - 
    (df_raw['total_orders'] * 0.2)
)
# Normalize to 0-1 and create an imbalanced target variable (approx 15% churn rate)
churn_probability = (churn_probability - churn_probability.min()) / (churn_probability.max() - churn_probability.min())
df_raw['churn'] = (churn_probability > 0.65).astype(int)

# 3. Advanced Feature Engineering
df_features = df_raw.copy()
df_features['avg_ticket_size'] = df_features['total_spend'] / df_features['total_orders']

# Categorical Encoding (One-Hot Encoding)
df_features = pd.get_dummies(df_features, columns=['gender', 'membership_tier'], drop_first=True)

# Drop ID column before scaling/modeling
X = df_features.drop(columns=['customer_id', 'churn'])
y = df_features['churn']

print(f"Dataset Shape: {X.shape} | Churn Rate: {y.mean()*100:.2f}% (Realistic Class Imbalance)")



#Phase 3 & 4: Model Training, Imbalance Handling, & Evaluation

print("\n=== PHASES 3 & 4: MODELING & DEVELOPER EVALUATION ===")

# Train-Test Split (70% Training, 30% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Feature Scaling (Crucial for distance-based comparisons/interpretability)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest with Class Weight Balancing to tackle imbalance
model = RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=42)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Evaluation Metrics
print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

# Extract Feature Importance
importances = model.feature_importances_
feature_ranking = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
print("\n--- Top Core Feature Drivers of Churn ---")
print(feature_ranking.head(3))



# Phase 5: Production Deployment (The Developer Edge)


from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# Initialize FastAPI application
app = FastAPI(title="Production Customer Churn Prediction API")

# Define the expected JSON payload format using Pydantic
class CustomerData(BaseModel):
    age: int
    total_orders: int
    total_spend: float
    days_since_last_purchase: int
    support_tickets_opened: int
    avg_ticket_size: float
    gender_M: int
    membership_tier_Gold: int
    membership_tier_Silver: int

@app.post("/predict")
def predict_churn(customer: CustomerData):
    # 1. Convert JSON input to standard format expected by model
    input_data = [[
        customer.age, customer.total_orders, customer.total_spend,
        customer.days_since_last_purchase, customer.support_tickets_opened,
        customer.avg_ticket_size, customer.gender_M, 
        customer.membership_tier_Gold, customer.membership_tier_Silver
    ]]
    
    # 2. Apply the pre-configured data scaling transforms
    input_scaled = scaler.transform(input_data)
    
    # 3. Calculate probability percentage
    probability = model.predict_proba(input_scaled)[0][1]
    prediction = int(probability > 0.5)
    
    # 4. Return API response
    return {
        "churn_prediction": prediction,
        "churn_probability_percentage": round(float(probability) * 100, 2),
        "risk_status": "High Risk - Trigger Retention Offer" if prediction == 1 else "Low Risk"
    }

# Entry point to execute the server script
if __name__ == "__main__":
    print("\n=== PHASE 5: RUNNING LIVE FASTAPI PRODUCTION SERVER ===")
    print("API documentation is live at http://127.0.0.1:9050/docs")
    # This fires up the local server engine
    uvicorn.run(app, host="127.0.0.1", port=9050)