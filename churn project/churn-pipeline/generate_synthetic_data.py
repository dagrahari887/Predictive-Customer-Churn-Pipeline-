"""
generate_synthetic_data.py
---------------------------------
Populates the churn_db MySQL database with realistic synthetic data:
  - 1000 customers
  - ~5 transactions per active customer, fewer for "churned" customers
  - ~1.5 support tickets per customer
 
Run this AFTER schema.sql has created the tables.
 
Usage:
    pip install faker mysql-connector-python
    python generate_synthetic_data.py
"""
 
import random
from datetime import datetime, timedelta
import mysql.connector
from faker import Faker
 
fake = Faker("en_IN")  # Indian locale for realistic names/cities
random.seed(42)
Faker.seed(42)
 
# ─────────────────────────────────────────────
#  CONFIG — edit these to match your MySQL setup
# ─────────────────────────────────────────────
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "1234",   # <-- CHANGE THIS
    "database": "churn_db",
}
 
NUM_CUSTOMERS = 1000
TODAY = datetime.now()
 
CITIES = ["Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai",
          "Pune", "Kolkata", "Ahmedabad", "Jaipur", "Lucknow",
          "Patna", "Indore", "Surat", "Nagpur"]
 
SUBSCRIPTION_PLANS = ["Free", "Silver", "Gold", "Premium"]
PRODUCT_CATEGORIES = ["Electronics", "Fashion", "FMCG", "Home & Kitchen",
                       "Books", "Beauty", "Sports", "Groceries"]
PAYMENT_METHODS = ["UPI", "Card", "COD", "Wallet", "NetBanking"]
ISSUE_TYPES = ["Refund", "Delivery Delay", "App Bug", "Payment Failed",
               "Wrong Item", "Product Quality", "Account Access"]
RESOLUTION_STATUSES = ["Resolved", "Pending", "Escalated"]
 
 
def get_connection():
    return mysql.connector.connect(**DB_CONFIG)
 
 
def generate_customers(cursor, n=NUM_CUSTOMERS):
    """
    Creates n customers. ~15% are intentionally designed to be 'churn-prone'
    (older signup, no recent activity) so the dataset has a realistic
    imbalanced churn distribution after the 60-day rule is applied later.
    """
    customer_ids = []
    insert_query = """
        INSERT INTO customers
            (age, gender, city, signup_date, subscription_plan, is_churned)
        VALUES (%s, %s, %s, %s, %s, %s)
    """
 
    for i in range(n):
        age = random.randint(18, 65)
        gender = random.choice(["Male", "Female", "Other"])
        city = random.choice(CITIES)
        # Signup spread over the last 3 years
        signup_date = fake.date_between(start_date="-3y", end_date="-30d")
        plan = random.choices(
            SUBSCRIPTION_PLANS, weights=[0.35, 0.30, 0.20, 0.15]
        )[0]
        # is_churned placeholder — will be recalculated properly
        # in feature_engineering.py based on actual recency. Here we
        # just store 0; the real label comes from transaction behavior.
        is_churned_placeholder = 0
 
        cursor.execute(insert_query, (
            age, gender, city, signup_date, plan, is_churned_placeholder
        ))
        customer_ids.append(cursor.lastrowid)
 
    print(f"Inserted {len(customer_ids)} customers.")
    return customer_ids
 
 
def generate_transactions(cursor, customer_ids):
    """
    For each customer, generates a random transaction history.
    ~15% of customers are made 'dormant' (no purchase in 60+ days)
    to simulate realistic churn behavior.
    """
    insert_query = """
        INSERT INTO transactions
            (customer_id, txn_date, amount, product_category, payment_method)
        VALUES (%s, %s, %s, %s, %s)
    """
 
    total_txns = 0
    for cid in customer_ids:
        # Decide if this customer is "dormant" (likely to churn)
        is_dormant = random.random() < 0.15
 
        num_txns = random.randint(1, 3) if is_dormant else random.randint(3, 15)
 
        for _ in range(num_txns):
            if is_dormant:
                # Last purchase was 70-300 days ago — pushes recency > 60
                days_ago = random.randint(70, 300)
            else:
                # Active customers purchased recently
                days_ago = random.randint(0, 55)
 
            txn_date = TODAY - timedelta(days=days_ago)
            amount = round(random.uniform(150, 8000), 2)
            category = random.choice(PRODUCT_CATEGORIES)
            payment = random.choice(PAYMENT_METHODS)
 
            cursor.execute(insert_query, (
                cid, txn_date, amount, category, payment
            ))
            total_txns += 1
 
    print(f"Inserted {total_txns} transactions.")
 
 
def generate_support_logs(cursor, customer_ids):
    """
    Generates support tickets for a subset of customers.
    Not every customer has raised a ticket (realistic LEFT JOIN scenario).
    """
    insert_query = """
        INSERT INTO support_logs
            (customer_id, ticket_date, issue_type, resolution_status, satisfaction_score)
        VALUES (%s, %s, %s, %s, %s)
    """
 
    total_logs = 0
    for cid in customer_ids:
        # 60% of customers have raised at least one ticket
        if random.random() > 0.6:
            continue
 
        num_tickets = random.randint(1, 4)
        for _ in range(num_tickets):
            days_ago = random.randint(0, 365)
            ticket_date = TODAY - timedelta(days=days_ago)
            issue = random.choice(ISSUE_TYPES)
            status = random.choices(
                RESOLUTION_STATUSES, weights=[0.7, 0.2, 0.1]
            )[0]
            satisfaction = random.randint(1, 5) if status == "Resolved" else None
 
            cursor.execute(insert_query, (
                cid, ticket_date, issue, status, satisfaction
            ))
            total_logs += 1
 
    print(f"Inserted {total_logs} support tickets.")
 
 
def main():
    print("Connecting to MySQL...")
    conn = get_connection()
    cursor = conn.cursor()
 
    print("\nGenerating customers...")
    customer_ids = generate_customers(cursor)
    conn.commit()
 
    print("\nGenerating transactions...")
    generate_transactions(cursor, customer_ids)
    conn.commit()
 
    print("\nGenerating support logs...")
    generate_support_logs(cursor, customer_ids)
    conn.commit()
 
    cursor.close()
    conn.close()
    print("\nDone. Synthetic data loaded into churn_db ✓")
 
 
if __name__ == "__main__":
    main()  