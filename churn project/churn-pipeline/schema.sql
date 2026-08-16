-- - Run this in your MySQL 8.x instance
CREATE DATABASE IF NOT EXISTS churn_db;
USE churn_db;

CREATE TABLE customers (
    customer_id      INT AUTO_INCREMENT PRIMARY KEY,
    age              INT,
    gender           VARCHAR(10),
    city             VARCHAR(100),
    signup_date      DATE,
    subscription_plan VARCHAR(20),
    is_churned       TINYINT(1) DEFAULT 0
);

CREATE TABLE transactions (
    txn_id           INT AUTO_INCREMENT PRIMARY KEY,
    customer_id      INT,
    txn_date         DATETIME,
    amount           DECIMAL(10, 2),
    product_category VARCHAR(50),
    payment_method   VARCHAR(30),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE support_logs (
    log_id              INT AUTO_INCREMENT PRIMARY KEY,
    customer_id         INT,
    ticket_date         DATETIME,
    issue_type          VARCHAR(50),
    resolution_status   VARCHAR(20),
    satisfaction_score  TINYINT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);