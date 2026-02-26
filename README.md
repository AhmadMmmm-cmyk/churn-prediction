# 📉 Customer Churn Prediction & Model Drift Handling

## 📌 Executive Summary
This project develops a high-precision machine learning system to identify at-risk subscribers in a service-based business. Going beyond standard classification, this project actively diagnoses **Target and Concept Drift** over a 24-period timeline. 

By utilizing a **Random Forest** model with a **Rolling 12-Month** time-series cross-validation strategy, the final decision threshold was mathematically optimized to **6%**. This proactive approach balances Customer Lifetime Value (LTV) against promotional costs, minimizing projected revenue leakage from ~$15,000 to ~$4,000 per period.

---

## 🔍 Exploratory Data Analysis & Drift Diagnostics
To build a reliable model, we first had to diagnose the health of the historical data and identify shifts in customer behavior.

* **Target Drift:** The churn rate steadily climbed from ~20% in Period 1 to >55% in Period 24, indicating a deteriorating external business environment.
* **Covariate Stability:** Despite the rising churn, the distribution of monthly billing (`Montant_Mensuel`) remained stable, proving that price hikes were *not* the cause of attrition.
* **Informative Missingness (MNAR):** We discovered that missing data in Customer Support logs (`Support_Appels`) was a "smoking gun." Customers with missing support data had a **55% churn rate** compared to 35% for those with complete records. We engineered this missingness into a highly predictive feature.

---

## ⚙️ Model Selection & Backtesting Strategy
Because the data was shifting chronologically, a standard train-test split would have resulted in a model that quickly decayed in production.

* **Baseline Performance Decay:** Our initial Logistic Regression model showed clear **Concept Drift**, with its AUC dropping from ~0.90 down to ~0.84 in later periods as customer behavior evolved.
* **Time-Series Cross-Validation:** To combat this, we implemented a custom backtesting framework to simulate real-world production. We evaluated an *Expanding Window* (all history) against various *Rolling Windows*.
* **The Winner:** The **Rolling_12** strategy was selected. It provides a full year of seasonal data (maximizing stability) while remaining agile enough to "forget" outdated patterns.

### Algorithm Comparison (Using Expanding Window)
* **Logistic Regression (Baseline):** AUC = 0.8681
* **Random Forest (Challenger):** **AUC = 0.8775** (+0.94% improvement)

The Random Forest was selected for production due to its superior ability to capture non-linear, complex churn interactions.

---

## 💰 Business Impact & Threshold Optimization
Machine learning probabilities are only useful if they drive profitable business decisions. We built a Cost-Benefit matrix to optimize the intervention threshold:

* **Cost of False Negative (Missed Churner):** $500 (Lost LTV)
* **Cost of False Positive (Wasted Promo):** $50 (Cost of Retention Campaign)

**Result:** Because losing a customer is 10x more expensive than attempting to save them, the cost curve revealed that the optimal decision threshold is **0.06 (6%)**. Triggering retention actions for any customer with a $\ge$ 6% probability of churning captures the maximum number of high-value churners while keeping promotional costs strictly controlled.

---

## 🛠️ Tech Stack
* **Language:** Python
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn` (Pipelines, ColumnTransformer, SimpleImputer, RandomForestClassifier)
* **Evaluation & Visualization:** `matplotlib`, `seaborn`, `roc_auc_score`

---

## 🚀 How to Run Locally
1. Clone the repository:
   ```bash
   git clone [https://github.com/AhmadMmmm-cmyk/churn-prediction.git](https://github.com/AhmadMmmm-cmyk/churn-prediction.git)