
# CS6220 - Data Mining Techniques Final Project Report
## Project Title: Customer Churn Prediction for Subscription Services  
**Project Team 7: Colorado Blue Spruce**  
Kushal Shankar, Ashmitha Appandaraju, Tej Sidda, Ronak Vadhaiya  

---

## 1. Abstract
This project focuses on predicting customer churn using the Telco Customer Churn dataset. We analyze key indicators of churn and build predictive models using data mining techniques. Our best model achieved strong predictive performance and offers actionable insights for subscription-based businesses.

---

## 2. Introduction
### i. Problem Statement
Customer churn impacts recurring revenue businesses. Predicting churn allows proactive retention strategies.

### ii. Importance
Reducing churn by even 1% can result in significant revenue savings.

### iii. Literature Review
Previous research has applied machine learning models like logistic regression, decision trees, and ensemble methods. Interpretability and business alignment remain key challenges.

---

## 3. Methodology
### Data Preprocessing
- Converted `TotalCharges` to numeric, handled missing values.
- Encoded categorical variables (binary: LabelEncoder; multi-category: One-hot encoding).
- Scaled numerical features.

### Exploratory Data Analysis
- Churn visualized against contract types, payment methods, tenure, and charges.
- Correlation heatmaps highlighted feature relationships.

### Feature Engineering
- Created tenure groups.
- Derived average monthly charge features.

### Models Used
- Baseline: Logistic Regression, Decision Tree
- Advanced: Random Forest, XGBoost (with hyperparameter tuning)
- Handled class imbalance with SMOTE.

### Evaluation Metrics
Accuracy, Precision, Recall, F1-Score, ROC-AUC

---

## 4. Results
- Logistic Regression baseline: Accuracy ~xx%, ROC-AUC ~xx%  
- Random Forest: Accuracy ~xx%, ROC-AUC ~xx%  
- XGBoost (best model): Accuracy ~xx%, ROC-AUC ~xx%  

| Model            | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|------------------|----------|-----------|--------|----------|---------|
| Logistic Regression |        |           |        |          |         |
| Decision Tree   |          |           |        |          |         |
| Random Forest   |          |           |        |          |         |
| XGBoost         |          |           |        |          |         |

---

## 5. Discussion
- Key drivers: Contract type, tenure, monthly charges.
- SHAP analysis confirmed business intuition.
- Observed churn patterns aligned with contract flexibility and payment methods.

---

## 6. Business Recommendations
- Target month-to-month contract customers with loyalty incentives.
- Encourage long-term plans through discounts.
- Improve experiences for fiber optic users with high churn rates.
- Promote automatic payment methods over electronic checks.

---

## 7. Conclusion
Our predictive models can help businesses retain customers through data-driven strategies. The best models were interpretable and actionable.

---

## 8. Future Work
- Integrating real-time churn prediction pipelines.
- Including sentiment analysis from customer feedback channels.
- Extending models for broader subscription-based industries.

---

## 9. References
- Telco Customer Churn Dataset (Kaggle): https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- Relevant literature and business whitepapers (to be filled).

