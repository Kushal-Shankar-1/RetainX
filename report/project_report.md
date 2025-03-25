
# CS6220 - Data Mining Techniques Final Project Report  
## Project Title: RetainX — Customer Churn Prediction for Subscription-Based Services  
**Team 7: Colorado Blue Spruce**  
- Kushal Shankar  
- Ashmitha Appandaraju  
- Tej Sidda  
- Ronak Vadhaiya  

---

## 1. Abstract
Customer churn is a major challenge for subscription-based businesses. In this project, we built RetainX, a predictive analytics solution using the Telco Customer Churn dataset. We performed exploratory data analysis, preprocessing, feature engineering, and built baseline and advanced models (Random Forest and XGBoost). SHAP explainability was used to interpret model predictions and derive business recommendations.

---

## 2. Introduction  
### i. Problem Statement  
Churn prediction enables proactive retention strategies and reduces revenue loss.  
### ii. Importance  
Reducing churn by even 1% leads to significant cost savings for businesses.  
### iii. Background  
We used proven machine learning techniques and explainability methods to build interpretable churn prediction models.

---

## 3. Methodology  
- **Data Preprocessing**: Handled nulls, encoded categorical features, scaled numerical columns.  
- **EDA**: Visualized churn patterns across contract types, payment methods, and service usage.  
- **Feature Engineering**: Created tenure groups and average monthly charge features.  
- **Modeling**:  
  - Baseline: Logistic Regression, Decision Tree  
  - Advanced: Random Forest, XGBoost with hyperparameter tuning  
- **Class imbalance**: Addressed using SMOTE.  
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC.  
- **Interpretation**: Used SHAP to explain feature importance.

---

## 4. Results  
| Model             | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.80     | 0.68      | 0.61   | 0.64     | 0.72    |
| Decision Tree     | 0.81     | 0.69      | 0.63   | 0.66     | 0.74    |
| Random Forest     | 0.85     | 0.75      | 0.69   | 0.72     | 0.80    |
| XGBoost (best)   | 0.87     | 0.78      | 0.72   | 0.75     | 0.82    |

---

## 5. Discussion  
- High churn observed among month-to-month contracts and electronic check payment customers.  
- Higher monthly charges correlate with churn risk.  
- SHAP confirmed contract type, tenure, and billing method as key features.

---

## 6. Business Recommendations  
- Target month-to-month customers with loyalty incentives.  
- Encourage long-term contracts and auto-pay billing.  
- Offer discounts to high-monthly-charge, short-tenure customers.

---

## 7. Conclusion  
RetainX successfully predicts churn and helps design targeted retention strategies for subscription-based businesses.

---

## 8. Future Work  
- Real-time churn prediction dashboard  
- Incorporate customer sentiment analysis  
- Expand to other industries beyond telecom

---

## 9. References  
- Telco Customer Churn Dataset (Kaggle)  
- XGBoost and SHAP documentation  
- Academic literature on churn prediction models
