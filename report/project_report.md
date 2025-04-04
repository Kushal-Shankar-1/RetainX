
# CS6220 - Data Mining Techniques
## Final Project Report

**Project Title:** RetainX: Customer Churn Prediction for Subscription-Based Services  
**Team:** Colorado Blue Spruce (Team 7)  

**Team Members:**  
- Kushal Shankar – shankar.ku@northeastern.edu  
- Ashmitha Appandaraju – appandaraju.a@northeastern.edu  
- Tej Sidda – sidda.t@northeastern.edu  
- Ronak Vadhaiya – vadhaiya.r@northeastern.edu  

---

## 1. Abstract

Customer churn poses a critical threat to subscription-driven businesses. Companies often invest heavily in acquiring new customers, making retention strategies more crucial than ever. This project, **RetainX**, leverages machine learning to predict customer churn using the **Telco Customer Churn** dataset from IBM (via Kaggle). We conduct extensive data cleaning, feature engineering, baseline and advanced modeling, and SHAP-based explainability to derive business insights. The models developed can enable proactive churn mitigation and drive customer-centric business decisions.

---

## 2. Introduction

### i. Problem Statement
Subscription churn results in direct revenue loss and increased customer acquisition costs. In industries like telecom and SaaS, churn rates range from 10–30% annually, drastically affecting profits and scalability. Predicting churn proactively allows businesses to design customer-specific retention strategies and reduce attrition.

### ii. Importance of Solving This Problem
Retaining customers is significantly more cost-effective than acquiring new ones. Even a 1% reduction in churn can yield millions in savings for large enterprises. With the ability to identify churn risk early, companies can customize interventions like discounts, improved service, or engagement programs.

### iii. Background and Literature Survey
Prior research has utilized statistical and machine learning models to detect churn patterns based on demographics, contract types, and usage. Algorithms like Logistic Regression, Decision Trees, Random Forests, and XGBoost have shown high efficacy when paired with proper preprocessing and imbalance mitigation. Interpretability methods like SHAP are also crucial in real-world business deployment.

---

## 3. Methodology

### Data Source
- **Dataset:** [Telco Customer Churn – Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Rows:** 7,043 customers  
- **Target Variable:** `Churn` (Yes/No)
- **Features:** Demographics, subscription services, tenure, billing and charges, contract details.

### Data Cleaning & EDA
- Converted `TotalCharges` to numeric and handled missing values using the mean.
- Visualized churn patterns across services, contract types, and tenure using bar charts and heatmaps.
- Identified high churn among month-to-month contracts and electronic check users.
- Saved correlation heatmap as a PNG for reporting.

### Feature Engineering
- Created `tenure_group` buckets (e.g., 0–12, 12–24 months, etc.)
- Engineered `avg_monthly_charge` = `TotalCharges` / `tenure`
- Applied one-hot encoding to categorical features (e.g., InternetService, PaymentMethod)
- Standardized numerical features (optional) and dropped redundant columns

### Handling Imbalance
- Applied **SMOTE** oversampling on the training set after the train-test split to balance churn classes.

### Modeling Strategy
- **Baseline:** Logistic Regression
- **Advanced Models:** Random Forest, Decision Tree, and XGBoost
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1 Score, ROC-AUC
- **Hyperparameter Tuning:** GridSearchCV for XGBoost

### Explainability
- Used SHAP values to evaluate feature importance for interpretability.
- Visualized feature impact and global summary to support business strategies.

---

## 4. Code Explanation

All implementation is in Python, structured into modular and reproducible Jupyter Notebooks:

### Notebooks Overview
- `data_cleaning_eda.ipynb`: Cleaning, imputation, EDA, correlation.
- `data_preprocessing.ipynb`: Feature encoding, numeric conversions, train-test split.
- `feature_engineering_baseline_models.ipynb`: Feature creation + baseline model evaluation.
- `advanced_modeling_and_evaluation.ipynb`: SMOTE, hyperparameter tuning (GridSearchCV), XGBoost.
- `interpretation_and_business_insights.ipynb`: SHAP values, insights, and business actionability.

### Key Libraries Used
- `pandas`, `numpy`, `matplotlib`, `seaborn` – Data analysis and visualization
- `scikit-learn` – ML models, preprocessing, evaluation metrics
- `imbalanced-learn` – SMOTE
- `xgboost` – Gradient boosting classifier
- `shap` – Model explainability

### Folder Structure
```
project_root/
│
├── data/                        # Raw and preprocessed CSVs
├── notebooks/                  # Jupyter notebooks for each phase
├── models/                     # Saved model pickle files (.pkl)
├── results/                    # Saved visualizations and PDF exports
├── report/                     # Final project report
├── presentation/               # Project slides
├── requirements.txt            # Python dependencies
├── README.md                   # Project overview
└── team_testing_checklist.md   # Notebook testing records
```

---

## 5. Results

| Model              | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.802   | 0.720     | 0.600  | 0.654    | 0.837   |
| Random Forest       | 0.832   | 0.750     | 0.678  | 0.712    | 0.862   |
| XGBoost (Tuned)     | **0.847** | **0.770** | **0.700** | **0.733** | **0.878** |

- SMOTE improved recall significantly by balancing the class distribution.
- XGBoost with GridSearch outperformed all others in terms of both ROC-AUC and F1.

---

## 6. Discussion

- **Key Churn Drivers Identified (via SHAP):**
  - Month-to-month contracts
  - High `MonthlyCharges` and low `tenure`
  - Electronic check payment method
  - Absence of internet services like security or backup

- **Business Interpretation:**
  - Customers on flexible plans with high charges are more likely to churn.
  - Bundled service offerings, tenure-based discounts, and switching payment methods may improve retention.

- **Visualization Highlights:**
  - Bar charts of churn across contract types and services.
  - Correlation heatmap linking churn with billing and service usage.
  - SHAP summary and bar plots showing global feature importance.

---

## 7. Future Work

- Include real-time churn prediction using live CRM data.
- Integrate NLP for churn signals from customer support conversations.
- Add time-series based survival analysis for churn prediction over time.

---

## 8. Conclusion

Our project successfully demonstrates the application of machine learning for churn prediction using a well-structured, end-to-end pipeline. With explainability at its core, **RetainX** offers actionable intelligence for customer retention, enabling businesses to target at-risk users early and reduce churn effectively.

---

## 9. References

- [Telco Customer Churn Dataset – Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Scikit-learn Documentation – https://scikit-learn.org/
- XGBoost Documentation – https://xgboost.readthedocs.io/
- SHAP Documentation – https://shap.readthedocs.io/
