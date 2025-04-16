
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

In today’s competitive subscription economy, customer churn presents a persistent and costly challenge across industries such as telecommunications, SaaS, and streaming services. Businesses invest substantial resources to acquire customers, yet retaining them is far more cost-effective and essential for sustainable growth. This project introduces **RetainX**, a machine learning-powered solution developed to proactively predict customer churn and proactively enable data-driven retention strategies. Using the **Telco Customer Churn** dataset from IBM (via Kaggle), we conducted extensive data preprocessing, exploratory analysis, and feature engineering to uncover churn patterns. We implemented both baseline models (Logistic Regression, Decision Tree) and advanced classifiers (Random Forest, XGBoost), with XGBoost achieving the highest predictive performance. To ensure interpretability, we incorporated **SHAP** explainability techniques that revealed key churn indicators such as contract type, monthly charges, and tenure. The insights derived from RetainX can directly support customer success teams and marketing units in reducing churn and maximizing customer lifetime value.

---

## 2. Introduction

### i. Problem Statement

Customer churn — the phenomenon where customers discontinue their subscription or service — is a critical problem for subscription-based businesses. Whether in telecommunications, SaaS, or OTT services, churn not only leads to direct revenue loss but also undermines long-term growth and customer lifetime value. In highly saturated markets, businesses face increasing challenges in maintaining loyalty as consumers have more choices and lower switching costs. Even a modest increase in churn can severely impact profitability. Predicting churn before it occurs is therefore essential for enabling timely and personalized retention strategies that help businesses stay competitive.

### ii. Importance of Solving This Problem

Retaining existing customers is significantly more cost-effective than acquiring new ones — studies estimate the cost to acquire a new customer can be 5 to 7 times higher. For enterprises that depend on recurring revenue, such as those in telecom or SaaS sectors, a reduction in churn by just 1–2% can lead to millions of dollars in savings annually. Beyond the financial benefits, reducing churn enhances brand trust, customer satisfaction, and market stability. By implementing machine learning models for churn prediction, businesses can transition from reactive to proactive engagement strategies — offering incentives, adjusting services, or providing support before customers decide to leave.

### iii. Background and Literature Review

Over the years, researchers and industry practitioners have explored various data mining and machine learning techniques to forecast customer churn. Early approaches used logistic regression and decision trees for their interpretability and low computational cost. More recently, ensemble-based methods such as Random Forests and gradient-boosting algorithms like XGBoost have gained prominence due to their improved accuracy and flexibility in handling high-dimensional datasets. These models benefit significantly from preprocessing steps such as feature engineering and balancing imbalanced datasets — a common issue in churn prediction where the number of churned customers is much smaller than non-churned ones. Furthermore, explainability techniques like **SHAP (SHapley Additive exPlanations)** have emerged as powerful tools to interpret model outputs and reveal the underlying drivers of churn, making the predictions more actionable for business stakeholders.

---

## 3. Methodology

The implementation of RetainX followed a well-defined machine learning workflow that included data preprocessing, exploratory data analysis (EDA), feature engineering, class imbalance handling, model training and evaluation, and model interpretability. Each stage was designed to ensure data quality, uncover meaningful patterns, build accurate predictive models, and extract business-relevant insights from the results.

### Data Source

We used the **Telco Customer Churn** dataset published by IBM and hosted on Kaggle. The dataset contains **7,043 customer records** with 21 features and one target label. It includes a diverse mix of variables ranging from customer demographics (e.g., gender, senior citizen status) to service-related attributes (e.g., type of internet service, tech support availability), and billing information (e.g., payment method, monthly charges). The target column, `Churn`, is binary, indicating whether a customer left the service (`Yes`) or not (`No`).

### Data Cleaning and Exploratory Data Analysis (EDA)

During preprocessing, we encountered a formatting issue in the `TotalCharges` column, which contained non-numeric characters. We converted it to a numeric format and imputed missing values using the column mean. We also encoded the target variable `Churn` into binary form, where "Yes" was mapped to 1 and "No" to 0.

Exploratory data analysis played a critical role in understanding the relationships between churn and various customer attributes. Visualizations such as bar plots and histograms revealed that customers with **month-to-month contracts**, **high monthly charges**, and **electronic check payments** had a significantly higher risk of churn. A correlation heatmap was generated to explore interactions among numerical features such as `tenure`, `MonthlyCharges`, and `TotalCharges`. These insights helped prioritize features and guided our feature engineering efforts.

### Feature Engineering

To improve model performance and provide more granular insights, we engineered several new features. The `tenure` column was segmented into categorical buckets — `0–12`, `12–24`, `24–48`, and `48+` months — to capture different stages in the customer lifecycle. This new feature, `tenure_group`, allowed the model to better learn churn patterns across customer durations. We also created a derived feature, `avg_monthly_charge`, by dividing `TotalCharges` by `tenure` (with 1 as a fallback to avoid division by zero). This metric helped capture the average amount a customer was paying per month, offering a behavioral perspective.

All categorical features — including both original and engineered ones — were encoded using one-hot encoding to convert them into a machine-readable binary format. Numerical features such as `MonthlyCharges` and `TotalCharges` were optionally scaled using standardization techniques, though tree-based models like Random Forest and XGBoost do not strictly require it.

### Handling Class Imbalance

As with most real-world churn datasets, the Telco dataset was imbalanced, with only about 26% of records labeled as churn. To address this issue, we applied **SMOTE (Synthetic Minority Over-sampling Technique)** on the training data after the train-test split. SMOTE generates synthetic examples for the minority class rather than duplicating existing records, which leads to a more balanced dataset and helps prevent model bias toward the majority class. This step proved essential in improving the recall and F1-score of our models without compromising precision.

### Model Training and Evaluation

We began with baseline models using **Logistic Regression** and **Decision Trees** to establish initial performance benchmarks. These models offered interpretability and ease of implementation but were limited in handling complex relationships. We then trained advanced models using **Random Forest** and **XGBoost**, both of which are ensemble learning methods capable of capturing nonlinear feature interactions.

To optimize XGBoost’s performance, we used **GridSearchCV** with 3-fold cross-validation, tuning hyperparameters such as `learning_rate`, `n_estimators`, and `max_depth`. The evaluation was performed using multiple metrics: **Accuracy**, **Precision**, **Recall**, **F1-Score**, and **ROC-AUC**. This multi-metric evaluation ensured that the model was not only accurate overall but also effective in identifying actual churn cases — a crucial requirement for any retention-focused application.

Among all models, the tuned XGBoost classifier performed the best, achieving an **ROC-AUC of 0.878**, along with the highest F1-score and recall. This validated our modeling and preprocessing choices, and gave us a strong basis for interpreting the results.

### Model Explainability with SHAP

While high performance was a key goal, model transparency was equally important for business adoption. We used **SHAP (SHapley Additive exPlanations)** to explain the contribution of each feature to model predictions. SHAP values were computed for the best-performing XGBoost model on the training set. The SHAP summary plot highlighted the top contributing features — with `Contract`, `tenure`, and `MonthlyCharges` emerging as the most influential drivers of churn.

These visualizations helped translate complex model logic into business-relevant language. For instance, the SHAP values showed that customers with month-to-month contracts and high monthly charges were more likely to churn, aligning with the patterns seen in EDA. The SHAP bar plot further provided a ranked list of features, supporting the formulation of targeted retention strategies. These explainability tools ensured that our model was not only powerful but also trustworthy and interpretable.

---

## 4. Code Explanation

The RetainX project was implemented entirely in **Python** using modular, reproducible, and well-documented **Jupyter Notebooks**. Our approach emphasized clarity, task separation, and collaborative development, with each notebook dedicated to a distinct phase in the data science pipeline.

The first notebook, `data_cleaning_eda.ipynb`, was responsible for loading the raw Telco dataset, performing initial preprocessing, and conducting exploratory data analysis. Tasks included converting the `TotalCharges` column to a numeric type, imputing missing values using the mean, and encoding the `Churn` label to binary. The notebook also produced correlation heatmaps and visualizations (bar plots, histograms, etc.) to uncover initial insights and relationships between churn and customer attributes.

The second notebook, `data_preprocessing.ipynb`, focused on transforming data into a machine learning-ready format. Categorical columns were one-hot encoded, and numeric columns were standardized where appropriate. A stratified train-test split was performed to preserve churn distribution, and the resulting data partitions were saved for reuse. This ensured consistency and separation of training logic across all models.

The third notebook, `feature_engineering_baseline_models.ipynb`, introduced domain-driven feature transformations such as the `tenure_group` and `avg_monthly_charge`. It also trained baseline models, including **Logistic Regression** and **Decision Tree Classifier**, allowing us to benchmark performance and establish reference points for future model improvements.

The fourth notebook, `advanced_modeling_and_evaluation.ipynb`, addressed class imbalance using **SMOTE** on the training set. It then trained two ensemble models: **Random Forest** and **XGBoost**, with XGBoost undergoing hyperparameter tuning via **GridSearchCV**. The notebook evaluated each model using metrics such as **Accuracy**, **Precision**, **Recall**, **F1-score**, and **ROC-AUC**. The best-performing model (tuned XGBoost) was serialized with `joblib` for later use in SHAP explainability.

The final notebook, `interpretation_and_business_insights.ipynb`, focused on model interpretability using **SHAP (SHapley Additive exPlanations)**. It loaded the saved XGBoost model and computed SHAP values to visualize feature contributions at both global and local levels. These explanations were vital in aligning our technical findings with actionable business strategies.

Throughout the project, we consistently used powerful libraries including `pandas` and `numpy` for data manipulation, `matplotlib` and `seaborn` for visualization, `scikit-learn` for modeling and metrics, `imbalanced-learn` for class balancing with SMOTE, `xgboost` for high-performance boosting, and `shap` for model explainability. The full environment is documented in `requirements.txt` to ensure reproducibility.

To maintain modularity and organization, we followed a clean project directory structure, as shown below:

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

This structure supported seamless handoffs across tasks, enabled version tracking of components, and ensured the project remained clean, scalable, and fully reproducible from start to finish.

---

## 5. Results

The table below summarizes the performance of all models evaluated on the test dataset using key classification metrics: Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

| Model              | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.802   | 0.720     | 0.600  | 0.654    | 0.837   |
| Random Forest       | 0.832   | 0.750     | 0.678  | 0.712    | 0.862   |
| XGBoost (Tuned)     | **0.847** | **0.770** | **0.700** | **0.733** | **0.878** |

The **Logistic Regression** model served as a solid baseline, achieving an accuracy of 80.2% and an ROC-AUC of 0.837. While it demonstrated decent precision (0.720), its relatively low recall (0.600) indicated that it missed a considerable number of churn cases. This limitation is critical in churn prediction, where identifying true churners is often more important than avoiding false positives.

The **Random Forest** model showed notable improvement across all metrics. With a recall of 0.678 and an ROC-AUC of 0.862, it performed significantly better in distinguishing churned customers from retained ones. Its higher F1-score (0.712) reflected a more balanced performance between precision and recall.

The best results were achieved using a **tuned XGBoost model**. After hyperparameter tuning with GridSearchCV and applying SMOTE to the training set, XGBoost achieved an accuracy of **84.7%**, a recall of **0.700**, and an **ROC-AUC of 0.878**, the highest among all models. Its F1-score of **0.733** indicated a strong balance between precision and recall, validating the effectiveness of the chosen pipeline.

The application of **SMOTE** played a pivotal role in improving recall across all models. Without addressing the class imbalance, models tended to favor the majority class (non-churn), leading to lower sensitivity toward churners. By synthetically augmenting the minority class during training, SMOTE ensured that the models could better recognize true churn cases without compromising overall accuracy.

Overall, the progression from baseline models to advanced tuned classifiers demonstrated clear performance gains, with each modeling choice — including feature engineering, imbalance correction, and hyperparameter optimization — contributing to better predictive accuracy and actionable outcomes.

---

## 6. Discussion

The results from our SHAP analysis and exploratory data visualizations revealed several important drivers of customer churn that align closely with domain expectations. Among the most influential predictors were contract type, monthly charges, tenure, and payment method. Specifically, customers subscribed to **month-to-month contracts** exhibited significantly higher churn rates compared to those with one-year or two-year contracts. This suggests that longer contractual commitments act as a natural retention mechanism, while the flexibility of monthly plans may signal lower customer loyalty or satisfaction.

Another key insight was the relationship between **monthly charges and tenure**. Customers with **high monthly charges** and **shorter tenures** were far more likely to churn, indicating early-stage dissatisfaction with cost or value. This finding is critical, as it suggests that churn risk is not only a function of service longevity but also of perceived affordability in the early months of engagement. Furthermore, customers who used **electronic check payments** — as opposed to automated bank transfers or credit cards — demonstrated higher churn rates, possibly due to weaker billing engagement or a higher likelihood of service disruption.

The SHAP summary and bar plots quantitatively confirmed these trends, with `Contract`, `MonthlyCharges`, and `tenure` emerging as the top contributors to the XGBoost model’s predictions. The explainability analysis gave us confidence that the model was learning from meaningful and interpretable patterns rather than noise.

From a business perspective, these insights can inform a range of targeted interventions. **Customers on flexible plans with high charges** represent a high-risk segment; offering them discounts, loyalty bonuses, or contract upgrades may significantly improve retention. Likewise, incentivizing early tenure customers with personalized engagement or onboarding support could mitigate first-year churn. Encouraging customers to switch from manual to automatic payment methods may also correlate with improved retention by reducing payment friction.

The visualizations generated throughout the project supported and contextualized these insights. **Bar charts** of churn by contract type and payment method, along with the **correlation heatmap** linking billing variables, helped uncover patterns before modeling even began. Post-modeling, the **SHAP plots** validated the statistical findings and translated model predictions into business language.

Together, the data-driven patterns, model explanations, and domain-relevant narratives form a comprehensive foundation for proactive churn management strategies.

---

## 7. Future Work

While RetainX successfully provides a robust and interpretable churn prediction pipeline using structured customer data, several areas offer potential for future enhancement. One promising direction is the **integration of real-time data pipelines**, enabling live churn prediction from customer relationship management (CRM) systems. This would allow businesses to monitor churn risk dynamically and intervene immediately with personalized retention strategies.

Additionally, incorporating **Natural Language Processing (NLP)** could further enrich the feature space. Unstructured data from customer support tickets, call transcripts, or social media interactions may contain early indicators of dissatisfaction or intent to cancel. Sentiment analysis, keyword tracking, or topic modeling on these text sources could complement existing churn predictors and improve model sensitivity.

Another valuable extension is the application of **survival analysis** to model customer tenure and churn probability over time. Unlike traditional classification models, survival models can estimate the expected time until churn, allowing for more precise targeting of retention efforts based on predicted churn windows. This would be particularly useful in contract-based industries or for tracking long-term customer lifecycles.

Collectively, these enhancements would transform RetainX from a static analytical tool into a real-time, multi-modal intelligence system for customer success and business growth.

---

## 8. Conclusion

This project successfully demonstrates how machine learning can be applied to solve a critical business challenge: predicting and mitigating customer churn. Through the development of **RetainX**, we implemented an end-to-end pipeline that began with careful data cleaning and exploratory analysis, progressed through feature engineering and class imbalance handling, and culminated in the deployment of high-performing, interpretable models.

By combining baseline models with advanced ensemble techniques like XGBoost and applying systematic hyperparameter tuning and SMOTE-based balancing, we were able to achieve strong predictive performance — with our best model reaching an ROC-AUC of 0.878. Importantly, we went beyond accuracy by incorporating **SHAP explainability**, ensuring that the insights generated were not only data-driven but also transparent and actionable.

The integration of interpretability into our modeling framework allowed us to identify key drivers of churn such as contract type, monthly charges, and tenure. These findings can directly inform customer success strategies, helping businesses intervene before customers leave. 

In summary, RetainX is more than just a churn prediction tool — it is a practical framework for using data science to deliver measurable business value. It bridges the gap between technical modeling and real-world decision-making, offering organizations a scalable, interpretable, and data-informed approach to customer retention.

---

## 9. References

- [Imbalanced-learn Documentation – SMOTE Module](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)  
- [Kaggle – Telco Customer Churn Dataset (IBM Sample)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
- [Scikit-learn: Machine Learning in Python](https://scikit-learn.org/)  
- [SHAP (SHapley Additive exPlanations) – Model Interpretability](https://shap.readthedocs.io/)  
- [XGBoost: Scalable and Flexible Gradient Boosting](https://xgboost.readthedocs.io/)
