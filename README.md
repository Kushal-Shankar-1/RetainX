
# RetainX — Customer Churn Prediction for Subscription-Based Services  
**Team 7: Colorado Blue Spruce**  
Kushal Shankar | Ashmitha Appandaraju | Tej Sidda | Ronak Vadhaiya  

## 📈 Project Overview  
RetainX is a predictive analytics tool designed to help subscription-based businesses proactively identify customers likely to churn and implement targeted retention strategies.

---

## 📂 Project Structure  
| Folder        | Contents                                                                    |
|---------------|-----------------------------------------------------------------------------|
| `data/`       | Preprocessed Telco Customer Churn dataset                                   |
| `models/`     | Saved XGBoost best model for interpretation                                 |
| `notebooks/`  | Jupyter notebooks for each stage: EDA, preprocessing, modeling, interpretation |
| `results/`    | Visuals and plots generated from the notebooks (PNG files)                  |
| `exports/`    | PDF exports of each notebook and the presentation                  |
| `report/`     | Final project report                                                        |
| `team_testing_checklist.md` | Documented testing done by each team member                                      |

---

## ✅ Workflow Summary  
- Data Cleaning & EDA  
- Preprocessing & Feature Engineering  
- Baseline Models (Logistic Regression, Decision Tree)  
- Advanced Modeling (Random Forest, XGBoost)  
- SHAP Explainability  
- Business Insights and Recommendations  

---

## 📚 Dataset  
[Telco Customer Churn dataset (IBM Sample on Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  

---

## 🛠 Technologies Used  
- Python: Pandas, NumPy, Scikit-learn, XGBoost, SHAP, Matplotlib, Seaborn  
- Jupyter Notebooks  
- Joblib (for model persistence)

---

## ✅ Key Findings  
- Customers with month-to-month contracts and high monthly charges are at the highest risk of churn.  
- Electronic check payment method users are more prone to churn.  
- Tenure is a strong retention factor.  

---

## 📥 Reproducibility  
```bash
git clone https://github.com/Kushal-Shankar-1/RetainX.git
cd RetainX_Final_Submission/
pip install -r requirements.txt
jupyter lab
```
Then run notebooks in order inside `/notebooks`.

---

## ✅ Conclusion  
RetainX offers an end-to-end churn prediction and analysis workflow and provides actionable business recommendations for proactive retention strategies.
