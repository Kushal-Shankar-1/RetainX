# RetainX

## 📈 Predictive Customer Churn Analysis and Retention Strategy Platform

**Team 7: Colorado Blue Spruce**  
Kushal Shankar | Ashmitha Appandaraju | Tej Sidda | Ronak Vadhaiya

---

### 🔎 About RetainX
RetainX is a predictive analytics solution designed to help subscription-based businesses proactively identify customers at risk of churn and implement targeted retention strategies.

---

### 📂 Project Structure
- **data/** — Contains the dataset  
- **notebooks/** — Jupyter notebooks for each step: EDA, preprocessing, feature engineering, modeling, and insights  
- **results/** — Plots, evaluation metrics, and feature importance charts  
- **report/** — Final report drafts and references  
- **presentation/** — Presentation template and final deck  

---

### ✅ Key Features
- Thorough exploratory data analysis  
- Feature engineering for actionable insights  
- Baseline models: Logistic Regression, Decision Tree  
- Advanced models: Random Forest, XGBoost with hyperparameter tuning  
- Addressed class imbalance using SMOTE  
- Model interpretability with SHAP plots  
- Business recommendations for churn reduction  

---

### 💻 Technologies Used
- Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, Imbalanced-learn)  
- JupyterLab  
- SHAP for interpretability  

---

### 📊 Dataset
**Telco Customer Churn Dataset** (IBM sample, via Kaggle)  
> Link: [Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

### 🚀 How to Run
1. Clone this repo:  
```bash
git clone https://github.com/Kushal-Shankar-1/RetainX.git
```
2. Create virtual environment:
```bash
python3 -m venv churn_env
source churn_env/bin/activate
pip install -r requirements.txt
```
3. Run each notebook sequentially in **notebooks/**
