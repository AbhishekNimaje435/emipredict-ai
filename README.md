# 💳 EMIPredict AI – Intelligent Financial Risk Assessment Platform

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://abhisheknimaje435-emipredict-ai-app-ccio5a.streamlit.app/)

**🔗 Live app:** https://abhisheknimaje435-emipredict-ai-app-ccio5a.streamlit.app/

EMIPredict AI is an end-to-end machine learning platform for the FinTech and banking domain. For every loan applicant it answers two questions in real time:

1. **Is the applicant eligible for the EMI?** (classification: `Eligible` / `High_Risk` / `Not_Eligible`)
2. **What is the maximum monthly EMI they can safely afford?** (regression, in INR)

Models were trained on **400,000 financial profiles** (22 input features, 5 EMI scenarios), tracked and compared with **MLflow**, registered in the **MLflow Model Registry**, and served through a multi-page **Streamlit** app deployed on Streamlit Cloud.

> ⚠️ Educational project. Predictions are not a substitute for real lending decisions.

---

## 📸 Screenshots

### Streamlit application

| Real-time prediction | Prediction result |
|---|---|
| ![Prediction form](screenshots/prediction_form.png) | ![Prediction result](screenshots/prediction_result.png) |

| Data Explorer | Eligibility by EMI scenario |
|---|---|
| ![Data Explorer](screenshots/data_explorer.png) | ![Eligibility by scenario](screenshots/eligibility_by_scenario.png) |

| Model Performance | Confusion matrix and feature importance |
|---|---|
| ![Model comparison](screenshots/model_performance.png) | ![Confusion matrix](screenshots/confusion_matrix_feature_importance.png) |

| Admin – CRUD | MLflow registry tab |
|---|---|
| ![Admin CRUD](screenshots/admin_crud.png) | ![Registry tab in app](screenshots/model_registry_tab.png) |

### MLflow experiment tracking and model registry

| Experiments list | Compare runs |
|---|---|
| ![MLflow experiments](screenshots/mlflow_experiments.png) | ![MLflow compare](screenshots/mlflow_compare.png) |

| Run details | Model registry |
|---|---|
| ![MLflow run](screenshots/mlflow_run_detail.png) | ![MLflow registry](screenshots/mlflow_registry.png) |

---

## ✨ Features

- **Real-time prediction** for a single applicant: eligibility, confidence, class probabilities, maximum safe EMI, requested EMI and headroom, plus a financial health snapshot (debt-to-income, expense-to-income, requested EMI-to-income, disposable income, savings buffer).
- **Batch scoring** by uploading a CSV.
- **Data Explorer** with filters (EMI scenario, eligibility, employment type) and tabs for distributions, relationships, engineered ratios, correlation and raw data.
- **Model Performance dashboard** with the MLflow comparison table, metric charts, confusion matrix and feature importance.
- **Admin Data Management** with full CRUD (Create, Read, Update, Delete), filtering and pagination.

## 📑 App pages

| Page | File | Purpose |
|---|---|---|
| Home | `app.py` | Overview and headline test metrics |
| 🎯 Prediction | `pages/1_Prediction.py` | Single applicant and batch CSV predictions |
| 📊 Data Explorer | `pages/2_Data_Explorer.py` | Interactive EDA |
| 📈 Model Performance | `pages/3_Model_Performance.py` | MLflow experiment comparison |
| 🛠️ Admin | `pages/4_Admin_Data_Management.py` | CRUD on financial records |

---

## 🧠 Methodology

```
Dataset (400K records)
   ↓
Data cleaning and quality checks
   ↓
Feature engineering (DTI, affordability ratios, risk features, interactions)
   ↓
Train 4 classifiers + 4 regressors, tracked in MLflow
   ↓
Model evaluation, best model selection, MLflow Model Registry
   ↓
Streamlit app → Streamlit Cloud
```

**Engineered features** include debt-to-income, expense-to-income, EMI-to-income, EMI-to-disposable-income, affordability ratio, disposable income, savings buffer (months) and interaction features such as salary × credit score and DTI × credit score.

### Dataset

| Item | Detail |
|---|---|
| Records | 400,000 financial profiles |
| Input features | 22 demographic, employment, housing, obligation and credit variables, plus the loan request |
| EMI scenarios | E-commerce, Home Appliances, Vehicle, Personal Loan, Education (80,000 records each) |
| Classification target | `emi_eligibility` (Eligible / High_Risk / Not_Eligible) |
| Regression target | `max_monthly_emi` (₹500–₹50,000) |

The repository ships `data/sample_emi.csv`, a **20,000-record sample** used by the Data Explorer and Admin pages so the cloud app stays fast.

---

## 📊 Results

Four models were trained for each task. Selection used the **validation set**; only the chosen models were scored once on the **held-out test set (60,004 records)**.

### Classification (EMI eligibility)

| Model | Val. accuracy | Val. macro F1 | Val. ROC-AUC |
|---|---|---|---|
| **XGBoost ✅** | **0.9752** | **0.9119** | **0.9969** |
| Decision Tree | 0.9403 | 0.8276 | 0.9681 |
| Random Forest | 0.9209 | 0.7932 | 0.9859 |
| Logistic Regression | 0.8601 | 0.7140 | 0.9654 |

### Regression (maximum monthly EMI)

| Model | Val. RMSE (₹) | Val. MAE (₹) | Val. R² | Val. MAPE (%) |
|---|---|---|---|---|
| **XGBoost ✅** | **704.2** | **239.5** | **0.9918** | **7.17** |
| Random Forest | 1,071.8 | 320.8 | 0.9809 | 8.42 |
| Decision Tree | 1,312.1 | 447.6 | 0.9714 | 10.10 |
| Linear Regression | 3,556.4 | 2,428.1 | 0.7901 | 161.64 |

### Final test-set performance (selected XGBoost models)

| Task | Metrics |
|---|---|
| Classification | Accuracy **97.5%** · Macro F1 **0.9127** · ROC-AUC **0.997** · High_Risk recall **90.47%** |
| Regression | RMSE **₹746.17** · MAE **₹240.47** · R² **0.9906** · MAPE **7.26%** |

Both results beat the project targets (accuracy > 90%, RMSE < ₹2,000). XGBoost won on every metric in both tasks, trained fast (47 s for the regressor versus about 20 minutes for Random Forest), and showed almost identical validation and test scores, so there is no sign of overfitting.

### Key insights

- Engineered affordability ratios (EMI-to-disposable-income, DTI × credit score) are the strongest drivers of eligibility.
- Existing loan status and disposable income dominate the maximum-EMI model.
- Eligibility is highest for small-ticket scenarios (e-commerce, home appliances) and lowest for vehicle and personal loans.
- `High_Risk` is the hardest class to predict, so it works best as a "refer to underwriter" bucket.

---

## 🔬 MLflow tracking and model registry

All 8 models were logged with parameters, validation/test metrics, training time and artifacts. The best model of each task is registered in the **MLflow Model Registry** with the alias `production`:

| Registered model | Version | Alias | Algorithm |
|---|---|---|---|
| `EMI_Eligibility_Classifier` | 1 | `production` | XGBoost Classifier |
| `EMI_Amount_Regressor` | 1 | `production` | XGBoost Regressor |

The deployed app loads the exported XGBoost models from `models/` and reads the MLflow comparison export from `reports/model_comparison.csv`. Tracking uses a local SQLite backend (`mlflow.db`), which is not committed to the repository. To browse the experiments, run this from the project folder that contains `mlflow.db`, then open http://127.0.0.1:5000:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --workers 1
```

---

## 🗂️ Project structure

```
emipredict-ai/
├── app.py                           # Home page and headline metrics
├── utils.py                         # Shared helpers: model loading, inputs, formatting
├── step3_feature_engineering.py     # Feature engineering used at training and inference time
├── requirements.txt
├── pages/
│   ├── 1_Prediction.py
│   ├── 2_Data_Explorer.py
│   ├── 3_Model_Performance.py
│   └── 4_Admin_Data_Management.py
├── models/
│   ├── xgb_classifier.json          # Selected classifier (native XGBoost format)
│   ├── xgb_regressor.json           # Selected regressor
│   ├── preprocessor.joblib
│   ├── label_encoder.joblib
│   ├── feature_cols.joblib
│   └── categories.json
├── data/
│   └── sample_emi.csv               # 20,000-record sample of the dataset
├── reports/
│   ├── model_comparison.csv         # Exported MLflow comparison of all 8 models
│   ├── registry_info.json           # Model registry versions and aliases
│   ├── cm_XGBoost.png
│   ├── fi_clf_XGBoost.png
│   └── fi_reg_XGBoost.png
└── screenshots/                     # Images used in this README
```

---

## 🚀 Run locally

```bash
# 1. Clone the repository
git clone https://github.com/AbhishekNimaje435/emipredict-ai.git
cd emipredict-ai

# 2. (Optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the app
streamlit run app.py
```

The app opens at http://localhost:8501.

### Requirements

```
scikit-learn==1.6.1
xgboost==3.4.1
pandas==2.2.3
numpy==2.1.3
joblib==1.6.0
streamlit>=1.40
plotly>=5.20
```

## ☁️ Deployment

The app is deployed on **Streamlit Cloud** directly from this GitHub repository, so every push to the main branch redeploys it automatically. XGBoost models are stored in XGBoost's native JSON format, which does not depend on the Python or library version used for training.

**Note:** the Streamlit Cloud file system is temporary, so records created or edited on the Admin page may reset when the app restarts. A production system would use a persistent database.

## 🛠️ Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · MLflow · Streamlit · Plotly · Streamlit Cloud · GitHub

## 👤 Author

**Abhishek Nimaje** – [GitHub](https://github.com/AbhishekNimaje435)
