import json
import pandas as pd
import streamlit as st

from utils import REPORTS_DIR, inr

st.set_page_config(page_title="EMIPredict AI", page_icon="💳", layout="wide")

st.title("💳 EMIPredict AI")
st.subheader("Intelligent Financial Risk Assessment Platform")
st.write(
    "Real-time assessment of EMI eligibility (classification) and maximum safe monthly EMI "
    "(regression) using machine learning. Models are tracked and registered with MLflow."
)

# ---- Headline metrics (reports/model_comparison.csv se) ----
comp_path = REPORTS_DIR / "model_comparison.csv"
if comp_path.exists():
    comp = pd.read_csv(comp_path)
    sel = comp[comp["selected"].astype(str).str.lower() == "true"]
    clf = sel[sel["task"] == "classification"]
    reg = sel[sel["task"] == "regression"]
    c1, c2, c3, c4 = st.columns(4)
    if not clf.empty:
        r = clf.iloc[0]
        c1.metric("Best classifier", r["model"])
        c2.metric("Test accuracy", f"{r['test_accuracy']:.1%}")
        c3.metric("Test macro F1", f"{r['test_f1_macro']:.3f}")
    if not reg.empty:
        r = reg.iloc[0]
        c4.metric(f"Best regressor ({r['model']}) RMSE", inr(r["test_rmse"]))
else:
    st.info("Model comparison report is not available yet.")

st.divider()
st.markdown("### Pages")
st.page_link("pages/1_Prediction.py", label="Prediction - single applicant or batch CSV", icon="🎯")
st.page_link("pages/2_Data_Explorer.py", label="Data Explorer - interactive EDA", icon="📊")
st.page_link("pages/3_Model_Performance.py", label="Model Performance - MLflow experiment comparison", icon="📈")
st.page_link("pages/4_Admin_Data_Management.py", label="Admin - financial records CRUD", icon="🛠️")

st.divider()
st.markdown(
    """
**Architecture:** Data cleaning → feature engineering (DTI, affordability, risk scores) →
4 classifiers + 4 regressors tracked in MLflow → best models registered → Streamlit app.

*This is an educational project. Predictions are not a substitute for real lending decisions.*
"""
)
