import pandas as pd
import streamlit as st

from theme import apply_theme
from utils import REPORTS_DIR, inr

st.set_page_config(page_title="EMIPredict AI", page_icon="💳", layout="wide")
apply_theme()

st.markdown("""<div class="hero">
<span class="pill"><i class="dot"></i> Live &middot; real-time loan risk assessment</span>
<h1 class="hero-title">EMIPredict <span>AI</span></h1>
<p class="hero-sub">Intelligent Financial Risk Assessment Platform</p>
<p class="hero-desc">Find out in seconds whether an applicant is eligible for an EMI and how much EMI is safe to afford. Two XGBoost models, tracked and registered with MLflow.</p>
<div class="stats">
<div class="stat"><b class="cnt c400"></b><span>financial profiles analysed</span></div>
<div class="stat"><b class="cnt c97"></b><span>eligibility accuracy</span></div>
<div class="stat"><b class="cnt c746"></b><span>RMSE on maximum EMI</span></div>
<div class="stat"><b class="cnt c8"></b><span>models tracked in MLflow</span></div>
</div></div>""", unsafe_allow_html=True)

# ---- Headline metrics (reports/model_comparison.csv) ----
comp_path = REPORTS_DIR / "model_comparison.csv"
if comp_path.exists():
    st.markdown("### Selected models")
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

st.markdown("### Explore the platform")
a, b = st.columns(2)
a.page_link("pages/1_Prediction.py", label="Prediction: single applicant or batch CSV", icon="🎯")
b.page_link("pages/2_Data_Explorer.py", label="Data Explorer: interactive EDA", icon="📊")
c, d = st.columns(2)
c.page_link("pages/3_Model_Performance.py", label="Model Performance: MLflow experiment comparison", icon="📈")
d.page_link("pages/4_Admin_Data_Management.py", label="Admin: financial records CRUD", icon="🛠️")

st.markdown(
    """
**How it works:** data cleaning → feature engineering (DTI, affordability, risk scores) →
4 classifiers + 4 regressors tracked in MLflow → best models registered → this app.

*This is an educational project. Predictions are not a substitute for real lending decisions.*
"""
)
