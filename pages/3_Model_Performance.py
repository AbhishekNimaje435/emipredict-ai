import json

import pandas as pd
import plotly.express as px
import streamlit as st

from theme import apply_theme

from utils import REPORTS_DIR

st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")
apply_theme()
st.title("📈 Model Performance & MLflow Tracking")

comp_path = REPORTS_DIR / "model_comparison.csv"
if not comp_path.exists():
    st.error("reports/model_comparison.csv not found. Run step6_prepare_app_assets.py first.")
    st.stop()

comp = pd.read_csv(comp_path)
comp["selected"] = comp["selected"].astype(str).str.lower() == "true"


def show_task(task: str, default_metric: str, lower_is_better: bool):
    d = comp[comp["task"] == task].dropna(axis=1, how="all").copy()
    metric_cols = [c for c in d.columns if c.startswith(("val_", "test_")) or c == "train_time_sec"]
    val_cols = [c for c in metric_cols if c.startswith("val_")]

    best = d[d["selected"]]
    if not best.empty:
        st.success(f"Selected for production: **{best.iloc[0]['model']}** "
                   f"(selection metric: {default_metric})")

    show = d[["model", "selected"] + metric_cols].sort_values(default_metric, ascending=lower_is_better)
    st.dataframe(show.round(4), hide_index=True)

    metric = st.selectbox("Chart metric", val_cols, index=val_cols.index(default_metric),
                          key=f"metric_{task}")
    st.plotly_chart(px.bar(d, x="model", y=metric, color="selected", text_auto=".3f",
                           color_discrete_map={True: "#2e9e5b", False: "#9aa5b1"},
                           title=f"{metric} by model"))
    return best


clf_tab, reg_tab, reg_tab2 = st.tabs(["Classification", "Regression", "MLflow registry"])

with clf_tab:
    best = show_task("classification", "val_f1_macro", False)
    if not best.empty:
        name = best.iloc[0]["model"]
        a, b = st.columns(2)
        for col_box, fname, cap in [(a, f"cm_{name}.png", "Confusion matrix (validation)"),
                                    (b, f"fi_clf_{name}.png", "Top feature importances")]:
            p = REPORTS_DIR / fname
            if p.exists():
                col_box.image(str(p), caption=cap)

with reg_tab:
    best = show_task("regression", "val_rmse", True)
    if not best.empty:
        name = best.iloc[0]["model"]
        p = REPORTS_DIR / f"fi_reg_{name}.png"
        if p.exists():
            st.image(str(p), caption="Top feature importances")

with reg_tab2:
    reg_path = REPORTS_DIR / "registry_info.json"
    if reg_path.exists():
        info = json.loads(reg_path.read_text())
        st.dataframe(pd.DataFrame(info["models"]), hide_index=True)
    st.markdown("**To open the MLflow UI locally** (from the project folder that contains `mlflow.db`):")
    st.code("mlflow ui --backend-store-uri sqlite:///mlflow.db --workers 1", language="bash")
