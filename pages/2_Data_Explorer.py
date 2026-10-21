import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from step3_feature_engineering import add_features
from utils import (LABEL_COLORS, TARGET_CLF, TARGET_REG, inr, label_of, load_categories,
                   load_sample, prepare_inputs, INPUT_COLS)

st.set_page_config(page_title="Data Explorer", page_icon="📊", layout="wide")
st.title("📊 Data Explorer")
st.caption("Random sample of 20,000 records from the cleaned dataset (for performance).")

df = load_sample()

with st.sidebar:
    st.header("Filters")
    scen = st.multiselect("EMI scenario", sorted(df["emi_scenario"].unique()))
    elig = st.multiselect("Eligibility", sorted(df[TARGET_CLF].unique()))
    emp = st.multiselect("Employment type", sorted(df["employment_type"].unique()))

fdf = df.copy()
if scen:
    fdf = fdf[fdf["emi_scenario"].isin(scen)]
if elig:
    fdf = fdf[fdf[TARGET_CLF].isin(elig)]
if emp:
    fdf = fdf[fdf["employment_type"].isin(emp)]

if fdf.empty:
    st.warning("No records match the current filters.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Records", f"{len(fdf):,}")
c2.metric("Eligible", f"{(fdf[TARGET_CLF] == 'Eligible').mean():.1%}")
c3.metric("Avg monthly salary", inr(fdf["monthly_salary"].mean()))
c4.metric("Avg max EMI", inr(fdf[TARGET_REG].mean()))

num_cols = [c for c in fdf.select_dtypes(include=np.number).columns]
cat_cols = list(load_categories()["categorical"].keys())

t1, t2, t3, t4, t5 = st.tabs(["Distributions", "Relationships", "Engineered ratios",
                              "Correlation", "Raw data"])

with t1:
    a, b = st.columns(2)
    col = a.selectbox("Numeric column", num_cols, index=num_cols.index("monthly_salary"))
    st.plotly_chart(px.histogram(fdf, x=col, color=TARGET_CLF, nbins=50, barmode="overlay",
                                 opacity=0.7, color_discrete_map=LABEL_COLORS,
                                 title=f"{label_of(col)} distribution"))
    ccol = b.selectbox("Categorical column", cat_cols, index=cat_cols.index("emi_scenario"))
    ct = pd.crosstab(fdf[ccol], fdf[TARGET_CLF], normalize="index").reset_index()
    ct = ct.melt(id_vars=ccol, var_name="eligibility", value_name="share")
    st.plotly_chart(px.bar(ct, x=ccol, y="share", color="eligibility", barmode="stack",
                           color_discrete_map=LABEL_COLORS,
                           title=f"Eligibility share by {label_of(ccol)}"))

with t2:
    a, b = st.columns(2)
    x = a.selectbox("X", num_cols, index=num_cols.index("monthly_salary"), key="sx")
    y = b.selectbox("Y", num_cols, index=num_cols.index(TARGET_REG), key="sy")
    samp = fdf.sample(min(5000, len(fdf)), random_state=1)
    st.plotly_chart(px.scatter(samp, x=x, y=y, color=TARGET_CLF, opacity=0.5,
                               color_discrete_map=LABEL_COLORS))
    avg = fdf.groupby("emi_scenario")[TARGET_REG].mean().reset_index()
    st.plotly_chart(px.bar(avg, x="emi_scenario", y=TARGET_REG,
                           title="Average max EMI by scenario"))

with t3:
    feats = add_features(prepare_inputs(fdf))
    feats[TARGET_CLF] = fdf[TARGET_CLF].values
    ratio_cols = ["debt_to_income", "expense_to_income", "emi_to_income", "affordability_ratio",
                  "loan_to_annual_income", "savings_months", "credit_risk_score",
                  "employment_stability"]
    r = st.selectbox("Ratio", ratio_cols)
    hi = feats[r].quantile(0.99)
    st.plotly_chart(px.box(feats[feats[r] <= hi], x=TARGET_CLF, y=r, color=TARGET_CLF,
                           color_discrete_map=LABEL_COLORS,
                           title=f"{label_of(r)} by eligibility (top 1% outliers removed)"))

with t4:
    corr = fdf[num_cols].corr()
    st.plotly_chart(px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                              aspect="auto", height=700))

with t5:
    st.dataframe(fdf.head(500))
    st.download_button("Filtered data CSV", fdf.to_csv(index=False), "filtered_emi.csv", "text/csv")
