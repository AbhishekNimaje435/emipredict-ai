import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (INPUT_COLS, LABEL_COLORS, NUM_INPUTS, inr, load_artifacts, load_categories,
                   predict, render_inputs)

st.set_page_config(page_title="EMI Prediction", page_icon="🎯", layout="wide")
st.title("🎯 Real-time EMI Assessment")

tab_single, tab_batch = st.tabs(["Single applicant", "Batch (CSV)"])

# ------------------------------------------------------------------ single
with tab_single:
    with st.form("single_applicant"):
        values = render_inputs("pred")
        submitted = st.form_submit_button("Assess applicant", type="primary")

    if submitted:
        try:
            res, feats = predict(pd.DataFrame([values]))
        except Exception as e:  # model/feature mismatch etc.
            st.error(f"Prediction failed: {e}")
            st.stop()

        label = res.loc[0, "predicted_eligibility"]
        max_emi = float(res.loc[0, "predicted_max_emi"])
        classes = list(load_artifacts()["le"].classes_)
        probs = {c: float(res.loc[0, f"prob_{c}"]) for c in classes}
        requested_emi = values["requested_amount"] / max(values["requested_tenure"], 1)

        st.divider()
        msg = {
            "Eligible": "✅ Eligible - low risk, EMI comfortably affordable.",
            "High_Risk": "⚠️ High risk - marginal case; consider a higher interest rate or a smaller loan amount.",
            "Not_Eligible": "❌ Not eligible - loan not recommended.",
        }.get(label, label)
        {"Eligible": st.success, "High_Risk": st.warning}.get(label, st.error)(msg)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Eligibility", label.replace("_", " "))
        m2.metric("Confidence", f"{probs[label]:.1%}")
        m3.metric("Max safe monthly EMI", inr(max_emi))
        headroom = max_emi - requested_emi
        m4.metric("Requested EMI (approx., no interest)", inr(requested_emi),
                  delta=f"{'+' if headroom >= 0 else '-'}{inr(abs(headroom))} headroom",
                  delta_color="normal")

        left, right = st.columns(2)
        with left:
            fig = go.Figure(go.Bar(
                x=list(probs.keys()), y=list(probs.values()),
                marker_color=[LABEL_COLORS.get(c, "#888") for c in probs],
                text=[f"{v:.1%}" for v in probs.values()], textposition="outside"))
            fig.update_layout(title="Class probabilities", yaxis_range=[0, 1.1],
                              yaxis_tickformat=".0%", height=360)
            st.plotly_chart(fig)
        with right:
            f = feats.iloc[0]
            ratios = pd.DataFrame({
                "Metric": ["Debt-to-income", "Expense-to-income", "Requested EMI-to-income",
                           "Disposable income / month", "Savings buffer (months)"],
                "Value": [f"{f['debt_to_income']:.1%}", f"{f['expense_to_income']:.1%}",
                          f"{f['emi_to_income']:.1%}", inr(f["disposable_income"]),
                          f"{f['savings_months']:.1f}"],
            })
            st.markdown("**Financial health snapshot**")
            st.dataframe(ratios, hide_index=True)
            if requested_emi > max_emi:
                st.warning("Requested EMI exceeds the model's maximum safe EMI. "
                           "Increase the tenure or reduce the loan amount.")

# ------------------------------------------------------------------- batch
with tab_batch:
    st.write("Your CSV must contain these columns: " + ", ".join(f"`{c}`" for c in INPUT_COLS))
    cats = load_categories()
    template = pd.DataFrame([{**{c: cats["categorical"][c][0] for c in cats["categorical"]},
                              **{c: cats["numeric"][c]["median"] for c in NUM_INPUTS}}])[INPUT_COLS]
    st.download_button("Download template CSV", template.to_csv(index=False),
                       "emi_template.csv", "text/csv")

    up = st.file_uploader("Upload applicants CSV", type="csv")
    if up is not None:
        df = pd.read_csv(up)
        missing = [c for c in INPUT_COLS if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        elif len(df) > 50_000:
            st.error("A maximum of 50,000 rows is allowed per upload.")
        else:
            try:
                res, _ = predict(df)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()
            out = pd.concat([df.reset_index(drop=True), res], axis=1)
            st.success(f"Assessed {len(out):,} applicants.")
            st.dataframe(out.head(200))
            counts = res["predicted_eligibility"].value_counts()
            st.bar_chart(counts)
            st.download_button("Download results CSV", out.to_csv(index=False),
                               "emi_predictions.csv", "text/csv")
