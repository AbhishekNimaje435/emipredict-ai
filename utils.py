"""Shared helpers for the EMIPredict AI Streamlit app."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

from step3_feature_engineering import add_features, CAT_COLS

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"

TARGET_CLF = "emi_eligibility"
TARGET_REG = "max_monthly_emi"

INPUT_COLS = [
    "age", "gender", "marital_status", "education", "monthly_salary", "employment_type",
    "years_of_employment", "company_type", "house_type", "monthly_rent", "family_size",
    "dependents", "school_fees", "college_fees", "travel_expenses", "groceries_utilities",
    "other_monthly_expenses", "existing_loans", "current_emi_amount", "credit_score",
    "bank_balance", "emergency_fund", "emi_scenario", "requested_amount", "requested_tenure",
]
NUM_INPUTS = [c for c in INPUT_COLS if c not in CAT_COLS]

GROUPS = {
    "Personal": ["age", "gender", "marital_status", "education"],
    "Employment & Income": ["monthly_salary", "employment_type", "years_of_employment", "company_type"],
    "Housing & Family": ["house_type", "monthly_rent", "family_size", "dependents"],
    "Monthly Obligations": ["school_fees", "college_fees", "travel_expenses",
                            "groceries_utilities", "other_monthly_expenses"],
    "Credit & Assets": ["existing_loans", "current_emi_amount", "credit_score",
                        "bank_balance", "emergency_fund"],
    "Loan Request": ["emi_scenario", "requested_amount", "requested_tenure"],
}
INT_COLS = {"age", "family_size", "dependents", "credit_score", "requested_tenure"}
STEPS = {"years_of_employment": 0.5}

LABEL_COLORS = {"Eligible": "#2e9e5b", "High_Risk": "#e08a1e", "Not_Eligible": "#d64545"}


def label_of(col: str) -> str:
    return col.replace("_", " ").title().replace("Emi", "EMI")


def inr(x) -> str:
    return f"₹{float(x):,.0f}"


def _load_booster(path: Path) -> "xgb.Booster":
    # Load from XGBoost's native JSON format (unlike pickle, it is not version-sensitive)
    booster = xgb.Booster()
    booster.load_model(str(path))
    return booster


@st.cache_resource(show_spinner="Loading models...")
def load_artifacts():
    return {
        "pre": joblib.load(MODELS_DIR / "preprocessor.joblib"),
        "le": joblib.load(MODELS_DIR / "label_encoder.joblib"),
        "feature_cols": joblib.load(MODELS_DIR / "feature_cols.joblib"),
        "clf": _load_booster(MODELS_DIR / "xgb_classifier.json"),
        "reg": _load_booster(MODELS_DIR / "xgb_regressor.json"),
    }


@st.cache_data
def load_categories():
    return json.loads((MODELS_DIR / "categories.json").read_text())


@st.cache_data
def load_sample():
    return pd.read_csv(DATA_DIR / "sample_emi.csv")


def prepare_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric columns to numbers and fill missing values with the median."""
    cats = load_categories()
    out = df[INPUT_COLS].copy()
    for c in NUM_INPUTS:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out[c] = out[c].fillna(cats["numeric"][c]["median"])
    return out


def predict(df_raw: pd.DataFrame):
    """Raw applicant rows -> (predictions DataFrame, engineered features DataFrame)."""
    art = load_artifacts()
    feats = add_features(prepare_inputs(df_raw))
    X = art["pre"].transform(feats[art["feature_cols"]])

    dm = xgb.DMatrix(X)
    proba = art["clf"].predict(dm)            # multi:softprob -> (n_rows, n_classes)
    labels = art["le"].inverse_transform(proba.argmax(axis=1))
    max_emi = np.clip(art["reg"].predict(dm), 0, None)

    out = pd.DataFrame({
        "predicted_eligibility": labels,
        "predicted_max_emi": max_emi.round(0),
    })
    for i, cls in enumerate(art["le"].classes_):
        out[f"prob_{cls}"] = proba[:, i]
    return out, feats


def render_inputs(prefix: str, defaults: dict | None = None) -> dict:
    """Applicant form widgets (call inside st.form). Returns a dict of raw inputs."""
    cats = load_categories()
    defaults = defaults or {}
    values = {}
    for group, cols in GROUPS.items():
        st.markdown(f"**{group}**")
        grid = st.columns(3)
        for i, col in enumerate(cols):
            box = grid[i % 3]
            key = f"{prefix}_{col}"
            if col in CAT_COLS:
                opts = cats["categorical"][col]
                d = defaults.get(col, opts[0])
                values[col] = box.selectbox(label_of(col), opts,
                                            index=opts.index(d) if d in opts else 0, key=key)
            else:
                stats = cats["numeric"][col]
                d = defaults.get(col, stats["median"])
                if col in INT_COLS:
                    lo, hi = int(stats["min"]), int(stats["max"])
                    values[col] = box.number_input(label_of(col), min_value=lo, max_value=hi,
                                                   value=int(min(max(d, lo), hi)), step=1, key=key)
                else:
                    step = STEPS.get(col, 500.0)
                    values[col] = box.number_input(label_of(col), min_value=0.0,
                                                   value=float(d), step=float(step), key=key)
    return {c: values[c] for c in INPUT_COLS}
