"""
STEP 3: Feature Engineering + Encoding + Scaling
Run: python step3_feature_engineering.py

add_features() is also imported by the Streamlit app,
so training and prediction use exactly the same logic.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

TARGET_CLF, TARGET_REG = "emi_eligibility", "max_monthly_emi"

CAT_COLS = [
    "gender", "marital_status", "education", "employment_type", "company_type",
    "house_type", "existing_loans", "emi_scenario",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    sal = df["monthly_salary"].replace(0, np.nan)

    # --- Financial ratios ---
    df["total_expenses"] = (
        df["monthly_rent"] + df["school_fees"] + df["college_fees"]
        + df["travel_expenses"] + df["groceries_utilities"] + df["other_monthly_expenses"]
    )
    df["debt_to_income"] = df["current_emi_amount"] / sal
    df["expense_to_income"] = df["total_expenses"] / sal
    df["disposable_income"] = (
        df["monthly_salary"] - df["total_expenses"] - df["current_emi_amount"]
    )
    df["affordability_ratio"] = df["disposable_income"] / sal
    df["requested_emi_est"] = df["requested_amount"] / df["requested_tenure"]
    df["emi_to_income"] = df["requested_emi_est"] / sal
    df["emi_to_disposable"] = df["requested_emi_est"] / df["disposable_income"].clip(lower=1)
    df["loan_to_annual_income"] = df["requested_amount"] / (sal * 12)

    # --- Savings / buffer ---
    monthly_burden = df["total_expenses"] + df["current_emi_amount"] + 1
    df["savings_months"] = (df["bank_balance"] + df["emergency_fund"]) / monthly_burden
    df["emergency_cover"] = df["emergency_fund"] / (df["total_expenses"] + 1)

    # --- Risk scoring ---
    df["credit_risk_score"] = (df["credit_score"] - 300) / 550
    df["employment_stability"] = (
        df["years_of_employment"] / (df["age"] - 18).clip(lower=1)
    ).clip(0, 1)
    df["dependents_ratio"] = df["dependents"] / df["family_size"].clip(lower=1)

    # --- Interaction features ---
    df["salary_x_credit"] = df["monthly_salary"] * df["credit_risk_score"]
    df["dti_x_credit"] = df["debt_to_income"] * (1 - df["credit_risk_score"])
    df["emi_burden_x_dependents"] = df["emi_to_income"] * (1 + df["dependents"])

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    return df


def build_preprocessor(df_features: pd.DataFrame):
    num_cols = [c for c in df_features.columns
                if c not in CAT_COLS + [TARGET_CLF, TARGET_REG]]
    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
    ])
    return pre, num_cols


if __name__ == "__main__":
    train = add_features(pd.read_csv("data/train.csv"))
    val = add_features(pd.read_csv("data/val.csv"))
    test = add_features(pd.read_csv("data/test.csv"))

    pre, num_cols = build_preprocessor(train)
    feature_cols = [c for c in train.columns if c not in [TARGET_CLF, TARGET_REG]]

    # Sirf train par fit -> data leakage nahi hoga
    X_train = pre.fit_transform(train[feature_cols])
    X_val = pre.transform(val[feature_cols])
    X_test = pre.transform(test[feature_cols])

    # Classification target encode (Eligible / High_Risk / Not_Eligible -> 0/1/2)
    le = LabelEncoder().fit(train[TARGET_CLF])
    y_train_clf = le.transform(train[TARGET_CLF])
    y_val_clf = le.transform(val[TARGET_CLF])
    y_test_clf = le.transform(test[TARGET_CLF])
    print("Classes:", dict(zip(le.classes_, le.transform(le.classes_))))

    y_train_reg = train[TARGET_REG].values
    y_val_reg = val[TARGET_REG].values
    y_test_reg = test[TARGET_REG].values

    out_names = pre.get_feature_names_out()
    joblib.dump(
        dict(
            X_train=X_train, X_val=X_val, X_test=X_test,
            y_train_clf=y_train_clf, y_val_clf=y_val_clf, y_test_clf=y_test_clf,
            y_train_reg=y_train_reg, y_val_reg=y_val_reg, y_test_reg=y_test_reg,
            feature_names=list(out_names),
        ),
        "data/processed.joblib",
    )
    joblib.dump(pre, "models/preprocessor.joblib")
    joblib.dump(le, "models/label_encoder.joblib")
    joblib.dump(feature_cols, "models/feature_cols.joblib")

    print("X_train shape:", X_train.shape)
    print("Total features after encoding:", len(out_names))
    print("Saved: data/processed.joblib, models/preprocessor.joblib, label_encoder, feature_cols")
