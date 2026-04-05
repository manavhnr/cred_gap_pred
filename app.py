import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Strategic Credit Gap Predictor", layout="wide")

# -----------------------------
# Load trained artifacts
# -----------------------------
artifacts = joblib.load("credit_risk_artifacts.pkl")
model = artifacts["model"]
feature_columns = artifacts["feature_columns"]
threshold = artifacts["threshold"]

try:
    feature_mapping = joblib.load("feature_mapping.pkl")
except Exception:
    feature_mapping = {}

# -----------------------------
# Human-readable labels
# -----------------------------
display_name_mapping = {
    "AMT_INCOME_TOTAL": "Annual Income",
    "AMT_CREDIT": "Requested Loan Amount",
    "AMT_ANNUITY": "Loan Annuity",
    "AMT_GOODS_PRICE": "Goods Price",
    "CNT_CHILDREN": "Number of Children",
    "CNT_FAM_MEMBERS": "Number of Family Members",
    "DAYS_BIRTH": "Age in Days",
    "DAYS_EMPLOYED": "Days Employed",
    "EXT_SOURCE_1": "External Credit Score 1",
    "EXT_SOURCE_2": "External Credit Score 2",
    "EXT_SOURCE_3": "External Credit Score 3",
    "AMT_REQ_CREDIT_BUREAU_YEAR": "Credit Inquiries in Last Year",
    "OBS_30_CNT_SOCIAL_CIRCLE": "Defaults in Social Circle (30 Days)",
    "CODE_GENDER_M": "Male Applicant",
    "NAME_INCOME_TYPE_Working": "Working Professional",
    "NAME_INCOME_TYPE_Commercial_associate": "Commercial Associate",
    "NAME_EDUCATION_TYPE_Secondary_secondary_special": "Secondary Education",
    "NAME_EDUCATION_TYPE_Higher_education": "Higher Education",
    "NAME_TYPE_SUITE_Family": "Family Accompanied Application",
    "NAME_TYPE_SUITE_Unaccompanied": "Unaccompanied Application",
    "HOUSETYPE_MODE_block_of_flats": "Lives in Apartment",
    "EMERGENCYSTATE_MODE_Unknown": "Emergency Information Missing",
    "income_to_credit_ratio": "Income to Credit Ratio",
    "annuity_to_income_ratio": "Annuity to Income Ratio",
    "credit_to_goods_ratio": "Credit to Goods Ratio",
    "children_ratio": "Children Dependency Ratio",
    "employment_to_age_ratio": "Employment to Age Ratio",
    "ext_source_mean": "Average External Credit Score",
}

important_features = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
    "OBS_30_CNT_SOCIAL_CIRCLE",
    "CODE_GENDER_M",
    "NAME_INCOME_TYPE_Working",
    "NAME_INCOME_TYPE_Commercial_associate",
    "NAME_EDUCATION_TYPE_Secondary_secondary_special",
    "NAME_EDUCATION_TYPE_Higher_education",
    "NAME_TYPE_SUITE_Family",
    "NAME_TYPE_SUITE_Unaccompanied",
    "HOUSETYPE_MODE_block_of_flats",
    "EMERGENCYSTATE_MODE_Unknown",
    "income_to_credit_ratio",
    "annuity_to_income_ratio",
    "credit_to_goods_ratio",
    "children_ratio",
    "employment_to_age_ratio",
    "ext_source_mean",
]

boolean_features = {
    "CODE_GENDER_M",
    "NAME_INCOME_TYPE_Working",
    "NAME_INCOME_TYPE_Commercial_associate",
    "NAME_EDUCATION_TYPE_Secondary_secondary_special",
    "NAME_EDUCATION_TYPE_Higher_education",
    "NAME_TYPE_SUITE_Family",
    "NAME_TYPE_SUITE_Unaccompanied",
    "HOUSETYPE_MODE_block_of_flats",
    "EMERGENCYSTATE_MODE_Unknown",
}

# -----------------------------
# Helpers
# -----------------------------
def readable_name(name: str) -> str:
    return feature_mapping.get(
        name,
        display_name_mapping.get(name, name.replace("_", " ").title())
    )


def prettify_feature_value(feature: str, value):
    if feature in boolean_features:
        return "Yes" if float(value) == 1 else "No"

    if feature == "DAYS_BIRTH":
        return f"{abs(int(value)) // 365} years"

    if feature == "DAYS_EMPLOYED":
        return f"{abs(int(value)) // 365} years"

    if isinstance(value, (int, float, np.integer, np.floating)):
        if abs(float(value)) >= 1000:
            return f"{float(value):,.2f}"
        return round(float(value), 4)

    return str(value)


def format_display_value(parameter: str, value):
    reverse_map = {v: k for k, v in display_name_mapping.items()}
    original_feature = reverse_map.get(parameter)

    if original_feature:
        return prettify_feature_value(original_feature, value)

    return str(value)


def make_display_dataframe(input_df: pd.DataFrame) -> pd.DataFrame:
    display_df = input_df.copy().T.reset_index()
    display_df.columns = ["Parameter", "Value"]
    display_df["Parameter"] = display_df["Parameter"].map(
        lambda x: readable_name(x)
    )
    display_df["Value"] = display_df.apply(
        lambda row: format_display_value(row["Parameter"], row["Value"]),
        axis=1
    )
    display_df["Value"] = display_df["Value"].astype(str)
    return display_df


def build_raw_input_dataframe() -> pd.DataFrame:
    st.sidebar.header("Applicant Information")

    gender = st.sidebar.selectbox("Gender", ["F", "M"])
    income_type = st.sidebar.selectbox(
        "Income Type",
        ["Working", "Commercial associate", "Pensioner", "State servant", "Student"]
    )
    education = st.sidebar.selectbox(
        "Education Level",
        ["Higher education", "Secondary / secondary special", "Incomplete higher"]
    )
    suite_type = st.sidebar.selectbox(
        "Application Companion",
        ["Family", "Unaccompanied", "Spouse, partner"]
    )
    house_type = st.sidebar.selectbox(
        "Housing Type",
        ["block of flats", "house / apartment"]
    )

    amt_income_total = st.sidebar.number_input(
        "Annual Income", min_value=0.0, value=200000.0, step=10000.0
    )
    amt_credit = st.sidebar.number_input(
        "Requested Loan Amount", min_value=1.0, value=500000.0, step=10000.0
    )
    amt_annuity = st.sidebar.number_input(
        "Loan Annuity", min_value=0.0, value=25000.0, step=1000.0
    )
    amt_goods_price = st.sidebar.number_input(
        "Goods Price", min_value=1.0, value=450000.0, step=10000.0
    )

    cnt_children = st.sidebar.number_input(
        "Number of Children", min_value=0, value=0, step=1
    )
    cnt_fam_members = st.sidebar.number_input(
        "Number of Family Members", min_value=1, value=2, step=1
    )

    age_years = st.sidebar.number_input(
        "Age (Years)", min_value=18, max_value=100, value=33
    )
    employment_years = st.sidebar.number_input(
        "Employment Length (Years)", min_value=0, max_value=50, value=5
    )

    ext_source_1 = st.sidebar.slider("External Credit Score 1", 0.0, 1.0, 0.4)
    ext_source_2 = st.sidebar.slider("External Credit Score 2", 0.0, 1.0, 0.5)
    ext_source_3 = st.sidebar.slider("External Credit Score 3", 0.0, 1.0, 0.6)

    amt_req_credit_bureau_year = st.sidebar.number_input(
        "Credit Inquiries in Last Year", min_value=0.0, value=1.0, step=1.0
    )
    obs_30_cnt_social_circle = st.sidebar.number_input(
        "Defaults in Social Circle (30 Days)", min_value=0.0, value=0.0, step=1.0
    )

    emergency_unknown = st.sidebar.selectbox(
        "Emergency Information Available?", ["Yes", "No"]
    )

    days_birth = -age_years * 365
    days_employed = -employment_years * 365

    raw = {
        "AMT_INCOME_TOTAL": amt_income_total,
        "AMT_CREDIT": amt_credit,
        "AMT_ANNUITY": amt_annuity,
        "AMT_GOODS_PRICE": amt_goods_price,
        "CNT_CHILDREN": cnt_children,
        "CNT_FAM_MEMBERS": cnt_fam_members,
        "DAYS_BIRTH": days_birth,
        "DAYS_EMPLOYED": days_employed,
        "EXT_SOURCE_1": ext_source_1,
        "EXT_SOURCE_2": ext_source_2,
        "EXT_SOURCE_3": ext_source_3,
        "AMT_REQ_CREDIT_BUREAU_YEAR": amt_req_credit_bureau_year,
        "OBS_30_CNT_SOCIAL_CIRCLE": obs_30_cnt_social_circle,
        "CODE_GENDER_M": 1 if gender == "M" else 0,
        "NAME_INCOME_TYPE_Working": 1 if income_type == "Working" else 0,
        "NAME_INCOME_TYPE_Commercial_associate": 1 if income_type == "Commercial associate" else 0,
        "NAME_EDUCATION_TYPE_Secondary_secondary_special": 1 if education == "Secondary / secondary special" else 0,
        "NAME_EDUCATION_TYPE_Higher_education": 1 if education == "Higher education" else 0,
        "NAME_TYPE_SUITE_Family": 1 if suite_type == "Family" else 0,
        "NAME_TYPE_SUITE_Unaccompanied": 1 if suite_type == "Unaccompanied" else 0,
        "HOUSETYPE_MODE_block_of_flats": 1 if house_type == "block of flats" else 0,
        "EMERGENCYSTATE_MODE_Unknown": 1 if emergency_unknown == "No" else 0,
    }

    raw["income_to_credit_ratio"] = raw["AMT_INCOME_TOTAL"] / (raw["AMT_CREDIT"] + 1)
    raw["annuity_to_income_ratio"] = raw["AMT_ANNUITY"] / (raw["AMT_INCOME_TOTAL"] + 1)
    raw["credit_to_goods_ratio"] = raw["AMT_CREDIT"] / (raw["AMT_GOODS_PRICE"] + 1)
    raw["children_ratio"] = raw["CNT_CHILDREN"] / (raw["CNT_FAM_MEMBERS"] + 1)
    raw["employment_to_age_ratio"] = raw["DAYS_EMPLOYED"] / (raw["DAYS_BIRTH"] + 1)
    raw["ext_source_mean"] = np.mean([
        raw["EXT_SOURCE_1"],
        raw["EXT_SOURCE_2"],
        raw["EXT_SOURCE_3"]
    ])

    return pd.DataFrame([raw])


def align_features(user_input_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    aligned = pd.DataFrame(0.0, index=[0], columns=feature_columns)
    common_cols = [col for col in user_input_df.columns if col in feature_columns]
    aligned.loc[0, common_cols] = user_input_df.loc[0, common_cols].values
    return aligned


def get_risk_label(prob: float):
    if prob < 0.2:
        return "Low Risk", "Approve"
    if prob < 0.4:
        return "Medium Risk", "Manual Review"
    return "High Risk", "Reject"


def explain_prediction(model, model_input: pd.DataFrame):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(model_input)

    if isinstance(shap_values, list):
        shap_row = shap_values[1][0]
        base_value = explainer.expected_value[1]
    else:
        shap_row = shap_values[0]
        base_value = explainer.expected_value

    return shap_row, base_value


def build_reason_table(model_input: pd.DataFrame, shap_row: np.ndarray) -> pd.DataFrame:
    features = model_input.iloc[0]

    reason_df = pd.DataFrame({
        "Feature": model_input.columns,
        "SHAP Impact": shap_row,
        "Applicant Value": features.values,
    })

    reason_df["Abs Impact"] = reason_df["SHAP Impact"].abs()
    reason_df = reason_df[reason_df["Feature"].isin(important_features)]
    reason_df = reason_df[
        (reason_df["Applicant Value"] != 0) | (reason_df["Abs Impact"] > 1e-6)
    ]

    reason_df["Readable Feature"] = reason_df["Feature"].apply(readable_name)
    reason_df["Applicant Value"] = reason_df.apply(
        lambda row: prettify_feature_value(row["Feature"], row["Applicant Value"]),
        axis=1
    )
    reason_df["SHAP Impact"] = reason_df["SHAP Impact"].round(4)
    reason_df = reason_df.sort_values("Abs Impact", ascending=False).head(8)

    return reason_df


def plot_feature_impact_chart(reason_df: pd.DataFrame):
    chart_df = reason_df.copy().sort_values("SHAP Impact", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#d62728" if x > 0 else "#2ca02c" for x in chart_df["SHAP Impact"]]

    ax.barh(chart_df["Readable Feature"], chart_df["SHAP Impact"], color=colors)
    ax.axvline(0, color="black", linewidth=1)

    ax.set_title("Top Feature Impacts on Default Risk")
    ax.set_xlabel("SHAP Impact")
    ax.set_ylabel("")

    plt.tight_layout()
    return fig


# -----------------------------
# App UI
# -----------------------------
st.title("Strategic Credit Gap Predictor")
st.caption("Alternative-data-inspired credit risk screening dashboard")

input_df = build_raw_input_dataframe()
model_input = align_features(input_df.copy(), feature_columns)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Applicant Summary")
    display_df = make_display_dataframe(input_df)
    st.dataframe(display_df, width="stretch", hide_index=True)

with col2:
    st.subheader("Prediction")

    prob = model.predict_proba(model_input)[0, 1]
    pred = int(prob >= threshold)
    risk_label, action = get_risk_label(prob)

    st.metric("Default Probability", f"{prob:.2%}")
    st.metric("Decision Threshold", f"{threshold:.2f}")
    st.metric("Recommended Action", action)
    st.metric("Risk Category", risk_label)

    if pred == 1:
        st.error("Applicant is predicted as risky based on current threshold.")
    else:
        st.success("Applicant is predicted as relatively safer based on current threshold.")

# -----------------------------
# Explanation
# -----------------------------
st.subheader("Why this decision was made")

try:
    shap_row, base_value = explain_prediction(model, model_input)
    reason_df = build_reason_table(model_input, shap_row)

    if len(reason_df) > 0:
        table_df = reason_df[["Readable Feature", "Applicant Value", "SHAP Impact"]].copy()
        table_df["Applicant Value"] = table_df["Applicant Value"].astype(str)
        table_df["SHAP Impact"] = table_df["SHAP Impact"].astype(str)

        st.write("Top factors influencing this applicant's prediction:")
        st.dataframe(table_df, width="stretch", hide_index=True)

        st.caption(
            "Positive SHAP impact increases default risk. Negative SHAP impact reduces default risk."
        )

        st.subheader("Feature Impact Chart")
        fig = plot_feature_impact_chart(reason_df)
        st.pyplot(fig)

    else:
        st.info("No strong feature contributions were found for the current applicant input.")

except Exception as e:
    st.warning("Explanation could not be generated for this input.")
    st.code(str(e))