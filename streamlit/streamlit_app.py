import streamlit as st
import requests


API_URL = "http://127.0.0.1:8000/predict"
# Cuando despliegues en Cloud Run, cambia por:
# API_URL = "https://TU_URL_DE_CLOUD_RUN/predict"


st.title("Loan Default Risk Prediction")

st.subheader("Loan Information")

loan_limit = st.selectbox("Loan Limit", ["cf", "ncf"])
approv_in_adv = st.selectbox("Approved in Advance", ["pre", "nopre"])
loan_type = st.selectbox("Loan Type", ["type1", "type2", "type3"])
loan_purpose = st.selectbox("Loan Purpose", ["p1", "p2", "p3", "p4"])
loan_amount = st.number_input("Loan Amount", min_value=0.0)
term = st.number_input("Term", min_value=0.0)
property_value = st.number_input("Property Value", min_value=0.0)
LTV = st.number_input("LTV", min_value=0.0)
dtir1 = st.number_input("DTI Ratio", min_value=0.0)

st.subheader("Applicant Information")

Gender = st.selectbox("Gender", ["Male", "Female", "Joint", "Sex Not Available"])
age = st.selectbox("Age", ["<25", "25-34", "35-44", "45-54", "55-64", "65-74", ">74"])
income = st.number_input("Income", min_value=0.0)
Credit_Score = st.number_input("Credit Score", min_value=300.0, max_value=850.0)

st.subheader("Credit Information")

Credit_Worthiness = st.selectbox("Credit Worthiness", ["l1", "l2"])
open_credit = st.selectbox("Open Credit", ["nopc", "opc"])
business_or_commercial = st.selectbox("Business or Commercial", ["nob/c", "b/c"])
Neg_ammortization = st.selectbox("Negative Amortization", ["not_neg", "neg_amm"])
interest_only = st.selectbox("Interest Only", ["not_int", "int_only"])
lump_sum_payment = st.selectbox("Lump Sum Payment", ["not_lpsm", "lpsm"])
co_applicant_credit_type = st.selectbox("Co-applicant Credit Type", ["CIB", "EXP"])

st.subheader("Property Information")

construction_type = st.selectbox("Construction Type", ["sb", "mh"])
occupancy_type = st.selectbox("Occupancy Type", ["pr", "ir", "sr"])
Secured_by = st.selectbox("Secured By", ["home", "land"])
total_units = st.selectbox("Total Units", ["1U", "2U", "3U", "4U"])
submission_of_application = st.selectbox("Submission of Application", ["to_inst", "not_inst"])
Region = st.selectbox("Region", ["North", "south", "central", "North-East"])
Security_Type = st.selectbox("Security Type", ["direct", "Indriect"])

if st.button("Predict Default Risk"):

    payload = {
        "loan_limit": loan_limit,
        "Gender": Gender,
        "approv_in_adv": approv_in_adv,
        "loan_type": loan_type,
        "loan_purpose": loan_purpose,
        "Credit_Worthiness": Credit_Worthiness,
        "open_credit": open_credit,
        "business_or_commercial": business_or_commercial,
        "loan_amount": loan_amount,
        "term": term,
        "Neg_ammortization": Neg_ammortization,
        "interest_only": interest_only,
        "lump_sum_payment": lump_sum_payment,
        "property_value": property_value,
        "construction_type": construction_type,
        "occupancy_type": occupancy_type,
        "Secured_by": Secured_by,
        "total_units": total_units,
        "income": income,
        "Credit_Score": Credit_Score,
        "co_applicant_credit_type": co_applicant_credit_type,
        "age": age,
        "submission_of_application": submission_of_application,
        "LTV": LTV,
        "Region": Region,
        "Security_Type": Security_Type,
        "dtir1": dtir1
    }

    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        result = response.json()

        st.metric(
            "Probability of Default",
            f"{result['probability_default']:.1%}"
        )

        st.write(f"Risk Level: **{result['risk_label']}**")
        st.write(f"Prediction: **{result['prediction']}**")

    else:
        st.error("Error calling prediction API")
        st.write(response.text)
