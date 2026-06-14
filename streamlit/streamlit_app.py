import streamlit as st
import requests

API_URL = "https://loan-default-api-449555437256.europe-west1.run.app/predict"
# Cuando despliegues:
# API_URL = "https://TU_URL_CLOUD_RUN/predict"

st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Loan Default Risk Prediction")

st.markdown(
    """
    Complete the loan application information and obtain
    the estimated probability of default.
    """
)

# =====================================================
# Loan Information
# =====================================================

st.header("Loan Information")

col1, col2 = st.columns(2)

with col1:

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0.0,
        value=150000.0,
        step=1000.0
    )

    property_value = st.number_input(
        "Property Value",
        min_value=1.0,
        value=250000.0,
        step=1000.0
    )

    term = st.number_input(
        "Term (months)",
        min_value=1.0,
        value=360.0
    )

    loan_type = st.selectbox(
        "Loan Type",
        ["type1", "type2", "type3"]
    )

    loan_purpose = st.selectbox(
        "Loan Purpose",
        ["p1", "p2", "p3", "p4"]
    )

with col2:

    income = st.number_input(
        "Income",
        min_value=0.0,
        value=6000.0
    )

    Credit_Score = st.number_input(
        "Credit Score",
        min_value=300.0,
        max_value=850.0,
        value=700.0
    )

    dtir1 = st.number_input(
        "Debt-To-Income Ratio",
        min_value=0.0,
        max_value=100.0,
        value=35.0
    )

    # LTV calculado automáticamente
    LTV = round(
        (loan_amount / property_value) * 100,
        2
    )

    st.metric(
        "Calculated LTV (%)",
        f"{LTV:.2f}"
    )

# =====================================================
# Credit Information
# =====================================================

st.header("Credit Information")

col1, col2 = st.columns(2)

with col1:

    Credit_Worthiness = st.selectbox(
        "Credit Worthiness",
        ["l1", "l2"]
    )

    open_credit = st.selectbox(
        "Open Credit",
        ["nopc", "opc"]
    )

    co_applicant_credit_type = st.selectbox(
        "Co-Applicant Credit Type",
        ["CIB", "EXP"]
    )

with col2:

    approv_in_adv = st.selectbox(
        "Approved In Advance",
        ["pre", "nopre"]
    )

    business_or_commercial = st.selectbox(
        "Business or Commercial",
        ["nob/c", "b/c"]
    )

# =====================================================
# Loan Structure
# =====================================================

st.header("Loan Structure")

col1, col2 = st.columns(2)

with col1:

    Neg_ammortization = st.selectbox(
        "Negative Amortization",
        ["not_neg", "neg_amm"]
    )

with col2:

    lump_sum_payment = st.selectbox(
        "Lump Sum Payment",
        ["not_lpsm", "lpsm"]
    )

# =====================================================
# Property Information
# =====================================================

st.header("Property Information")

occupancy_type = st.selectbox(
    "Occupancy Type",
    ["pr", "ir", "sr"]
)

# =====================================================
# Prediction
# =====================================================

if st.button("Predict Default Risk"):

    payload = {
        "loan_amount": loan_amount,
        "term": term,
        "property_value": property_value,
        "income": income,
        "Credit_Score": Credit_Score,
        "LTV": LTV,
        "dtir1": dtir1,
        "loan_purpose": loan_purpose,
        "occupancy_type": occupancy_type,
        "business_or_commercial": business_or_commercial,
        "open_credit": open_credit,
        "Credit_Worthiness": Credit_Worthiness,
        "loan_type": loan_type,
        "Neg_ammortization": Neg_ammortization,
        "lump_sum_payment": lump_sum_payment,
        "co_applicant_credit_type": co_applicant_credit_type,
        "approv_in_adv": approv_in_adv
    }

    response = requests.post(
        API_URL,
        json=payload
    )

    if response.status_code == 200:

        result = response.json()

        st.success("Prediction completed")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Probability of Default",
                f"{result['probability_default']:.1%}"
            )

        with col2:

            st.metric(
                "Risk Level",
                result["risk_label"]
            )

        st.write(
            f"Prediction Class: **{result['prediction']}**"
        )

        st.subheader("Top Factors Influencing Prediction")
    
        features_df = pd.DataFrame(
            result["top_features"]
        )
        
        features_df["direction"] = features_df["shap_value"].apply(
            lambda x: "Increase Risk" if x > 0 else "Reduce Risk"
        )
        
        st.dataframe(
            features_df,
            use_container_width=True
        )
    
        st.subheader("Risk Drivers")
    
        positive_features = [
            x for x in result["top_features"]
            if x["shap_value"] > 0
        ]
        
        negative_features = [
            x for x in result["top_features"]
            if x["shap_value"] < 0
        ]
        
        col1, col2 = st.columns(2)
        
        with col1:
        
            st.markdown("### ⚠ Factors Increasing Risk")
        
            for feature in positive_features:
        
                st.error(
                    f"{feature['feature']} "
                    f"({feature['shap_value']:.3f})"
                )
        
        with col2:
        
            st.markdown("### ✅ Factors Reducing Risk")
        
            for feature in negative_features:
        
                st.success(
                    f"{feature['feature']} "
                    f"({feature['shap_value']:.3f})"
                )
    else:

        st.error("Error calling API")
        st.write(response.text)
