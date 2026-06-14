import joblib
import pandas as pd
import shap

from fastapi import FastAPI
from pydantic import BaseModel

# =====================================================
# LOAD MODEL
# =====================================================

MODEL_PATH = "loan_default_xgboost_final.pkl"

model = joblib.load(MODEL_PATH)

preprocessor = model.named_steps["preprocessor"]
xgb_model = model.named_steps["classifier"]

explainer = shap.TreeExplainer(xgb_model)

app = FastAPI(
    title="Loan Default Prediction API"
)

# =====================================================
# INPUT SCHEMA
# =====================================================

class LoanApplication(BaseModel):

    loan_amount: float
    term: float
    property_value: float
    income: float
    Credit_Score: float
    LTV: float
    dtir1: float

    loan_purpose: str
    occupancy_type: str
    business_or_commercial: str
    open_credit: str
    Credit_Worthiness: str
    loan_type: str
    Neg_ammortization: str
    lump_sum_payment: str
    co_applicant_credit_type: str
    approv_in_adv: str


# =====================================================
# HEALTH CHECK
# =====================================================

@app.get("/")
def health_check():

    return {
        "status": "ok",
        "message": "Loan Default API is running"
    }


# =====================================================
# PREDICT
# =====================================================

@app.post("/predict")
def predict(application: LoanApplication):

    input_df = pd.DataFrame(
        [application.model_dump()]
    )

    input_df = input_df.rename(
        columns={
            "co_applicant_credit_type":
            "co-applicant_credit_type"
        }
    )

    # ==========================================
    # PREDICTION
    # ==========================================

    probability_default = (
        model.predict_proba(input_df)[:, 1][0]
    )

    prediction = int(
        probability_default >= 0.50
    )

    if probability_default >= 0.60:

        risk_label = "High Risk"

    elif probability_default >= 0.30:

        risk_label = "Medium Risk"

    else:

        risk_label = "Low Risk"

    # ==========================================
    # SHAP EXPLANATION
    # ==========================================

    top_features = []

    try:

        X_transformed = preprocessor.transform(
            input_df
        )

        # convertir sparse a dense si aplica
        if hasattr(
            X_transformed,
            "toarray"
        ):
            X_dense = X_transformed.toarray()
        else:
            X_dense = X_transformed

        shap_values = explainer.shap_values(
            X_dense
        )

        feature_names = (
            preprocessor
            .get_feature_names_out()
        )
        
        clean_feature_names = []
        
        for feature in feature_names:
        
            clean_name = (
                feature
                .replace("num__", "")
                .replace("cat__", "")
            )
        
            clean_feature_names.append(
                clean_name
            )
        
        feature_values = X_dense[0]
        
        shap_df = pd.DataFrame({
            "feature": clean_feature_names,
            "shap_value": shap_values[0],
            "feature_value": feature_values
        })

        shap_df = shap_df[
            ~(
                shap_df["feature"].str.contains("_")
                &
                (shap_df["feature_value"] == 0)
            )
        ]

        friendly_names = {
        
            "LTV":
                "Loan-To-Value Ratio",
        
            "income":
                "Income",
        
            "property_value":
                "Property Value",
        
            "loan_amount":
                "Loan Amount",
        
            "Credit_Score":
                "Credit Score",
        
            "dtir1":
                "Debt-To-Income Ratio",
        
            "loan_type_type1":
                "Loan Type = Type 1",
        
            "loan_type_type2":
                "Loan Type = Type 2",
        
            "loan_type_type3":
                "Loan Type = Type 3",
        
            "loan_purpose_p1":
                "Loan Purpose = P1",
        
            "loan_purpose_p2":
                "Loan Purpose = P2",
        
            "loan_purpose_p3":
                "Loan Purpose = P3",
        
            "loan_purpose_p4":
                "Loan Purpose = P4",
        
            "occupancy_type_pr":
                "Primary Residence",
        
            "occupancy_type_ir":
                "Investment Property",
        
            "occupancy_type_sr":
                "Secondary Residence",
        
            "co-applicant_credit_type_CIB":
                "Co-Applicant Credit Type = CIB",
        
            "co-applicant_credit_type_EXP":
                "Co-Applicant Credit Type = EXP",
        
            "approv_in_adv_pre":
                "Pre-Approved",
        
            "approv_in_adv_nopre":
                "Not Pre-Approved"
        }

        shap_df["feature"] = (
            shap_df["feature"]
            .replace(friendly_names)
        )
        
        shap_df["abs_shap"] = (
            shap_df["shap_value"]
            .abs()
        )

        shap_df = (
            shap_df
            .sort_values(
                "abs_shap",
                ascending=False
            )
        )

        top_features = (
            shap_df[
                ["feature", "shap_value"]
            ]
            .head(10)
            .to_dict(
                orient="records"
            )
        )

    except Exception as e:

        top_features = [
            {
                "feature": "SHAP_ERROR",
                "shap_value": str(e)
            }
        ]

    # ==========================================
    # RESPONSE
    # ==========================================

    return {
        "prediction": prediction,
        "probability_default": round(
            float(probability_default),
            4
        ),
        "risk_label": risk_label,
        "top_features": top_features
    }
