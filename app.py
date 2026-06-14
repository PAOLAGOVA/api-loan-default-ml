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

        shap_df = pd.DataFrame({
            "feature": feature_names,
            "shap_value": shap_values[0]
        })

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
