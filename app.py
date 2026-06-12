import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


MODEL_PATH = "loan_default_xgboost_final.pkl"
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Loan Default Prediction API")


class LoanApplication(BaseModel):
    loan_limit: str
    Gender: str
    approv_in_adv: str
    loan_type: str
    loan_purpose: str
    Credit_Worthiness: str
    open_credit: str
    business_or_commercial: str
    loan_amount: float
    term: float
    Neg_ammortization: str
    interest_only: str
    lump_sum_payment: str
    property_value: float
    construction_type: str
    occupancy_type: str
    Secured_by: str
    total_units: str
    income: float
    Credit_Score: float
    co_applicant_credit_type: str
    age: str
    submission_of_application: str
    LTV: float
    Region: str
    Security_Type: str
    dtir1: float


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Loan Default API is running"}


@app.post("/predict")
def predict(application: LoanApplication):
    input_df = pd.DataFrame([application.model_dump()])

    input_df = input_df.rename(
        columns={
            "co_applicant_credit_type": "co-applicant_credit_type"
        }
    )

    probability_default = model.predict_proba(input_df)[:, 1][0]
    prediction = int(probability_default >= 0.50)

    if probability_default >= 0.60:
        risk_label = "High Risk"
    elif probability_default >= 0.30:
        risk_label = "Medium Risk"
    else:
        risk_label = "Low Risk"

    return {
        "prediction": prediction,
        "probability_default": round(float(probability_default), 4),
        "risk_label": risk_label
    }

