import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


MODEL_PATH = "loan_default_xgboost_final.pkl"
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Loan Default Prediction API")


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

    # Transformación para SHAP
    X_transformed = preprocessor.transform(input_df)
    
    # SHAP local
    shap_values = explainer.shap_values(X_transformed)
    
    feature_names = preprocessor.get_feature_names_out()
    
    shap_df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_values[0]
    })
    
    shap_df["abs_shap"] = shap_df["shap_value"].abs()
    
    shap_df = (
        shap_df
        .sort_values("abs_shap", ascending=False)
    )
    
    top_features = (
        shap_df[["feature", "shap_value"]]
        .head(10)
        .to_dict(orient="records")
    )

    if probability_default >= 0.60:
        risk_label = "High Risk"
    elif probability_default >= 0.30:
        risk_label = "Medium Risk"
    else:
        risk_label = "Low Risk"

    return {
        "prediction": prediction,
        "probability_default": round(float(probability_default), 4),
        "risk_label": risk_label,
        "top_features": top_features
    }

