import joblib

model = joblib.load("loan_default_xgboost_final.pkl")

print("Model loaded successfully")
print(type(model))
