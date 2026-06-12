# Loan Default Prediction API

API para predicción de default usando un modelo XGBoost entrenado previamente.

## Estructura

- `app.py`: API FastAPI.
- `streamlit/streamlit_app.py`: interfaz web local.
- `loan_default_xgboost_final.pkl`: pipeline entrenado.
- `Dockerfile`: contenedor para Cloud Run.
- `cloudbuild.yaml`: build y deploy en GCP.

## Ejecutar API local

```bash
pip install -r requirements.txt
uvicorn app:app --reload
