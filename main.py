from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from utils import REV_YN_MAP

MODEL_PATH = "./model/telco_churn_pipeline.joblib"

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    model = joblib.load(MODEL_PATH)
    print(f"Loaded model: {type(model).__name__}")

    yield

    model = None


app = FastAPI(
    title="Churn Prediction API",
    lifespan=lifespan,
)


class Customer(BaseModel):
    customerID: str
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


class PredictionResponse(BaseModel):
    prediction: str
    churn_probability: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: Customer):
    data = customer.model_dump()
    data.pop("customerID")
    X = pd.DataFrame([data])

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0, 1]

    return {
        "prediction": REV_YN_MAP[int(prediction)],
        "churn_probability": float(probability),
    }
