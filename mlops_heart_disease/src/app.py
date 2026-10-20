from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import numpy as np
import logging
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, generate_latest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("heart_disease_api")

# Prometheus metrics
prediction_counter = Counter("predictions_total", "Total predictions made")
confidence_histogram = Histogram("prediction_confidence", "Confidence scores")

app = FastAPI()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response


# Load pipeline
pipeline = joblib.load("models/heart_disease_pipeline.pkl")


# Define request schema
class PatientData(BaseModel):
    features: list[float]


@app.post("/predict")
def predict(data: PatientData):
    sample = np.array([data.features])
    prediction = pipeline.predict(sample)[0]
    probability = pipeline.predict_proba(sample)[0][prediction]
    prediction_counter.inc()
    confidence = float(probability)
    confidence_histogram.observe(confidence)
    return {
        "prediction": int(prediction),
        "confidence": confidence
    }


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
