"""FastAPI application entrypoint."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .service import PredictionService

app = FastAPI(title="Student Job Prediction API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

prediction_service = PredictionService()


@app.get("/")
def home() -> dict[str, str]:
    return {"message": "API is running"}


@app.post("/predict")
def predict(data: dict) -> dict:
    return prediction_service.predict(data)
