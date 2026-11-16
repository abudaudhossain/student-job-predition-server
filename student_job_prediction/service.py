"""Model loading and prediction service."""

import joblib

from .config import MODEL_PATH, SCALER_PATH
from .preprocessing import build_model_input


class PredictionService:
    def __init__(self) -> None:
        self.model = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        model_features = getattr(self.model, "feature_names_", None)
        if model_features is None:
            model_features = getattr(self.model, "feature_names_in_", None)
        if model_features is None:
            raise RuntimeError(
                "The loaded model does not expose feature names required for preprocessing."
            )
        self.model_features = list(model_features)

    def predict(self, data: dict) -> dict:
        model_input = build_model_input(data, self.model_features, self.scaler)
        prediction = int(self.model.predict(model_input)[0])
        confidence = float(self.model.predict_proba(model_input)[0][1])
        return {
            "status": "Got a Job" if prediction == 1 else "Did Not Get a Job",
            "confidence": confidence,
        }
