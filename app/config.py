import os

MODEL_MODE = os.getenv("MODEL_MODE", "ml").lower()
LOCAL_MODEL_PATH = os.getenv(
    "LOCAL_MODEL_PATH",
    "ml/artifacts/spam_model.joblib",
)
