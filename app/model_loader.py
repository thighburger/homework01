from functools import lru_cache

import joblib
import mlflow
import mlflow.sklearn

try:
    from app.config import LOCAL_MODEL_PATH, MLFLOW_TRACKING_URI, MODEL_URI
except ModuleNotFoundError:
    from config import LOCAL_MODEL_PATH, MLFLOW_TRACKING_URI, MODEL_URI


@lru_cache(maxsize=1)
def load_model():
    if MODEL_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        return mlflow.sklearn.load_model(MODEL_URI)

    return joblib.load(LOCAL_MODEL_PATH)
