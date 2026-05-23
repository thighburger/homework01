from functools import lru_cache

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

try:
    from app.config import LOCAL_MODEL_PATH, MLFLOW_TRACKING_URI, MODEL_URI
except ModuleNotFoundError:
    from config import LOCAL_MODEL_PATH, MLFLOW_TRACKING_URI, MODEL_URI


@lru_cache(maxsize=1)
def load_model():
    if MODEL_URI:
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_registry_uri(MLFLOW_TRACKING_URI)
            return mlflow.sklearn.load_model(MODEL_URI)
        except Exception:
            pass

    return joblib.load(LOCAL_MODEL_PATH)


@lru_cache(maxsize=1)
def get_model_info():
    if MODEL_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_registry_uri(MLFLOW_TRACKING_URI)
        try:
            info = mlflow.models.get_model_info(MODEL_URI)
            run = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI).get_run(info.run_id)
            return {
                "run_id": info.run_id,
                "model_type": run.data.params.get("model_type"),
                "test_accuracy": run.data.metrics.get("test_accuracy"),
            }
        except Exception:
            pass

    return {
        "run_id": "local-fallback",
        "model_type": "joblib",
        "test_accuracy": None,
    }
