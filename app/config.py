import os

MODEL_MODE = os.getenv("MODEL_MODE", "ml").lower()
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_URI = os.getenv("MODEL_URI", "models:/spam-model@champion")
LOCAL_MODEL_PATH = os.getenv(
    "LOCAL_MODEL_PATH",
    "ml/artifacts/spam_model.joblib",
)

TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
MODEL_NAME = "spam_model.joblib"
ARTIFACT_DIR_NAME = "artifacts"
DATA_DIR_NAME = "data"
