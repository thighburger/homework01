import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline


BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.getenv("SPAM_DATA_PATH", os.path.join(BASE_DIR, "data", "spam.csv"))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "spam_model.joblib")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT_NAME = "spam-classification-local"
REGISTERED_MODEL_NAME = "spam-model"
MODEL_ALIAS = "champion"

os.makedirs(ARTIFACT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

X = df["text"]
y = df["label"]

pipeline = Pipeline(
    [
        ("vectorizer", CountVectorizer()),
        ("classifier", LogisticRegression(max_iter=200)),
    ]
)

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run() as run:
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("vectorizer", "CountVectorizer")
    mlflow.log_param("max_iter", 200)
    mlflow.log_param("data_path", DATA_PATH)
    mlflow.log_param("row_count", len(df))

    pipeline.fit(X, y)

    preds = pipeline.predict(X)
    acc = accuracy_score(y, preds)
    mlflow.log_metric("train_accuracy", acc)

    joblib.dump(pipeline, MODEL_PATH)

    mlflow.log_artifact(DATA_PATH)
    mlflow.log_artifact(MODEL_PATH)
    mlflow.sklearn.log_model(
        pipeline,
        name="model",
        registered_model_name=REGISTERED_MODEL_NAME,
    )

    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    latest_version = max(int(version.version) for version in versions)
    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME,
        MODEL_ALIAS,
        str(latest_version),
    )

print(f"Model saved to: {MODEL_PATH}")
print(f"train_accuracy: {acc:.4f}")
print(f"run_id: {run.info.run_id}")
print(f"model_uri: models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}")
