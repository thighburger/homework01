import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from app.config import (
    ARTIFACT_DIR_NAME,
    DATA_DIR_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_NAME,
    TEST_FILE_NAME,
    TRAIN_FILE_NAME,
)


BASE_DIR = os.path.dirname(__file__)
TRAIN_DATA_PATH = os.path.join(BASE_DIR, DATA_DIR_NAME, TRAIN_FILE_NAME)
TEST_DATA_PATH = os.path.join(BASE_DIR, DATA_DIR_NAME, TEST_FILE_NAME)
ARTIFACT_DIR = os.path.join(BASE_DIR, ARTIFACT_DIR_NAME)
MODEL_PATH = os.path.join(ARTIFACT_DIR, MODEL_NAME)
EXPERIMENT_NAME = "spam-classification-local"
REGISTERED_MODEL_NAME = "spam-model"
MODEL_ALIAS = "champion"


def train_model(model_name, model, train_df, test_df):
    X_train = train_df["text"]
    y_train = train_df["label"]
    X_test = test_df["text"]
    y_test = test_df["label"]

    pipeline = Pipeline(
        [
            ("vectorizer", CountVectorizer()),
            ("classifier", model),
        ]
    )

    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("vectorizer", "CountVectorizer")
        mlflow.log_param("train_data_path", TRAIN_DATA_PATH)
        mlflow.log_param("test_data_path", TEST_DATA_PATH)
        mlflow.log_param("train_row_count", len(train_df))
        mlflow.log_param("test_row_count", len(test_df))

        pipeline.fit(X_train, y_train)

        train_preds = pipeline.predict(X_train)
        test_preds = pipeline.predict(X_test)
        train_acc = accuracy_score(y_train, train_preds)
        test_acc = accuracy_score(y_test, test_preds)
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy", test_acc)

        run_model_path = os.path.join(ARTIFACT_DIR, f"{model_name}_spam_model.joblib")
        joblib.dump(pipeline, run_model_path)
        mlflow.log_artifact(TRAIN_DATA_PATH, artifact_path="data")
        mlflow.log_artifact(TEST_DATA_PATH, artifact_path="data")
        mlflow.log_artifact(run_model_path, artifact_path="models")
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        return {
            "run_id": run.info.run_id,
            "model_name": model_name,
            "pipeline": pipeline,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
        }


def find_registered_version(client, run_id):
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    matching_versions = [
        int(version.version) for version in versions if version.run_id == run_id
    ]
    if matching_versions:
        return str(max(matching_versions))

    all_versions = [int(version.version) for version in versions]
    return str(max(all_versions))


def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=200),
        "NaiveBayes": MultinomialNB(),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_registry_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    results = [
        train_model(model_name, model, train_df, test_df)
        for model_name, model in models.items()
    ]
    best_result = max(
        results,
        key=lambda result: (result["test_accuracy"], result["train_accuracy"]),
    )

    joblib.dump(best_result["pipeline"], MODEL_PATH)
    client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    best_version = find_registered_version(client, best_result["run_id"])
    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME,
        MODEL_ALIAS,
        best_version,
    )

    print(f"Model saved to: {MODEL_PATH}")
    print(f"best_model: {best_result['model_name']}")
    print(f"train_accuracy: {best_result['train_accuracy']:.4f}")
    print(f"test_accuracy: {best_result['test_accuracy']:.4f}")
    print(f"run_id: {best_result['run_id']}")
    print(f"model_uri: models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}")


if __name__ == "__main__":
    main()
