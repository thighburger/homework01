from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_classify_api_contract():
    response = client.post("/classify", json={"text": "hello"})

    assert response.status_code == 200

    data = response.json()
    assert "label" in data and "score" in data
    assert "model_info" in data
    assert "run_id" in data["model_info"]
    assert "model_type" in data["model_info"]
    assert "test_accuracy" in data["model_info"]
