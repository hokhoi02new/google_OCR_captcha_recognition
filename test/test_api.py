from fastapi.testclient import TestClient
from app.app import app, startup_event
import pytest

client = TestClient(app)

@pytest.fixture(scope="session", autouse=True)
def load_model():
    import asyncio
    asyncio.run(startup_event())  # load model trước khi chạy test
    yield

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict_single():
    with open("captcha_images_v2/2b827.png", "rb") as f:
        response = client.post("/predict", files={"file": ("2b827.png", f, "image/png")})
    assert response.status_code == 200
    result = response.json()
    assert "captcha_text" in result
    print("Predict result:", result)

def test_predict_batch():
    files = [
        ("files", ("2b827.png", open("captcha_images_v2/2b827.png", "rb"), "image/png")),
        ("files", ("2bg48.png", open("captcha_images_v2/2bg48.png", "rb"), "image/png")),
    ]
    response = client.post("/predict_batch", files=files)
    assert response.status_code == 200
    result = response.json()
    assert "results" in result
    print("Batch result:", result)

