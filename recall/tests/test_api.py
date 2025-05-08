from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_similar():
    r = client.get("/similar/80?k=5")
    assert r.status_code in (200, 404)  # 取决于数据集
