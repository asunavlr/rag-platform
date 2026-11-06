from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_organization_bootstrap_and_document_list():
    created = client.post("/v1/organizations", json={"name": "Test Workspace"})
    assert created.status_code == 201
    payload = created.json()
    headers = {
        "Authorization": f"Bearer {payload['access_token']}",
        "X-Organization-ID": payload["organization_id"],
    }
    response = client.get("/v1/documents", headers=headers)
    assert response.status_code == 200
    assert response.json() == []
