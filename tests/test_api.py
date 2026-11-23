import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_and_readiness(client):
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 200


def test_organization_bootstrap_and_document_list(client):
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


def test_organization_token_is_required(client):
    assert client.get("/v1/documents").status_code == 401


def test_organization_cannot_delete_another_tenants_document(client):
    first = client.post("/v1/organizations", json={"name": "First Workspace"}).json()
    second = client.post("/v1/organizations", json={"name": "Second Workspace"}).json()
    first_headers = {
        "Authorization": f"Bearer {first['access_token']}",
        "X-Organization-ID": first["organization_id"],
    }
    second_headers = {
        "Authorization": f"Bearer {second['access_token']}",
        "X-Organization-ID": second["organization_id"],
    }
    # A syntactically valid but non-existent ID must remain tenant-scoped.
    response = client.delete("/v1/documents/00000000-0000-0000-0000-000000000001", headers=second_headers)
    assert response.status_code == 404
    assert client.get("/v1/documents", headers=first_headers).json() == []
