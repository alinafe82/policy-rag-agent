"""Tests for FastAPI application endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.app import app


@pytest.fixture(scope="module", autouse=True)
def initialize_app():
    """Initialize app lifespan for all tests."""
    with TestClient(app) as client:
        yield client


class TestHealthEndpoints:
    """Test health and monitoring endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "environment" in data
        assert "store_size" in data

    def test_metrics_endpoint(self, client):
        """Test metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "total_requests" in data
        assert "successful_requests" in data
        assert "failed_requests" in data
        assert "success_rate" in data

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "docs_url" in data


class TestAskEndpoint:
    """Test /ask endpoint functionality."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_ask_valid_query(self, client):
        """Test asking a valid query."""
        response = client.post(
            "/ask", json={"query": "What are the MFA requirements for administrators?"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "confidence" in data
        assert "citations_found" in data
        assert "MFA" in data["answer"] or "mfa" in data["answer"].lower()
        assert len(data["sources"]) > 0

    def test_ask_secrets_query(self, client):
        """Test asking about secrets policy."""
        response = client.post(
            "/ask", json={"query": "How should we handle secrets in git?"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "vault" in data["answer"].lower() or "secret" in data["answer"].lower()

    def test_ask_pii_query(self, client):
        """Test asking about PII policy."""
        response = client.post(
            "/ask",
            json={
                "query": "How should we handle personal data and PII in system logs?"
            },
        )
        assert response.status_code == 200
        data = response.json()
        # Should get a valid policy answer with citations
        assert len(data["answer"]) > 10
        assert len(data["sources"]) > 0
        assert data["confidence"] > 0.0

    def test_ask_empty_query(self, client):
        """Test asking with empty query."""
        response = client.post("/ask", json={"query": ""})
        assert response.status_code == 422  # Validation error

    def test_ask_query_too_short(self, client):
        """Test asking with too short query."""
        response = client.post("/ask", json={"query": "ab"})
        assert response.status_code == 422

    def test_ask_query_too_long(self, client):
        """Test asking with too long query."""
        response = client.post("/ask", json={"query": "x" * 1001})
        assert response.status_code == 422

    def test_ask_whitespace_query(self, client):
        """Test asking with whitespace-only query."""
        response = client.post("/ask", json={"query": "   "})
        assert response.status_code == 422

    def test_ask_caching(self, client):
        """Test response caching."""
        query = {"query": "What are the MFA requirements?"}

        # First request
        response1 = client.post("/ask", json=query)
        assert response1.status_code == 200
        data1 = response1.json()
        assert data1["cached"] is False

        # Second request should be cached
        response2 = client.post("/ask", json=query)
        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["cached"] is True
        assert data2["answer"] == data1["answer"]

    def test_ask_response_metadata(self, client):
        """Test response includes all required metadata."""
        response = client.post(
            "/ask", json={"query": "What are the password requirements?"}
        )
        assert response.status_code == 200
        data = response.json()

        # Check all required fields
        assert "answer" in data
        assert "sources" in data
        assert "confidence" in data
        assert "citations_found" in data
        assert "cached" in data
        assert "process_time_ms" in data

        # Check types
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["confidence"], (int | float))
        assert isinstance(data["citations_found"], int)
        assert isinstance(data["cached"], bool)
        assert isinstance(data["process_time_ms"], (int | float))

    def test_ask_process_time_header(self, client):
        """Test that process time header is added."""
        response = client.post("/ask", json={"query": "What is the training policy?"})
        assert "X-Process-Time" in response.headers
        process_time = float(response.headers["X-Process-Time"])
        assert process_time > 0

    def test_ask_invalid_json(self, client):
        """Test handling of invalid JSON."""
        response = client.post(
            "/ask", data="invalid json", headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_ask_missing_query_field(self, client):
        """Test handling of missing query field."""
        response = client.post("/ask", json={"wrong_field": "test"})
        assert response.status_code == 422
