"""Tests for LLM functionality."""

import pytest

from src.llm import AnthropicLLM, MockLLM, OpenAILLM, get_llm
from src.store import Chunk


class TestMockLLM:
    """Test MockLLM implementation."""

    @pytest.fixture
    def llm(self):
        """Create MockLLM instance."""
        return MockLLM()

    @pytest.fixture
    def mfa_context(self):
        """MFA-related context."""
        return [
            Chunk("IT-101", "MFA is required for administrative access to all systems.")
        ]

    @pytest.fixture
    def secrets_context(self):
        """Secrets-related context."""
        return [Chunk("SEC-007", "Secrets must not be committed to git. Use a vault.")]

    async def test_mock_llm_mfa_query(self, llm, mfa_context):
        """Test MockLLM handles MFA queries."""
        response = await llm.complete("What are the MFA requirements?", mfa_context)
        assert "MFA" in response or "mfa" in response.lower()
        assert "[IT-101]" in response

    async def test_mock_llm_secrets_query(self, llm, secrets_context):
        """Test MockLLM handles secrets queries."""
        response = await llm.complete("How should we handle secrets?", secrets_context)
        assert "vault" in response.lower() or "secret" in response.lower()
        assert "[SEC-007]" in response

    async def test_mock_llm_pii_query(self, llm):
        """Test MockLLM handles PII queries."""
        context = [Chunk("PRIV-010", "PII should be masked in logs and backups.")]
        response = await llm.complete("What about PII in logs?", context)
        assert "pii" in response.lower() or "mask" in response.lower()
        assert "[PRIV-010]" in response

    async def test_mock_llm_training_query(self, llm):
        """Test MockLLM handles training queries."""
        context = [
            Chunk("HR-001", "Employees must complete security training annually.")
        ]
        response = await llm.complete("What is the training policy?", context)
        assert "training" in response.lower()
        assert "[HR-001]" in response

    async def test_mock_llm_password_query(self, llm):
        """Test MockLLM handles password queries."""
        context = [Chunk("SEC-009", "Password requirements: minimum 12 characters.")]
        response = await llm.complete("What are password requirements?", context)
        assert "password" in response.lower()
        assert "[SEC-009]" in response

    async def test_mock_llm_default_response(self, llm):
        """Test MockLLM default response with unrecognized query."""
        context = [Chunk("TEST-001", "Some test content.")]
        response = await llm.complete("Random query about something", context)
        assert "[TEST-001]" in response
        assert len(response) > 0

    async def test_mock_llm_no_context(self, llm):
        """Test MockLLM with no context."""
        response = await llm.complete("What is the policy?", [])
        assert "no relevant" in response.lower()


class TestOpenAILLM:
    """Test OpenAILLM implementation."""

    @pytest.fixture
    def llm(self):
        """Create OpenAILLM instance."""
        return OpenAILLM()

    async def test_openai_not_implemented(self, llm):
        """Test that OpenAI LLM raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            await llm.complete("test", [])


class TestAnthropicLLM:
    """Test AnthropicLLM implementation."""

    @pytest.fixture
    def llm(self):
        """Create AnthropicLLM instance."""
        return AnthropicLLM()

    async def test_anthropic_not_implemented(self, llm):
        """Test that Anthropic LLM raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            await llm.complete("test", [])


class TestGetLLM:
    """Test get_llm factory function."""

    def test_get_llm_default(self, monkeypatch):
        """Test get_llm returns MockLLM by default."""
        llm = get_llm()
        assert isinstance(llm, MockLLM)

    def test_get_llm_mock(self, monkeypatch):
        """Test get_llm with mock provider."""
        from src.config import Settings

        def mock_get_settings():
            settings = Settings()
            settings.llm_provider = "mock"
            return settings

        monkeypatch.setattr("src.llm.get_settings", mock_get_settings)
        llm = get_llm()
        assert isinstance(llm, MockLLM)

    def test_get_llm_unknown_provider(self, monkeypatch):
        """Test get_llm with unknown provider falls back to mock."""
        from src.config import Settings

        def mock_get_settings():
            settings = Settings()
            settings.llm_provider = "unknown"  # type: ignore[assignment]
            return settings

        monkeypatch.setattr("src.llm.get_settings", mock_get_settings)
        llm = get_llm()
        assert isinstance(llm, MockLLM)
