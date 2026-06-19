"""LLM interface for the local policy-answer generator."""

from abc import ABC, abstractmethod

from .config import get_settings
from .logger import setup_logger
from .store import Chunk

logger = setup_logger(__name__)


class BaseLLM(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def complete(self, prompt: str, context: list[Chunk]) -> str:
        """Generate completion with context."""
        pass


class MockLLM(BaseLLM):
    """Mock LLM for testing and development."""

    async def complete(self, prompt: str, context: list[Chunk]) -> str:
        """Generate mock completion based on context."""
        logger.debug(f"MockLLM completing prompt: {prompt[:100]}...")

        prompt_lower = prompt.lower()

        # Smart mock responses based on content
        if "mfa" in prompt_lower or "multi-factor" in prompt_lower:
            relevant = [c for c in context if "mfa" in c.text.lower()]
            if relevant:
                return f"MFA is required for administrative access to all systems [{relevant[0].id}]."

        if "secret" in prompt_lower or "credential" in prompt_lower:
            relevant = [
                c
                for c in context
                if "secret" in c.text.lower() or "vault" in c.text.lower()
            ]
            if relevant:
                return f"Secrets must not be committed to git. Use a vault like HashiCorp Vault or AWS Secrets Manager [{relevant[0].id}]."

        if "pii" in prompt_lower or "personal" in prompt_lower:
            relevant = [c for c in context if "pii" in c.text.lower()]
            if relevant:
                return f"PII should be masked in logs and backups. Use tokenization for sensitive data [{relevant[0].id}]."

        if "training" in prompt_lower:
            relevant = [c for c in context if "training" in c.text.lower()]
            if relevant:
                return f"Employees must complete security training annually [{relevant[0].id}]."

        if "password" in prompt_lower:
            relevant = [c for c in context if "password" in c.text.lower()]
            if relevant:
                return f"Password requirements: minimum 12 characters, complexity required, 90-day rotation [{relevant[0].id}]."

        # Default response with first relevant chunk
        if context:
            return f"{context[0].text} [{context[0].id}]."

        return "No relevant policy found in the knowledge base."


def get_llm() -> BaseLLM:
    """Get LLM instance based on configuration."""
    settings = get_settings()

    llm_providers: dict[str, type[BaseLLM]] = {
        "mock": MockLLM,
    }

    if settings.llm_provider not in llm_providers:
        raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")

    provider_class = llm_providers[settings.llm_provider]

    logger.info(f"Initialized LLM provider: {provider_class.__name__}")
    return provider_class()
