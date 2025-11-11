"""LLM interface with support for multiple providers."""

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


class OpenAILLM(BaseLLM):
    """OpenAI LLM implementation (placeholder for future integration)."""

    async def complete(self, prompt: str, context: list[Chunk]) -> str:
        """Generate completion using OpenAI API."""
        # TODO: Implement OpenAI integration
        # from openai import AsyncOpenAI
        # client = AsyncOpenAI(api_key=settings.llm_api_key)
        # response = await client.chat.completions.create(...)
        raise NotImplementedError("OpenAI integration not yet implemented")


class AnthropicLLM(BaseLLM):
    """Anthropic Claude implementation (placeholder for future integration)."""

    async def complete(self, prompt: str, context: list[Chunk]) -> str:
        """Generate completion using Anthropic API."""
        # TODO: Implement Anthropic integration
        # from anthropic import AsyncAnthropic
        # client = AsyncAnthropic(api_key=settings.llm_api_key)
        # response = await client.messages.create(...)
        raise NotImplementedError("Anthropic integration not yet implemented")


def get_llm() -> MockLLM | OpenAILLM | AnthropicLLM:
    """Get LLM instance based on configuration."""
    settings = get_settings()

    llm_providers: dict[str, type[BaseLLM]] = {
        "mock": MockLLM,
        "openai": OpenAILLM,
        "anthropic": AnthropicLLM,
    }

    provider_class = llm_providers.get(settings.llm_provider, MockLLM)
    if provider_class is None or settings.llm_provider not in llm_providers:
        logger.warning(
            f"Unknown LLM provider: {settings.llm_provider}, falling back to mock"
        )
        provider_class = MockLLM

    logger.info(f"Initialized LLM provider: {settings.llm_provider}")
    return provider_class()
