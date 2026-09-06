"""Unified Async LLM Client: OpenAI y Anthropic bajo una interfaz común y asíncrona.

Uso básico:

    from llm_client import AsyncLLMManager, ChatMessage, ModelConfig

    manager = AsyncLLMManager(provider="anthropic")
    response = await manager.generate([ChatMessage(role="user", content="Hola")])
    async for chunk in manager.stream([ChatMessage(role="user", content="Hola")]):
        print(chunk, end="")
"""

from .anthropic_client import AnthropicClient
from .base import BaseLLMClient
from .errors import (
    LLMAuthError,
    LLMConfigError,
    LLMError,
    LLMNetworkError,
    LLMProviderError,
    LLMRateLimitError,
    LLMServerError,
)
from .manager import AsyncLLMManager
from .openai_client import OpenAIClient
from .schemas import ChatMessage, ModelConfig, ModelResponse

__all__ = [
    "AsyncLLMManager",
    "BaseLLMClient",
    "OpenAIClient",
    "AnthropicClient",
    "ChatMessage",
    "ModelConfig",
    "ModelResponse",
    "LLMError",
    "LLMConfigError",
    "LLMAuthError",
    "LLMRateLimitError",
    "LLMNetworkError",
    "LLMServerError",
    "LLMProviderError",
]
