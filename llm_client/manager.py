"""AsyncLLMManager: elige y construye el proveedor a partir de la configuración.

Es la única pieza del paquete que lee variables de entorno. Los clientes reciben
todo por parámetro, lo que los hace fáciles de probar.
"""

import os
from collections.abc import AsyncIterator

from .anthropic_client import AnthropicClient
from .base import BaseLLMClient
from .errors import LLMConfigError
from .openai_client import OpenAIClient
from .schemas import ChatMessage, ModelConfig, ModelResponse

DEFAULT_PROVIDER = "anthropic"
SUPPORTED_PROVIDERS = ("openai", "anthropic")


class AsyncLLMManager:
    """Punto de entrada: `AsyncLLMManager()` o `AsyncLLMManager(provider="openai")`.

    Sin argumento, usa la variable de entorno LLM_PROVIDER (default: anthropic).
    """

    def __init__(self, provider: str | None = None) -> None:
        name = (provider or os.environ.get("LLM_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
        self.client: BaseLLMClient = self._build_client(name)
        self.provider = self.client.provider

    async def generate(
        self,
        messages: list[ChatMessage],
        config: ModelConfig | None = None,
    ) -> ModelResponse:
        return await self.client.generate(messages, config)

    def stream(
        self,
        messages: list[ChatMessage],
        config: ModelConfig | None = None,
    ) -> AsyncIterator[str]:
        return self.client.stream(messages, config)

    # --- Internos ---------------------------------------------------------------

    @staticmethod
    def _require_env(var: str, provider: str) -> str:
        value = os.environ.get(var, "").strip()
        if not value:
            raise LLMConfigError(
                f"Falta la variable de entorno {var}. Copiá .env.example a .env y completala.",
                provider=provider,
            )
        return value

    def _build_client(self, name: str) -> BaseLLMClient:
        if name == "openai":
            return OpenAIClient(
                api_key=self._require_env("OPENAI_API_KEY", name),
                model=os.environ.get("OPENAI_MODEL") or OpenAIClient.DEFAULT_MODEL,
                base_url=os.environ.get("OPENAI_BASE_URL") or None,
            )
        if name == "anthropic":
            return AnthropicClient(
                api_key=self._require_env("ANTHROPIC_API_KEY", name),
                model=os.environ.get("ANTHROPIC_MODEL") or AnthropicClient.DEFAULT_MODEL,
            )
        raise LLMConfigError(
            f"Proveedor desconocido: {name!r}. Opciones: {', '.join(SUPPORTED_PROVIDERS)}.",
            provider=name,
        )
