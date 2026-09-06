"""Contrato común de todos los clientes de LLM."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import ClassVar

from .schemas import ChatMessage, ModelConfig, ModelResponse, Provider


class BaseLLMClient(ABC):
    """Interfaz que todo proveedor implementa.

    `generate` es una corrutina: devuelve la respuesta completa.
    `stream` es un generador asíncrono: entrega fragmentos de texto con `yield`.
    Son métodos separados porque una función no puede ser las dos cosas a la vez.
    """

    provider: ClassVar[Provider]

    @abstractmethod
    async def generate(
        self,
        messages: list[ChatMessage],
        config: ModelConfig | None = None,
    ) -> ModelResponse:
        """Envía la conversación y espera la respuesta completa."""

    @abstractmethod
    def stream(
        self,
        messages: list[ChatMessage],
        config: ModelConfig | None = None,
    ) -> AsyncIterator[str]:
        """Envía la conversación y entrega el texto por fragmentos, a medida que llega."""
