"""Cliente asíncrono para la API de OpenAI y endpoints compatibles (por ejemplo Gemini)."""

from collections.abc import AsyncIterator
from typing import Any, ClassVar

import openai
from openai import AsyncOpenAI

from .base import BaseLLMClient
from .errors import LLMError, LLMProviderError, translate_sdk_error
from .retry import with_retry
from .schemas import ChatMessage, ModelConfig, ModelResponse, Provider


class OpenAIClient(BaseLLMClient):
    provider: ClassVar[Provider] = "openai"
    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str | None = None,
        client: AsyncOpenAI | None = None,
        attempts: int = 3,
        base_delay: float = 1.0,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self._attempts = attempts
        self._base_delay = base_delay
        # max_retries=0: el SDK no reintenta por su cuenta; with_retry es la única
        # lógica de reintento. `client` se inyecta en los tests.
        self._client = client or AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=0)

    # --- API pública ------------------------------------------------------------

    async def generate(
        self,
        messages: list[ChatMessage],
        config: ModelConfig | None = None,
    ) -> ModelResponse:
        config = config or ModelConfig()
        return await with_retry(
            lambda: self._generate_once(messages, config),
            attempts=self._attempts,
            base_delay=self._base_delay,
        )

    async def stream(
        self,
        messages: list[ChatMessage],
        config: ModelConfig | None = None,
    ) -> AsyncIterator[str]:
        config = config or ModelConfig()
        # Solo la apertura se reintenta: ahí aparecen los errores de auth y de cuota.
        # Reintentar a mitad de stream duplicaría texto ya entregado.
        stream = await with_retry(
            lambda: self._open_stream(messages, config),
            attempts=self._attempts,
            base_delay=self._base_delay,
        )
        try:
            async for chunk in stream:
                if not chunk.choices:  # el chunk final de uso no trae choices
                    continue
                text = chunk.choices[0].delta.content
                if text:
                    yield text
        except Exception as error:
            # El SDK no envuelve los errores de transporte durante la iteración: un
            # corte de conexión a mitad de stream llega como excepción de httpx, no
            # como openai.APIError. Se traduce igual para no filtrar excepciones crudas.
            raise self._translate(error) from error
        finally:
            await stream.close()

    # --- Internos ---------------------------------------------------------------

    def _build_request(self, messages: list[ChatMessage], config: ModelConfig) -> dict[str, Any]:
        payload = [{"role": m.role, "content": m.content} for m in messages]
        if config.system_prompt:
            # OpenAI acepta el system prompt como un mensaje más, al principio.
            payload.insert(0, {"role": "system", "content": config.system_prompt})
        request: dict[str, Any] = {
            "model": config.model or self.model,
            "messages": payload,
            "max_tokens": config.max_tokens,
        }
        if config.temperature is not None:
            request["temperature"] = config.temperature
        return request

    async def _generate_once(self, messages: list[ChatMessage], config: ModelConfig) -> ModelResponse:
        try:
            response = await self._client.chat.completions.create(
                **self._build_request(messages, config)
            )
        except openai.APIError as error:
            raise self._translate(error) from error

        if not response.choices:
            raise LLMProviderError(
                "La API no devolvió ninguna respuesta (choices vacío)", provider=self.provider
            )

        choice = response.choices[0]
        usage = response.usage
        return ModelResponse(
            content=choice.message.content or "",
            model=response.model,
            provider=self.provider,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
            finish_reason=choice.finish_reason,
        )

    async def _open_stream(self, messages: list[ChatMessage], config: ModelConfig):
        try:
            return await self._client.chat.completions.create(
                **self._build_request(messages, config), stream=True
            )
        except openai.APIError as error:
            raise self._translate(error) from error

    def _translate(self, error: openai.APIError) -> LLMError:
        return translate_sdk_error(error, provider=self.provider, sdk=openai)
