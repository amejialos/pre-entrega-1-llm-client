"""Cliente asíncrono para la API de Anthropic (Claude)."""

from collections.abc import AsyncIterator
from typing import Any, ClassVar

import anthropic
from anthropic import AsyncAnthropic

from .base import BaseLLMClient
from .errors import LLMError, LLMProviderError, translate_sdk_error
from .retry import with_retry
from .schemas import ChatMessage, ModelConfig, ModelResponse, Provider


class AnthropicClient(BaseLLMClient):
    provider: ClassVar[Provider] = "anthropic"
    DEFAULT_MODEL = "claude-haiku-4-5"
    MAX_TEMPERATURE = 1.0  # Anthropic acepta 0 a 1; la consigna valida 0 a 2

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        client: AsyncAnthropic | None = None,
        attempts: int = 3,
        base_delay: float = 1.0,
    ) -> None:
        self.model = model
        self._attempts = attempts
        self._base_delay = base_delay
        # max_retries=0: el SDK no reintenta por su cuenta; with_retry es la única
        # lógica de reintento. `client` se inyecta en los tests.
        self._client = client or AsyncAnthropic(api_key=api_key, max_retries=0)

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
        # Solo la apertura se reintenta (ver OpenAIClient.stream).
        stream = await with_retry(
            lambda: self._open_stream(messages, config),
            attempts=self._attempts,
            base_delay=self._base_delay,
        )
        try:
            async for event in stream:
                # Anthropic emite varios tipos de evento; el texto viaja en los
                # content_block_delta cuyo delta es text_delta.
                if event.type == "content_block_delta" and event.delta.type == "text_delta":
                    yield event.delta.text
        except Exception as error:
            # El SDK no envuelve los errores de transporte durante la iteración: un
            # corte de conexión a mitad de stream llega como excepción de httpx, no
            # como anthropic.APIError. Se traduce igual para no filtrar excepciones crudas.
            raise self._translate(error) from error
        finally:
            await stream.close()

    # --- Internos ---------------------------------------------------------------

    def _build_request(self, messages: list[ChatMessage], config: ModelConfig) -> dict[str, Any]:
        # Anthropic no acepta rol "system" en la lista: va en el parámetro `system`.
        system_parts = [m.content for m in messages if m.role == "system"]
        if config.system_prompt:
            system_parts.insert(0, config.system_prompt)
        payload = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]

        request: dict[str, Any] = {
            "model": config.model or self.model,
            "messages": payload,
            "max_tokens": config.max_tokens,
        }
        if system_parts:
            request["system"] = "\n\n".join(system_parts)
        if config.temperature is not None:
            # El SDK 1.x quitó temperature de la firma de messages.create(), pero la API
            # sigue aceptándolo en Haiku 4.5 y modelos anteriores: se envía por extra_body.
            # Los modelos más nuevos lo rechazan con 400 (LLMProviderError).
            request["extra_body"] = {"temperature": min(config.temperature, self.MAX_TEMPERATURE)}
        return request

    async def _generate_once(self, messages: list[ChatMessage], config: ModelConfig) -> ModelResponse:
        try:
            response = await self._client.messages.create(**self._build_request(messages, config))
        except anthropic.APIError as error:
            raise self._translate(error) from error

        try:
            # response.content es una lista de bloques; solo nos interesan los de texto.
            text = "".join(block.text for block in response.content if block.type == "text")
            return ModelResponse(
                content=text,
                model=response.model,
                provider=self.provider,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                finish_reason=response.stop_reason,
            )
        except Exception as error:  # forma de respuesta inesperada (endpoint no estándar, cambio del SDK)
            raise LLMProviderError(
                f"Respuesta inesperada del proveedor: {error}", provider=self.provider, original=error
            ) from error

    async def _open_stream(self, messages: list[ChatMessage], config: ModelConfig):
        try:
            return await self._client.messages.create(
                **self._build_request(messages, config), stream=True
            )
        except anthropic.APIError as error:
            raise self._translate(error) from error

    def _translate(self, error: anthropic.APIError) -> LLMError:
        return translate_sdk_error(error, provider=self.provider, sdk=anthropic)
