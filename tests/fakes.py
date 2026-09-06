"""Dobles de prueba (fakes) que imitan la forma mínima de los SDKs de OpenAI y Anthropic.

Los clientes reciben el SDK por inyección de dependencias: en los tests les pasamos
estos objetos en vez de AsyncOpenAI / AsyncAnthropic. Nada de acá toca la red.
"""

from types import SimpleNamespace

import anthropic
import openai

# Las versiones instaladas (openai 3.8.0, anthropic 1.4.0) están construidas sobre
# httpx2; versiones anteriores de cada SDK usan httpx. Se resuelve el módulo HTTP
# de cada SDK por separado porque uno podría migrar antes que el otro.
try:
    import httpx2 as openai_httpx
except ImportError:  # pragma: no cover
    import httpx as openai_httpx

try:  # anthropic 1.x está construido sobre httpx2; versiones anteriores usan httpx
    import httpx2 as anthropic_httpx
except ImportError:  # pragma: no cover
    import httpx as anthropic_httpx


class FakeStream:
    """Imita el stream de un SDK: se recorre con `async for` y se cierra con `close()`.

    `fail_after` permite simular un corte a mitad del stream: después de entregar
    todos los items, lanza esa excepción.
    """

    def __init__(self, items, fail_after: Exception | None = None):
        self.items = list(items)
        self.fail_after = fail_after
        self.closed = False

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for item in self.items:
            yield item
        if self.fail_after is not None:
            raise self.fail_after

    async def close(self):
        self.closed = True


class FakeEndpoint:
    """Imita el método `create(...)` de un SDK.

    Recibe una secuencia de resultados; cada llamada consume el siguiente.
    Si el resultado es una excepción, la lanza; si no, lo devuelve tal cual.
    Registra cada llamada en `calls` para que los tests inspeccionen los argumentos.
    """

    def __init__(self, *results):
        self.results = list(results)
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.results:
            raise AssertionError("FakeEndpoint recibió más llamadas de las esperadas")
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def fake_openai_sdk(*results):
    """Devuelve (sdk, endpoint): `sdk.chat.completions.create` es el endpoint falso."""
    endpoint = FakeEndpoint(*results)
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=endpoint))
    return sdk, endpoint


def fake_anthropic_sdk(*results):
    """Devuelve (sdk, endpoint): `sdk.messages.create` es el endpoint falso."""
    endpoint = FakeEndpoint(*results)
    sdk = SimpleNamespace(messages=endpoint)
    return sdk, endpoint


# --- Respuestas con la forma que devuelve cada SDK ---------------------------


def openai_response(
    text,
    *,
    model="gpt-4o-mini",
    prompt_tokens=10,
    completion_tokens=5,
    finish_reason="stop",
):
    return SimpleNamespace(
        model=model,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
    )


def openai_chunk(text):
    """Un fragmento de streaming. `text=None` imita el chunk final sin contenido."""
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


def openai_empty_chunk():
    """Chunk sin `choices` (OpenAI manda uno así al final con el uso de tokens)."""
    return SimpleNamespace(choices=[])


def anthropic_response(
    text,
    *,
    model="claude-haiku-4-5",
    input_tokens=10,
    output_tokens=5,
    stop_reason="end_turn",
):
    return SimpleNamespace(
        model=model,
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
        stop_reason=stop_reason,
    )


def anthropic_text_event(text):
    return SimpleNamespace(
        type="content_block_delta",
        delta=SimpleNamespace(type="text_delta", text=text),
    )


def anthropic_json_delta_event():
    """Delta que no es texto (lo usan las herramientas). El cliente debe ignorarlo."""
    return SimpleNamespace(
        type="content_block_delta",
        delta=SimpleNamespace(type="input_json_delta", partial_json="{"),
    )


def anthropic_event(event_type):
    """Evento sin texto: message_start, content_block_stop, message_stop, etc."""
    return SimpleNamespace(type=event_type)


# --- Excepciones reales de cada SDK ------------------------------------------


def _http_response(http, status):
    request = http.Request("POST", "https://example.test/v1")
    return http.Response(status, request=request)


def openai_status_error(cls, status):
    """Instancia una excepción HTTP real del SDK de OpenAI, ej. openai.RateLimitError."""
    return cls("error de prueba", response=_http_response(openai_httpx, status), body=None)


def openai_connection_error():
    return openai.APIConnectionError(
        request=openai_httpx.Request("POST", "https://example.test/v1")
    )


def anthropic_status_error(cls, status):
    """Instancia una excepción HTTP real del SDK de Anthropic."""
    return cls("error de prueba", response=_http_response(anthropic_httpx, status), body=None)


def anthropic_connection_error():
    return anthropic.APIConnectionError(
        request=anthropic_httpx.Request("POST", "https://example.test/v1")
    )


def transport_error():
    """Corte de conexión a mitad de stream: no es un error propio del SDK (openai/anthropic
    APIError), sino una excepción de httpx que la iteración del SDK no envuelve."""
    return anthropic_httpx.ReadError("conexión cortada")
