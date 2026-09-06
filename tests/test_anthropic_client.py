"""Tests de AnthropicClient con un SDK falso. No tocan la red ni necesitan keys."""

from types import SimpleNamespace

import anthropic
import pytest

from llm_client import errors
from llm_client.anthropic_client import AnthropicClient
from llm_client.schemas import ChatMessage, ModelConfig
from tests.fakes import (
    FakeStream,
    anthropic_connection_error,
    anthropic_event,
    anthropic_json_delta_event,
    anthropic_response,
    anthropic_status_error,
    anthropic_text_event,
    fake_anthropic_sdk,
)

MESSAGES = [ChatMessage(role="user", content="¿Qué es la entropía?")]


def make_client(*results, attempts=3, base_delay=0.0):
    sdk, endpoint = fake_anthropic_sdk(*results)
    client = AnthropicClient(api_key="falsa", client=sdk, attempts=attempts, base_delay=base_delay)
    return client, endpoint


# --- Modo normal --------------------------------------------------------------


async def test_generate_traduce_la_respuesta_del_sdk():
    client, _ = make_client(
        anthropic_response("La entropía mide el desorden.", input_tokens=12, output_tokens=7)
    )

    response = await client.generate(MESSAGES)

    assert response.content == "La entropía mide el desorden."
    assert response.provider == "anthropic"
    assert response.model == "claude-haiku-4-5"
    assert response.input_tokens == 12
    assert response.output_tokens == 7
    assert response.finish_reason == "end_turn"


async def test_generate_envia_modelo_mensajes_y_max_tokens():
    client, endpoint = make_client(anthropic_response("ok"))

    await client.generate(MESSAGES, ModelConfig(max_tokens=50))

    request = endpoint.calls[0]
    assert request["model"] == "claude-haiku-4-5"
    assert request["messages"] == [{"role": "user", "content": "¿Qué es la entropía?"}]
    assert request["max_tokens"] == 50
    assert "temperature" not in request
    assert "system" not in request
    assert "stream" not in request


async def test_mensajes_system_y_system_prompt_van_en_el_parametro_system():
    client, endpoint = make_client(anthropic_response("ok"))
    messages = [
        ChatMessage(role="system", content="Respondé en una línea."),
        ChatMessage(role="user", content="hola"),
    ]

    await client.generate(messages, ModelConfig(system_prompt="Sos un físico."))

    request = endpoint.calls[0]
    assert request["system"] == "Sos un físico.\n\nRespondé en una línea."
    assert request["messages"] == [{"role": "user", "content": "hola"}]


async def test_temperature_se_recorta_a_uno():
    client, endpoint = make_client(anthropic_response("ok"))
    await client.generate(MESSAGES, ModelConfig(temperature=1.5))
    assert endpoint.calls[0]["temperature"] == 1.0


async def test_temperature_dentro_del_rango_no_cambia():
    client, endpoint = make_client(anthropic_response("ok"))
    await client.generate(MESSAGES, ModelConfig(temperature=0.3))
    assert endpoint.calls[0]["temperature"] == 0.3


async def test_generate_concatena_solo_bloques_de_texto():
    response = anthropic_response("Hola")
    response.content = [
        SimpleNamespace(type="thinking", thinking="..."),
        SimpleNamespace(type="text", text="Hola"),
        SimpleNamespace(type="text", text=" mundo"),
    ]
    client, _ = make_client(response)

    assert (await client.generate(MESSAGES)).content == "Hola mundo"


# --- Streaming ------------------------------------------------------------------


async def test_stream_solo_emite_text_delta():
    stream = FakeStream(
        [
            anthropic_event("message_start"),
            anthropic_text_event("La "),
            anthropic_json_delta_event(),
            anthropic_text_event("entropía"),
            anthropic_event("message_stop"),
        ]
    )
    client, endpoint = make_client(stream)

    chunks = [chunk async for chunk in client.stream(MESSAGES)]

    assert chunks == ["La ", "entropía"]
    assert endpoint.calls[0]["stream"] is True
    assert stream.closed


async def test_stream_reintenta_la_apertura():
    client, endpoint = make_client(
        anthropic_status_error(anthropic.RateLimitError, 429),
        FakeStream([anthropic_text_event("ok")]),
    )
    assert [chunk async for chunk in client.stream(MESSAGES)] == ["ok"]
    assert len(endpoint.calls) == 2


async def test_error_a_mitad_de_stream_se_traduce_sin_reintentar():
    stream = FakeStream([anthropic_text_event("La ")], fail_after=anthropic_connection_error())
    client, endpoint = make_client(stream)
    received = []

    with pytest.raises(errors.LLMNetworkError):
        async for chunk in client.stream(MESSAGES):
            received.append(chunk)

    assert received == ["La "]
    assert len(endpoint.calls) == 1
    assert stream.closed


# --- Traducción de errores y reintentos ---------------------------------------


@pytest.mark.parametrize(
    "sdk_error, expected",
    [
        (anthropic_status_error(anthropic.AuthenticationError, 401), errors.LLMAuthError),
        (anthropic_status_error(anthropic.PermissionDeniedError, 403), errors.LLMAuthError),
        (anthropic_status_error(anthropic.RateLimitError, 429), errors.LLMRateLimitError),
        (anthropic_status_error(anthropic.InternalServerError, 500), errors.LLMServerError),
        (anthropic_status_error(anthropic.APIStatusError, 529), errors.LLMServerError),
        (anthropic_status_error(anthropic.BadRequestError, 400), errors.LLMProviderError),
        (anthropic_status_error(anthropic.NotFoundError, 404), errors.LLMProviderError),
        (anthropic_connection_error(), errors.LLMNetworkError),
    ],
)
async def test_traduce_errores_del_sdk(sdk_error, expected):
    client, _ = make_client(sdk_error, attempts=1)

    with pytest.raises(expected) as info:
        await client.generate(MESSAGES)

    assert info.value.provider == "anthropic"
    assert info.value.original is sdk_error


async def test_rate_limit_se_reintenta_y_luego_responde():
    client, endpoint = make_client(
        anthropic_status_error(anthropic.RateLimitError, 429), anthropic_response("ok")
    )
    assert (await client.generate(MESSAGES)).content == "ok"
    assert len(endpoint.calls) == 2


async def test_auth_error_no_se_reintenta():
    client, endpoint = make_client(
        anthropic_status_error(anthropic.AuthenticationError, 401), anthropic_response("ok")
    )
    with pytest.raises(errors.LLMAuthError):
        await client.generate(MESSAGES)
    assert len(endpoint.calls) == 1


# --- Construcción del SDK real (sin red) ----------------------------------------


def test_construye_el_sdk_sin_reintentos_propios():
    client = AnthropicClient(api_key="falsa")
    assert client.model == AnthropicClient.DEFAULT_MODEL
    assert client._client.max_retries == 0
