"""Tests de OpenAIClient con un SDK falso. No tocan la red ni necesitan keys."""

import openai
import pytest

from llm_client import errors
from llm_client.openai_client import OpenAIClient
from llm_client.schemas import ChatMessage, ModelConfig
from tests.fakes import (
    FakeStream,
    fake_openai_sdk,
    openai_chunk,
    openai_connection_error,
    openai_empty_chunk,
    openai_response,
    openai_status_error,
)

MESSAGES = [ChatMessage(role="user", content="¿Qué es la entropía?")]


def make_client(*results, attempts=3, base_delay=0.0):
    """Cliente con SDK falso. base_delay=0 hace que los reintentos no esperen."""
    sdk, endpoint = fake_openai_sdk(*results)
    client = OpenAIClient(api_key="falsa", client=sdk, attempts=attempts, base_delay=base_delay)
    return client, endpoint


# --- Modo normal --------------------------------------------------------------


async def test_generate_traduce_la_respuesta_del_sdk():
    client, _ = make_client(
        openai_response("La entropía mide el desorden.", prompt_tokens=12, completion_tokens=7)
    )

    response = await client.generate(MESSAGES)

    assert response.content == "La entropía mide el desorden."
    assert response.provider == "openai"
    assert response.model == "gpt-4o-mini"
    assert response.input_tokens == 12
    assert response.output_tokens == 7
    assert response.finish_reason == "stop"


async def test_generate_envia_modelo_mensajes_y_max_tokens():
    client, endpoint = make_client(openai_response("ok"))

    await client.generate(MESSAGES, ModelConfig(max_tokens=50))

    request = endpoint.calls[0]
    assert request["model"] == "gpt-4o-mini"
    assert request["messages"] == [{"role": "user", "content": "¿Qué es la entropía?"}]
    assert request["max_tokens"] == 50
    assert "temperature" not in request
    assert "stream" not in request


async def test_config_model_gana_sobre_el_default():
    client, endpoint = make_client(openai_response("ok"))
    await client.generate(MESSAGES, ModelConfig(model="gpt-4.1"))
    assert endpoint.calls[0]["model"] == "gpt-4.1"


async def test_temperature_se_envia_solo_si_esta_definida():
    client, endpoint = make_client(openai_response("ok"))
    await client.generate(MESSAGES, ModelConfig(temperature=1.5))
    assert endpoint.calls[0]["temperature"] == 1.5


async def test_system_prompt_va_como_primer_mensaje_system():
    client, endpoint = make_client(openai_response("ok"))
    await client.generate(MESSAGES, ModelConfig(system_prompt="Sos un físico."))
    assert endpoint.calls[0]["messages"] == [
        {"role": "system", "content": "Sos un físico."},
        {"role": "user", "content": "¿Qué es la entropía?"},
    ]


async def test_contenido_none_se_convierte_en_cadena_vacia():
    client, _ = make_client(openai_response(None))
    response = await client.generate(MESSAGES)
    assert response.content == ""


# --- Streaming ------------------------------------------------------------------


async def test_stream_entrega_fragmentos_en_orden_e_ignora_chunks_vacios():
    stream = FakeStream(
        [openai_chunk("La "), openai_empty_chunk(), openai_chunk(None), openai_chunk("entropía")]
    )
    client, endpoint = make_client(stream)

    chunks = [chunk async for chunk in client.stream(MESSAGES)]

    assert chunks == ["La ", "entropía"]
    assert endpoint.calls[0]["stream"] is True
    assert stream.closed


async def test_stream_reintenta_la_apertura():
    client, endpoint = make_client(
        openai_status_error(openai.RateLimitError, 429),
        FakeStream([openai_chunk("ok")]),
    )

    assert [chunk async for chunk in client.stream(MESSAGES)] == ["ok"]
    assert len(endpoint.calls) == 2


async def test_error_a_mitad_de_stream_se_traduce_sin_reintentar():
    stream = FakeStream([openai_chunk("La ")], fail_after=openai_connection_error())
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
        (openai_status_error(openai.AuthenticationError, 401), errors.LLMAuthError),
        (openai_status_error(openai.PermissionDeniedError, 403), errors.LLMAuthError),
        (openai_status_error(openai.RateLimitError, 429), errors.LLMRateLimitError),
        (openai_status_error(openai.InternalServerError, 500), errors.LLMServerError),
        (openai_status_error(openai.BadRequestError, 400), errors.LLMProviderError),
        (openai_status_error(openai.NotFoundError, 404), errors.LLMProviderError),
        (openai_connection_error(), errors.LLMNetworkError),
    ],
)
async def test_traduce_errores_del_sdk(sdk_error, expected):
    client, _ = make_client(sdk_error, attempts=1)

    with pytest.raises(expected) as info:
        await client.generate(MESSAGES)

    assert info.value.provider == "openai"
    assert info.value.original is sdk_error


async def test_rate_limit_se_reintenta_y_luego_responde():
    client, endpoint = make_client(
        openai_status_error(openai.RateLimitError, 429), openai_response("ok")
    )
    response = await client.generate(MESSAGES)
    assert response.content == "ok"
    assert len(endpoint.calls) == 2


async def test_auth_error_no_se_reintenta():
    client, endpoint = make_client(
        openai_status_error(openai.AuthenticationError, 401), openai_response("ok")
    )
    with pytest.raises(errors.LLMAuthError):
        await client.generate(MESSAGES)
    assert len(endpoint.calls) == 1


# --- Construcción del SDK real (sin red) ----------------------------------------


def test_construye_el_sdk_sin_reintentos_propios_y_con_base_url():
    client = OpenAIClient(api_key="falsa", base_url="https://example.test/v1/")
    assert client.base_url == "https://example.test/v1/"
    assert client.model == OpenAIClient.DEFAULT_MODEL
    assert client._client.max_retries == 0
    assert str(client._client.base_url) == "https://example.test/v1/"
