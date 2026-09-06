"""Tests de los modelos Pydantic: validan datos en el momento de crearlos."""

import pytest
from pydantic import ValidationError

from llm_client.schemas import ChatMessage, ModelConfig, ModelResponse


def test_chat_message_valido():
    message = ChatMessage(role="user", content="hola")
    assert message.role == "user"
    assert message.content == "hola"


def test_chat_message_rechaza_rol_invalido():
    with pytest.raises(ValidationError):
        ChatMessage(role="robot", content="hola")


def test_chat_message_rechaza_contenido_vacio():
    with pytest.raises(ValidationError):
        ChatMessage(role="user", content="")


def test_model_config_por_defecto():
    config = ModelConfig()
    assert config.model is None
    assert config.temperature is None
    assert config.max_tokens == 1024
    assert config.system_prompt is None


@pytest.mark.parametrize("value", [0, 0.7, 1, 2])
def test_model_config_acepta_temperatura_en_rango(value):
    assert ModelConfig(temperature=value).temperature == value


@pytest.mark.parametrize("value", [-0.1, 2.5])
def test_model_config_rechaza_temperatura_fuera_de_rango(value):
    with pytest.raises(ValidationError):
        ModelConfig(temperature=value)


def test_model_config_rechaza_max_tokens_cero():
    with pytest.raises(ValidationError):
        ModelConfig(max_tokens=0)


def test_model_config_rechaza_campo_desconocido():
    with pytest.raises(ValidationError):
        ModelConfig(temprature=0.5)  # typo a propósito


def test_model_response_minima():
    response = ModelResponse(content="hola", model="m", provider="openai")
    assert response.input_tokens is None
    assert response.output_tokens is None
    assert response.finish_reason is None


def test_model_response_rechaza_proveedor_desconocido():
    with pytest.raises(ValidationError):
        ModelResponse(content="hola", model="m", provider="gemini")
