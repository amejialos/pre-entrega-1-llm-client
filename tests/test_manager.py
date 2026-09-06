"""Tests de AsyncLLMManager: la única pieza que lee variables de entorno."""

import pytest

from llm_client import (
    AnthropicClient,
    AsyncLLMManager,
    ChatMessage,
    LLMConfigError,
    ModelConfig,
    ModelResponse,
    OpenAIClient,
)

PROJECT_VARS = [
    "LLM_PROVIDER",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_MODEL",
]


@pytest.fixture(autouse=True)
def entorno_limpio(monkeypatch):
    """Cada test arranca sin ninguna variable del proyecto en el entorno."""
    for var in PROJECT_VARS:
        monkeypatch.delenv(var, raising=False)


def test_proveedor_desconocido():
    with pytest.raises(LLMConfigError) as info:
        AsyncLLMManager(provider="gemini")
    assert "gemini" in str(info.value)


def test_key_faltante_nombra_la_variable():
    with pytest.raises(LLMConfigError) as info:
        AsyncLLMManager(provider="anthropic")
    assert "ANTHROPIC_API_KEY" in str(info.value)


def test_key_vacia_cuenta_como_faltante(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "   ")
    with pytest.raises(LLMConfigError) as info:
        AsyncLLMManager(provider="openai")
    assert "OPENAI_API_KEY" in str(info.value)


def test_el_default_es_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "falsa")
    manager = AsyncLLMManager()
    assert isinstance(manager.client, AnthropicClient)
    assert manager.provider == "anthropic"


def test_llm_provider_elige_openai(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "falsa")
    manager = AsyncLLMManager()
    assert isinstance(manager.client, OpenAIClient)
    assert manager.provider == "openai"


def test_argumento_explicito_gana_sobre_la_variable(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "falsa")
    assert isinstance(AsyncLLMManager(provider="anthropic").client, AnthropicClient)


def test_el_nombre_del_proveedor_se_normaliza(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "falsa")
    assert isinstance(AsyncLLMManager(provider="  OpenAI ").client, OpenAIClient)


def test_modelo_y_base_url_desde_el_entorno(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "falsa")
    monkeypatch.setenv("OPENAI_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")

    manager = AsyncLLMManager(provider="openai")

    assert manager.client.model == "gemini-2.5-flash"
    assert manager.client.base_url == "https://generativelanguage.googleapis.com/v1beta/openai/"


def test_modelo_por_defecto_si_no_hay_variable(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "falsa")
    assert AsyncLLMManager(provider="anthropic").client.model == "claude-haiku-4-5"


class ClienteFalso:
    """Reemplaza al cliente real para verificar que el manager solo delega."""

    provider = "openai"

    def __init__(self):
        self.calls = []

    async def generate(self, messages, config=None):
        self.calls.append(("generate", messages, config))
        return ModelResponse(content="ok", model="m", provider="openai")

    async def stream(self, messages, config=None):
        self.calls.append(("stream", messages, config))
        yield "o"
        yield "k"


async def test_generate_delega_en_el_cliente(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "falsa")
    manager = AsyncLLMManager(provider="openai")
    manager.client = ClienteFalso()
    messages = [ChatMessage(role="user", content="hola")]
    config = ModelConfig(max_tokens=5)

    response = await manager.generate(messages, config)

    assert response.content == "ok"
    assert manager.client.calls == [("generate", messages, config)]


async def test_stream_delega_en_el_cliente(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "falsa")
    manager = AsyncLLMManager(provider="openai")
    manager.client = ClienteFalso()
    messages = [ChatMessage(role="user", content="hola")]

    chunks = [chunk async for chunk in manager.stream(messages)]

    assert chunks == ["o", "k"]
    assert manager.client.calls == [("stream", messages, None)]
