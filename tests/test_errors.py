"""Tests de la jerarquía LLMError."""

import pytest

from llm_client.errors import (
    LLMAuthError,
    LLMConfigError,
    LLMError,
    LLMNetworkError,
    LLMProviderError,
    LLMRateLimitError,
    LLMServerError,
)


def test_str_incluye_proveedor_y_mensaje():
    error = LLMError("algo falló", provider="openai")
    assert str(error) == "[openai] algo falló"
    assert error.message == "algo falló"
    assert error.provider == "openai"
    assert error.original is None


def test_conserva_la_excepcion_original():
    original = ValueError("del SDK")
    error = LLMAuthError("key inválida", provider="anthropic", original=original)
    assert error.original is original


@pytest.mark.parametrize("cls", [LLMRateLimitError, LLMNetworkError, LLMServerError])
def test_errores_reintentables(cls):
    assert cls.retryable is True
    assert issubclass(cls, LLMError)


@pytest.mark.parametrize("cls", [LLMConfigError, LLMAuthError, LLMProviderError])
def test_errores_no_reintentables(cls):
    assert cls.retryable is False
    assert issubclass(cls, LLMError)
