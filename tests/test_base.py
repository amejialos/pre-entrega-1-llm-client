"""Tests del contrato BaseLLMClient."""

import pytest

from llm_client.base import BaseLLMClient


def test_la_base_no_se_puede_instanciar():
    with pytest.raises(TypeError):
        BaseLLMClient()


def test_subclase_incompleta_no_se_puede_instanciar():
    class SoloGenerate(BaseLLMClient):
        provider = "openai"

        async def generate(self, messages, config=None):
            raise NotImplementedError

    with pytest.raises(TypeError):
        SoloGenerate()


def test_subclase_completa_se_instancia():
    class Completo(BaseLLMClient):
        provider = "openai"

        async def generate(self, messages, config=None):
            raise NotImplementedError

        async def stream(self, messages, config=None):
            yield "x"

    assert Completo().provider == "openai"
