"""Tests de with_retry. `sleep` se inyecta para registrar las esperas sin dormir."""

import logging

import pytest

from llm_client.errors import LLMAuthError, LLMRateLimitError
from llm_client.retry import with_retry


class SleepRecorder:
    """Reemplaza asyncio.sleep: anota cuánto se pidió esperar y vuelve al instante."""

    def __init__(self):
        self.delays: list[float] = []

    async def sleep(self, seconds: float) -> None:
        self.delays.append(seconds)


def flaky(failures: int, result: str = "ok"):
    """Operación que falla `failures` veces con rate limit y después devuelve `result`."""
    state = {"calls": 0}

    async def operation():
        state["calls"] += 1
        if state["calls"] <= failures:
            raise LLMRateLimitError("límite de tasa", provider="test")
        return result

    return operation, state


async def test_exito_al_primer_intento_no_espera():
    operation, state = flaky(failures=0)
    recorder = SleepRecorder()

    assert await with_retry(operation, sleep=recorder.sleep) == "ok"
    assert state["calls"] == 1
    assert recorder.delays == []


async def test_reintenta_un_error_reintentable():
    operation, state = flaky(failures=1)
    recorder = SleepRecorder()

    assert await with_retry(operation, sleep=recorder.sleep) == "ok"
    assert state["calls"] == 2
    assert recorder.delays == [1.0]


async def test_backoff_exponencial():
    operation, _ = flaky(failures=2)
    recorder = SleepRecorder()

    await with_retry(operation, attempts=3, base_delay=1.0, sleep=recorder.sleep)
    assert recorder.delays == [1.0, 2.0]


async def test_se_rinde_al_agotar_los_intentos():
    operation, state = flaky(failures=5)
    recorder = SleepRecorder()

    with pytest.raises(LLMRateLimitError):
        await with_retry(operation, attempts=3, sleep=recorder.sleep)
    assert state["calls"] == 3
    assert recorder.delays == [1.0, 2.0]


async def test_no_reintenta_un_error_no_reintentable():
    calls = 0

    async def operation():
        nonlocal calls
        calls += 1
        raise LLMAuthError("key inválida", provider="test")

    recorder = SleepRecorder()
    with pytest.raises(LLMAuthError):
        await with_retry(operation, sleep=recorder.sleep)
    assert calls == 1
    assert recorder.delays == []


async def test_attempts_menor_a_uno_es_error_de_programacion():
    operation, _ = flaky(failures=0)
    with pytest.raises(ValueError):
        await with_retry(operation, attempts=0)


async def test_registra_un_warning_por_cada_reintento(caplog):
    operation, _ = flaky(failures=2)
    recorder = SleepRecorder()

    with caplog.at_level(logging.WARNING, logger="llm_client.retry"):
        await with_retry(operation, attempts=3, sleep=recorder.sleep)

    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warning_records) == 2
    for record in warning_records:
        assert "Reintentando" in record.getMessage()
