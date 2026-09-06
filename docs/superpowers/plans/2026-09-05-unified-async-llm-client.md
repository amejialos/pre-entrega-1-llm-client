# Unified Async LLM Client: plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construir el paquete `llm_client` (OpenAI + Anthropic bajo una interfaz común, asíncrono, con streaming, validación Pydantic y errores controlados), su script de validación `main.py`, tests sin red y README, listo para subir a GitHub.

**Architecture:** Un paquete con un módulo por responsabilidad. `schemas.py` define los datos; `errors.py` y `retry.py` el manejo de fallos; `base.py` el contrato; `openai_client.py` y `anthropic_client.py` traducen cada SDK a ese contrato; `manager.py` es la única pieza que lee el entorno y elige el proveedor. Los clientes reciben el SDK por inyección de dependencias, lo que permite testear todo con fakes.

**Tech Stack:** Python 3.12, `uv`, `openai` (AsyncOpenAI), `anthropic` (AsyncAnthropic), `pydantic` v2, `python-dotenv`, `pytest`, `pytest-asyncio`.

**Spec:** `docs/superpowers/specs/2026-09-05-unified-async-llm-client-design.md`

## Global Constraints

- `requires-python = ">=3.12"`. El entorno se crea con `uv` (Python 3.12.14 ya está instalado localmente).
- Dependencias de producción: `openai`, `anthropic`, `pydantic` (v2), `python-dotenv`. De desarrollo: `pytest`, `pytest-asyncio` (modo `auto`).
- Ninguna llamada síncrona a los SDKs: siempre `AsyncOpenAI` / `AsyncAnthropic` y `await`.
- Los SDKs se construyen con `max_retries=0`; `with_retry` es la única lógica de reintento.
- Solo `AsyncLLMManager` lee `os.environ`. Los clientes reciben todo por parámetro.
- Modelos por defecto: `gpt-4o-mini` y `claude-haiku-4-5`.
- Ninguna API key en el código ni en git. `.env` está en `.gitignore`; `.env.example` sí se sube.
- La suite de tests corre sin red y sin keys.
- Prosa, docstrings, comentarios y mensajes de commit en español.
- Cada commit termina con la línea `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Todos los comandos se corren desde la raíz del proyecto: `/Users/flaviasendino/Documents/Pre Entrega 1`.

---

### Task 1: Entorno, dependencias y esqueleto del proyecto

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version` (lo genera `uv python pin`)
- Create: `uv.lock` (lo genera `uv add`)
- Create: `llm_client/__init__.py`
- Create: `tests/__init__.py`
- Create: `.env.example`

**Interfaces:**
- Consumes: nada.
- Produces: un venv en `.venv` con Python 3.12 y todas las dependencias; el paquete `llm_client` importable; `uv run pytest` funcionando.

- [ ] **Step 1: Fijar la versión de Python**

Run: `uv python pin 3.12`
Expected: se crea `.python-version` con el contenido `3.12`.

- [ ] **Step 2: Escribir `pyproject.toml`**

```toml
[project]
name = "llm-client"
version = "0.1.0"
description = "Unified Async LLM Client: OpenAI y Anthropic bajo una interfaz común y asíncrona"
requires-python = ">=3.12"
dependencies = []

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["llm_client"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "function"
```

Qué es cada bloque: `[project]` son los metadatos y la lista de dependencias (vacía por ahora, `uv add` la completa). `[build-system]` le dice a `uv` cómo instalar el paquete en el venv en modo editable. `[tool.pytest.ini_options]` configura pytest: `asyncio_mode = "auto"` hace que los tests `async def` corran sin decorador.

- [ ] **Step 3: Agregar dependencias con uv**

Run:
```bash
uv add openai anthropic pydantic python-dotenv
uv add --dev pytest pytest-asyncio
```
Expected: `uv` crea `.venv/`, escribe `uv.lock`, y `pyproject.toml` ahora tiene `dependencies = [...]` con versiones y un bloque `[dependency-groups] dev = [...]`.

- [ ] **Step 4: Crear los paquetes vacíos**

`llm_client/__init__.py`:
```python
"""Unified Async LLM Client: OpenAI y Anthropic bajo una interfaz común y asíncrona."""
```

`tests/__init__.py`: archivo vacío. Hace que `tests` sea un paquete y que los tests puedan importar `tests.fakes`.

- [ ] **Step 5: Crear `.env.example`**

```
# Copiá este archivo a .env y completá las keys. El .env NUNCA se sube a git.

# Proveedor por defecto cuando el código no indica uno: openai | anthropic
LLM_PROVIDER=anthropic

# --- OpenAI ---
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
# Opcional: endpoint alternativo compatible con la API de OpenAI.
# Para probar el camino de OpenAI gratis con una key de Gemini (Google AI Studio):
#   OPENAI_API_KEY=<tu key de Gemini, empieza con AIza>
#   OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
#   OPENAI_MODEL=gemini-3.6-flash
# OPENAI_BASE_URL=

# --- Anthropic ---
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-haiku-4-5
```

- [ ] **Step 6: Verificar el entorno**

Run: `uv run python -c "import sys, openai, anthropic, pydantic, dotenv, llm_client; print(sys.version.split()[0], openai.__version__, anthropic.__version__, pydantic.VERSION)"`
Expected: una línea que empieza con `3.12.` seguida de las versiones. Sin errores de import.

Run: `uv run pytest`
Expected: `no tests ran` y código de salida 5 (todavía no hay tests; eso es correcto).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .python-version uv.lock llm_client/__init__.py tests/__init__.py .env.example
git commit -m "Crear entorno Python 3.12 con uv y esqueleto del paquete

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Estructura de datos con Pydantic (`schemas.py`)

**Files:**
- Create: `llm_client/schemas.py`
- Test: `tests/test_schemas.py`

**Interfaces:**
- Consumes: nada.
- Produces:
  - `Role = Literal["system", "user", "assistant"]`
  - `Provider = Literal["openai", "anthropic"]`
  - `ChatMessage(role: Role, content: str)`; `content` no vacío.
  - `ModelConfig(model: str | None = None, temperature: float | None = None, max_tokens: int = 1024, system_prompt: str | None = None)`; `temperature` en [0, 2] si se indica; `max_tokens > 0`; campos desconocidos rechazados.
  - `ModelResponse(content: str, model: str, provider: Provider, input_tokens: int | None = None, output_tokens: int | None = None, finish_reason: str | None = None)`.

- [ ] **Step 1: Escribir los tests que fallan**

`tests/test_schemas.py`:
```python
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
```

- [ ] **Step 2: Correr los tests y verificar que fallan**

Run: `uv run pytest tests/test_schemas.py -v`
Expected: FAIL en la importación con `ModuleNotFoundError: No module named 'llm_client.schemas'`.

- [ ] **Step 3: Implementar `schemas.py`**

`llm_client/schemas.py`:
```python
"""Estructuras de datos del cliente, validadas con Pydantic.

Todo el paquete habla en estos términos. Cada cliente traduce el formato de su SDK
a `ModelResponse` una sola vez, y el resto del programa nunca ve diccionarios anidados.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant"]
Provider = Literal["openai", "anthropic"]

# Rango que pide la consigna (0 a 2). Anthropic acepta 0 a 1: su cliente recorta.
Temperature = Annotated[float, Field(ge=0.0, le=2.0)]


class ChatMessage(BaseModel):
    """Un mensaje de la conversación."""

    model_config = ConfigDict(extra="forbid")

    role: Role
    content: str = Field(min_length=1)


class ModelConfig(BaseModel):
    """Parámetros de una llamada al modelo.

    `model` en None significa "usar el modelo por defecto del cliente".
    `temperature` en None significa "no enviar el parámetro": así funciona con
    los modelos que no lo aceptan.
    """

    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    temperature: Temperature | None = None
    max_tokens: int = Field(default=1024, gt=0)
    system_prompt: str | None = None


class ModelResponse(BaseModel):
    """Respuesta normalizada, igual para todos los proveedores."""

    model_config = ConfigDict(extra="forbid")

    content: str
    model: str
    provider: Provider
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str | None = None
```

- [ ] **Step 4: Correr los tests y verificar que pasan**

Run: `uv run pytest tests/test_schemas.py -v`
Expected: 14 PASSED (los parametrizados cuentan por separado).

- [ ] **Step 5: Commit**

```bash
git add llm_client/schemas.py tests/test_schemas.py
git commit -m "Agregar schemas Pydantic: ChatMessage, ModelConfig y ModelResponse

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Errores propios y reintentos (`errors.py`, `retry.py`)

**Files:**
- Create: `llm_client/errors.py`
- Create: `llm_client/retry.py`
- Test: `tests/test_errors.py`
- Test: `tests/test_retry.py`

**Interfaces:**
- Consumes: nada.
- Produces:
  - `LLMError(message: str, *, provider: str, original: Exception | None = None)` con atributos `message`, `provider`, `original`, clase-atributo `retryable: bool = False`, y `str(e) == f"[{provider}] {message}"`.
  - Subclases: `LLMConfigError`, `LLMAuthError`, `LLMProviderError` (no reintentables); `LLMRateLimitError`, `LLMNetworkError`, `LLMServerError` (`retryable = True`).
  - `async def with_retry(operation, *, attempts=3, base_delay=1.0, sleep=asyncio.sleep) -> T`.

- [ ] **Step 1: Escribir los tests de errores**

`tests/test_errors.py`:
```python
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
```

- [ ] **Step 2: Escribir los tests de reintentos**

`tests/test_retry.py`:
```python
"""Tests de with_retry. `sleep` se inyecta para registrar las esperas sin dormir."""

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
```

- [ ] **Step 3: Correr los tests y verificar que fallan**

Run: `uv run pytest tests/test_errors.py tests/test_retry.py -v`
Expected: FAIL en la importación con `ModuleNotFoundError: No module named 'llm_client.errors'`.

- [ ] **Step 4: Implementar `errors.py`**

`llm_client/errors.py`:
```python
"""Excepciones propias del cliente.

Cada cliente atrapa las excepciones de su SDK y las traduce a una de estas.
Así, quien usa el paquete solo necesita `except LLMError` y nunca depende de
los SDKs. El atributo `retryable` le dice a `with_retry` si vale la pena insistir.
"""


class LLMError(Exception):
    """Base de todos los errores del cliente."""

    retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        original: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.original = original  # la excepción del SDK, para depurar

    def __str__(self) -> str:
        return f"[{self.provider}] {self.message}"


class LLMConfigError(LLMError):
    """Proveedor desconocido o variable de entorno faltante."""


class LLMAuthError(LLMError):
    """API key inválida, ausente o sin permisos (HTTP 401 / 403)."""


class LLMRateLimitError(LLMError):
    """Límite de tasa o cuota agotada (HTTP 429). Se resuelve esperando."""

    retryable = True


class LLMNetworkError(LLMError):
    """Timeout o conexión caída antes de recibir respuesta."""

    retryable = True


class LLMServerError(LLMError):
    """Error del lado del proveedor (HTTP 5xx, incluida la sobrecarga 529)."""

    retryable = True


class LLMProviderError(LLMError):
    """Cualquier otro error HTTP de la API (400, 404, ...). No se reintenta."""
```

- [ ] **Step 5: Implementar `retry.py`**

`llm_client/retry.py`:
```python
"""Reintentos con backoff exponencial para operaciones asíncronas.

No conoce ningún SDK: solo entiende `LLMError` y su atributo `retryable`.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TypeVar

from .errors import LLMError

T = TypeVar("T")
logger = logging.getLogger(__name__)


async def with_retry(
    operation: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    base_delay: float = 1.0,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Ejecuta `operation()` hasta `attempts` veces.

    Reintenta solo si falla con un `LLMError` reintentable. Entre intentos espera
    `base_delay * 2 ** (n - 1)` segundos (1 s, 2 s, 4 s...). Si el error no es
    reintentable o se agotaron los intentos, relanza el mismo error.

    `sleep` se puede reemplazar en tests para no esperar de verdad.
    """
    if attempts < 1:
        raise ValueError("attempts debe ser al menos 1")

    for attempt in range(1, attempts + 1):
        try:
            return await operation()
        except LLMError as error:
            if not error.retryable or attempt == attempts:
                raise
            delay = base_delay * 2 ** (attempt - 1)
            logger.warning(
                "Intento %d/%d falló (%s). Reintentando en %.1fs.",
                attempt, attempts, error, delay,
            )
            await sleep(delay)

    raise AssertionError("inalcanzable: el bucle siempre devuelve o relanza")
```

- [ ] **Step 6: Correr los tests y verificar que pasan**

Run: `uv run pytest tests/test_errors.py tests/test_retry.py -v`
Expected: 14 PASSED.

- [ ] **Step 7: Commit**

```bash
git add llm_client/errors.py llm_client/retry.py tests/test_errors.py tests/test_retry.py
git commit -m "Agregar jerarquía LLMError y with_retry con backoff exponencial

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: La interfaz común (`base.py`)

**Files:**
- Create: `llm_client/base.py`
- Test: `tests/test_base.py`

**Interfaces:**
- Consumes: `ChatMessage`, `ModelConfig`, `ModelResponse`, `Provider` de `schemas.py`.
- Produces: `BaseLLMClient` (ABC) con `provider: ClassVar[Provider]`, `async def generate(messages, config=None) -> ModelResponse` y `def stream(messages, config=None) -> AsyncIterator[str]`, ambos abstractos.

- [ ] **Step 1: Escribir los tests que fallan**

`tests/test_base.py`:
```python
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
```

- [ ] **Step 2: Correr los tests y verificar que fallan**

Run: `uv run pytest tests/test_base.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'llm_client.base'`.

- [ ] **Step 3: Implementar `base.py`**

`llm_client/base.py`:
```python
"""Contrato común de todos los clientes de LLM."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import ClassVar

from .schemas import ChatMessage, ModelConfig, ModelResponse, Provider


class BaseLLMClient(ABC):
    """Interfaz que todo proveedor implementa.

    `generate` es una corrutina: devuelve la respuesta completa.
    `stream` es un generador asíncrono: entrega fragmentos de texto con `yield`.
    Son métodos separados porque una función no puede ser las dos cosas a la vez.
    """

    provider: ClassVar[Provider]

    @abstractmethod
    async def generate(
        self,
        messages: list[ChatMessage],
        config: ModelConfig | None = None,
    ) -> ModelResponse:
        """Envía la conversación y espera la respuesta completa."""

    @abstractmethod
    def stream(
        self,
        messages: list[ChatMessage],
        config: ModelConfig | None = None,
    ) -> AsyncIterator[str]:
        """Envía la conversación y entrega el texto por fragmentos, a medida que llega."""
```

- [ ] **Step 4: Correr los tests y verificar que pasan**

Run: `uv run pytest tests/test_base.py -v`
Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add llm_client/base.py tests/test_base.py
git commit -m "Agregar BaseLLMClient, el contrato común de los proveedores

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Fakes de los SDKs y `OpenAIClient`

**Files:**
- Create: `tests/fakes.py`
- Modify: `llm_client/errors.py` (agregar `translate_sdk_error`)
- Create: `llm_client/openai_client.py`
- Test: `tests/test_openai_client.py`

**Interfaces:**
- Consumes: `BaseLLMClient`; `LLMAuthError`, `LLMError`, `LLMNetworkError`, `LLMProviderError`, `LLMRateLimitError`, `LLMServerError`; `with_retry`; `ChatMessage`, `ModelConfig`, `ModelResponse`, `Provider`.
- Produces:
  - `translate_sdk_error(error: Exception, *, provider: str, sdk: ModuleType) -> LLMError` en `errors.py`: la misma tabla de traducción sirve para `openai` y `anthropic` porque ambos SDKs exponen excepciones con los mismos nombres.
  - `OpenAIClient(api_key: str, model: str = "gpt-4o-mini", base_url: str | None = None, client: AsyncOpenAI | None = None, attempts: int = 3, base_delay: float = 1.0)` con atributos públicos `model`, `base_url` y clase-atributo `DEFAULT_MODEL`.
  - Fakes reutilizables en `tests/fakes.py`: `FakeStream`, `FakeEndpoint`, `fake_openai_sdk`, `fake_anthropic_sdk`, `openai_response`, `openai_chunk`, `openai_empty_chunk`, `openai_status_error`, `openai_connection_error`, `anthropic_response`, `anthropic_text_event`, `anthropic_json_delta_event`, `anthropic_event`, `anthropic_status_error`, `anthropic_connection_error`.

- [ ] **Step 1: Escribir los fakes compartidos**

`tests/fakes.py`:
```python
"""Dobles de prueba (fakes) que imitan la forma mínima de los SDKs de OpenAI y Anthropic.

Los clientes reciben el SDK por inyección de dependencias: en los tests les pasamos
estos objetos en vez de AsyncOpenAI / AsyncAnthropic. Nada de acá toca la red.
"""

from types import SimpleNamespace

import anthropic
import httpx
import openai

try:  # anthropic 1.x está construido sobre httpx2; versiones anteriores usan httpx
    import httpx2 as anthropic_httpx
except ImportError:  # pragma: no cover
    anthropic_httpx = httpx


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
    return cls("error de prueba", response=_http_response(httpx, status), body=None)


def openai_connection_error():
    return openai.APIConnectionError(request=httpx.Request("POST", "https://example.test/v1"))


def anthropic_status_error(cls, status):
    """Instancia una excepción HTTP real del SDK de Anthropic."""
    return cls("error de prueba", response=_http_response(anthropic_httpx, status), body=None)


def anthropic_connection_error():
    return anthropic.APIConnectionError(
        request=anthropic_httpx.Request("POST", "https://example.test/v1")
    )
```

- [ ] **Step 2: Escribir los tests de `OpenAIClient`**

`tests/test_openai_client.py`:
```python
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
```

- [ ] **Step 3: Correr los tests y verificar que fallan**

Run: `uv run pytest tests/test_openai_client.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'llm_client.openai_client'`.

- [ ] **Step 4: Agregar `translate_sdk_error` a `errors.py`**

Agregar al final de `llm_client/errors.py` (y `from types import ModuleType` arriba, junto a los imports):
```python
# --- Traducción desde los SDKs -------------------------------------------------


def translate_sdk_error(error: Exception, *, provider: str, sdk: ModuleType) -> LLMError:
    """Convierte una excepción de un SDK (openai o anthropic) en la LLMError equivalente.

    Ambos SDKs exponen la misma jerarquía de excepciones con los mismos nombres
    (AuthenticationError, RateLimitError, APIStatusError, APIConnectionError...),
    así que una sola tabla sirve para los dos: se pasa el módulo del SDK.
    Se evalúa de la excepción más específica a la más general.
    """
    message = getattr(error, "message", None) or str(error)
    details = {"provider": provider, "original": error}
    if isinstance(error, (sdk.AuthenticationError, sdk.PermissionDeniedError)):
        return LLMAuthError(message, **details)
    if isinstance(error, sdk.RateLimitError):
        return LLMRateLimitError(message, **details)
    if isinstance(error, sdk.APIStatusError):
        status = error.status_code
        error_cls = LLMServerError if status >= 500 else LLMProviderError  # 529 "overloaded" es >= 500
        return error_cls(f"{message} (HTTP {status})", **details)
    if isinstance(error, sdk.APIConnectionError):
        return LLMNetworkError(message, **details)
    return LLMProviderError(message, **details)
```

- [ ] **Step 5: Implementar `openai_client.py`**

`llm_client/openai_client.py`:
```python
"""Cliente asíncrono para la API de OpenAI y endpoints compatibles (por ejemplo Gemini)."""

from collections.abc import AsyncIterator
from typing import Any, ClassVar

import openai
from openai import AsyncOpenAI

from .base import BaseLLMClient
from .errors import LLMError, translate_sdk_error
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
        except openai.APIError as error:
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
```

- [ ] **Step 6: Correr los tests y verificar que pasan**

Run: `uv run pytest tests/test_openai_client.py -v`
Expected: 19 PASSED.

Si `test_construye_el_sdk_sin_reintentos_propios_y_con_base_url` falla porque el SDK expone `max_retries` con otro nombre, revisar `uv run python -c "import openai; c = openai.AsyncOpenAI(api_key='x', max_retries=0); print(c.max_retries, c.base_url)"` y ajustar el test, no la implementación.

- [ ] **Step 7: Commit**

```bash
git add tests/fakes.py llm_client/errors.py llm_client/openai_client.py tests/test_openai_client.py
git commit -m "Agregar OpenAIClient con streaming, traducción de errores y reintentos

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: `AnthropicClient`

**Files:**
- Create: `llm_client/anthropic_client.py`
- Test: `tests/test_anthropic_client.py`

**Interfaces:**
- Consumes: `BaseLLMClient`; `LLMError` y `translate_sdk_error` de `errors.py`; `with_retry`; schemas; y los fakes `FakeStream`, `fake_anthropic_sdk`, `anthropic_response`, `anthropic_text_event`, `anthropic_json_delta_event`, `anthropic_event`, `anthropic_status_error`, `anthropic_connection_error` de `tests/fakes.py`.
- Produces: `AnthropicClient(api_key: str, model: str = "claude-haiku-4-5", client: AsyncAnthropic | None = None, attempts: int = 3, base_delay: float = 1.0)` con atributo público `model`, clase-atributos `DEFAULT_MODEL` y `MAX_TEMPERATURE = 1.0`.

- [ ] **Step 1: Escribir los tests que fallan**

`tests/test_anthropic_client.py`:
```python
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
```

- [ ] **Step 2: Correr los tests y verificar que fallan**

Run: `uv run pytest tests/test_anthropic_client.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'llm_client.anthropic_client'`.

- [ ] **Step 3: Implementar `anthropic_client.py`**

`llm_client/anthropic_client.py`:
```python
"""Cliente asíncrono para la API de Anthropic (Claude)."""

from collections.abc import AsyncIterator
from typing import Any, ClassVar

import anthropic
from anthropic import AsyncAnthropic

from .base import BaseLLMClient
from .errors import LLMError, translate_sdk_error
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
        except anthropic.APIError as error:
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
            request["temperature"] = min(config.temperature, self.MAX_TEMPERATURE)
        return request

    async def _generate_once(self, messages: list[ChatMessage], config: ModelConfig) -> ModelResponse:
        try:
            response = await self._client.messages.create(**self._build_request(messages, config))
        except anthropic.APIError as error:
            raise self._translate(error) from error

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

    async def _open_stream(self, messages: list[ChatMessage], config: ModelConfig):
        try:
            return await self._client.messages.create(
                **self._build_request(messages, config), stream=True
            )
        except anthropic.APIError as error:
            raise self._translate(error) from error

    def _translate(self, error: anthropic.APIError) -> LLMError:
        return translate_sdk_error(error, provider=self.provider, sdk=anthropic)
```

- [ ] **Step 4: Correr los tests y verificar que pasan**

Run: `uv run pytest tests/test_anthropic_client.py -v`
Expected: 20 PASSED.

- [ ] **Step 5: Commit**

```bash
git add llm_client/anthropic_client.py tests/test_anthropic_client.py
git commit -m "Agregar AnthropicClient con streaming, traducción de errores y reintentos

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: `AsyncLLMManager` y la API pública del paquete

**Files:**
- Create: `llm_client/manager.py`
- Modify: `llm_client/__init__.py`
- Test: `tests/test_manager.py`

**Interfaces:**
- Consumes: `OpenAIClient`, `AnthropicClient`, `BaseLLMClient`, `LLMConfigError`, schemas.
- Produces:
  - `AsyncLLMManager(provider: str | None = None)` con atributos `client: BaseLLMClient` y `provider: str`, y métodos `async generate(messages, config=None)` y `stream(messages, config=None)` que delegan.
  - `llm_client` exporta: `AsyncLLMManager`, `BaseLLMClient`, `OpenAIClient`, `AnthropicClient`, `ChatMessage`, `ModelConfig`, `ModelResponse`, `LLMError`, `LLMConfigError`, `LLMAuthError`, `LLMRateLimitError`, `LLMNetworkError`, `LLMServerError`, `LLMProviderError`.

- [ ] **Step 1: Escribir los tests que fallan**

`tests/test_manager.py`:
```python
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
    monkeypatch.setenv("OPENAI_MODEL", "gemini-3.6-flash")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")

    manager = AsyncLLMManager(provider="openai")

    assert manager.client.model == "gemini-3.6-flash"
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
```

- [ ] **Step 2: Correr los tests y verificar que fallan**

Run: `uv run pytest tests/test_manager.py -v`
Expected: FAIL con `ImportError: cannot import name 'AsyncLLMManager' from 'llm_client'`.

- [ ] **Step 3: Implementar `manager.py`**

`llm_client/manager.py`:
```python
"""AsyncLLMManager: elige y construye el proveedor a partir de la configuración.

Es la única pieza del paquete que lee variables de entorno. Los clientes reciben
todo por parámetro, lo que los hace fáciles de probar.
"""

import os
from collections.abc import AsyncIterator

from .anthropic_client import AnthropicClient
from .base import BaseLLMClient
from .errors import LLMConfigError
from .openai_client import OpenAIClient
from .schemas import ChatMessage, ModelConfig, ModelResponse

DEFAULT_PROVIDER = "anthropic"
SUPPORTED_PROVIDERS = ("openai", "anthropic")


class AsyncLLMManager:
    """Punto de entrada: `AsyncLLMManager()` o `AsyncLLMManager(provider="openai")`.

    Sin argumento, usa la variable de entorno LLM_PROVIDER (default: anthropic).
    """

    def __init__(self, provider: str | None = None) -> None:
        name = (provider or os.environ.get("LLM_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
        self.client: BaseLLMClient = self._build_client(name)
        self.provider = self.client.provider

    async def generate(
        self,
        messages: list[ChatMessage],
        config: ModelConfig | None = None,
    ) -> ModelResponse:
        return await self.client.generate(messages, config)

    def stream(
        self,
        messages: list[ChatMessage],
        config: ModelConfig | None = None,
    ) -> AsyncIterator[str]:
        return self.client.stream(messages, config)

    # --- Internos ---------------------------------------------------------------

    @staticmethod
    def _require_env(var: str, provider: str) -> str:
        value = os.environ.get(var, "").strip()
        if not value:
            raise LLMConfigError(
                f"Falta la variable de entorno {var}. Copiá .env.example a .env y completala.",
                provider=provider,
            )
        return value

    def _build_client(self, name: str) -> BaseLLMClient:
        if name == "openai":
            return OpenAIClient(
                api_key=self._require_env("OPENAI_API_KEY", name),
                model=os.environ.get("OPENAI_MODEL") or OpenAIClient.DEFAULT_MODEL,
                base_url=os.environ.get("OPENAI_BASE_URL") or None,
            )
        if name == "anthropic":
            return AnthropicClient(
                api_key=self._require_env("ANTHROPIC_API_KEY", name),
                model=os.environ.get("ANTHROPIC_MODEL") or AnthropicClient.DEFAULT_MODEL,
            )
        raise LLMConfigError(
            f"Proveedor desconocido: {name!r}. Opciones: {', '.join(SUPPORTED_PROVIDERS)}.",
            provider=name,
        )
```

- [ ] **Step 4: Completar `llm_client/__init__.py`**

Reemplazar el contenido por:
```python
"""Unified Async LLM Client: OpenAI y Anthropic bajo una interfaz común y asíncrona.

Uso básico:

    from llm_client import AsyncLLMManager, ChatMessage, ModelConfig

    manager = AsyncLLMManager(provider="anthropic")
    response = await manager.generate([ChatMessage(role="user", content="Hola")])
    async for chunk in manager.stream([ChatMessage(role="user", content="Hola")]):
        print(chunk, end="")
"""

from .anthropic_client import AnthropicClient
from .base import BaseLLMClient
from .errors import (
    LLMAuthError,
    LLMConfigError,
    LLMError,
    LLMNetworkError,
    LLMProviderError,
    LLMRateLimitError,
    LLMServerError,
)
from .manager import AsyncLLMManager
from .openai_client import OpenAIClient
from .schemas import ChatMessage, ModelConfig, ModelResponse

__all__ = [
    "AsyncLLMManager",
    "BaseLLMClient",
    "OpenAIClient",
    "AnthropicClient",
    "ChatMessage",
    "ModelConfig",
    "ModelResponse",
    "LLMError",
    "LLMConfigError",
    "LLMAuthError",
    "LLMRateLimitError",
    "LLMNetworkError",
    "LLMServerError",
    "LLMProviderError",
]
```

- [ ] **Step 5: Correr toda la suite y verificar que pasa**

Run: `uv run pytest -v`
Expected: 81 PASSED, sin warnings de pytest-asyncio.

- [ ] **Step 6: Commit**

```bash
git add llm_client/manager.py llm_client/__init__.py tests/test_manager.py
git commit -m "Agregar AsyncLLMManager y la API pública del paquete

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: Script de validación `main.py` y prueba real

**Files:**
- Create: `main.py`
- Create: `.env` (copia local de `.env.example`; NO se commitea)

**Interfaces:**
- Consumes: `AsyncLLMManager`, `ChatMessage`, `ModelConfig`, `LLMError` de `llm_client`.
- Produces: `python main.py [--provider openai|anthropic]`.

- [ ] **Step 1: Escribir `main.py`**

```python
"""Script de validación: pregunta "¿Qué es la entropía?" a cada proveedor.

Primero en modo normal (los proveedores corren en paralelo con asyncio.gather) y
después en modo streaming (uno a la vez, para que el texto no se mezcle).
Un proveedor sin key o con error imprime un "error controlado" y no frena al otro.
"""

import argparse
import asyncio
import logging
import time

from dotenv import load_dotenv

from llm_client import AsyncLLMManager, ChatMessage, LLMError, ModelConfig

PROVIDERS = ("openai", "anthropic")
QUESTION = "¿Qué es la entropía?"


def build_request() -> tuple[list[ChatMessage], ModelConfig]:
    messages = [ChatMessage(role="user", content=QUESTION)]
    config = ModelConfig(temperature=0.7, max_tokens=200)
    return messages, config


async def run_normal(provider: str) -> None:
    try:
        manager = AsyncLLMManager(provider=provider)
        messages, config = build_request()
        response = await manager.generate(messages, config)
        print(f"\n[{provider} / {response.model}]\n{response.content}")
        print(
            f"[{provider}] tokens: {response.input_tokens} de entrada, "
            f"{response.output_tokens} de salida"
        )
    except LLMError as error:
        print(f"\n[{provider}] error controlado: {error}")


async def run_stream(provider: str) -> None:
    try:
        manager = AsyncLLMManager(provider=provider)
        messages, config = build_request()
        print(f"\n[{provider}] streaming: ", end="", flush=True)
        async for chunk in manager.stream(messages, config):
            print(chunk, end="", flush=True)  # flush: mostrar cada fragmento ya mismo
        print()
    except LLMError as error:
        print(f"\n[{provider}] error controlado: {error}")


async def main(providers: tuple[str, ...]) -> None:
    print("=== Modo normal (proveedores en paralelo) ===")
    start = time.perf_counter()
    await asyncio.gather(*(run_normal(provider) for provider in providers))
    print(f"\nTiempo total del modo normal: {time.perf_counter() - start:.1f}s")

    print("\n=== Modo streaming (un proveedor a la vez) ===")
    for provider in providers:
        await run_stream(provider)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prueba los clientes en modo normal y streaming.")
    parser.add_argument(
        "--provider",
        choices=PROVIDERS,
        help="probar solo este proveedor (por defecto, ambos)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    load_dotenv()  # carga .env al entorno; el código nunca tiene keys escritas
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    asyncio.run(main((args.provider,) if args.provider else PROVIDERS))
```

- [ ] **Step 2: Correr sin keys y verificar el error controlado**

Run: `uv run python main.py; echo "exit=$?"`
Expected (todavía no hay `.env`): dos bloques con `error controlado: [openai] Falta la variable de entorno OPENAI_API_KEY...` y `[anthropic] Falta la variable de entorno ANTHROPIC_API_KEY...`, tiempo total, dos líneas más de error controlado en streaming, y `exit=0`.

- [ ] **Step 3: Crear el `.env` local**

Run: `cp .env.example .env`
Después, el usuario edita `.env` en el editor y pega su key de Gemini en la sección de OpenAI (las tres líneas: `OPENAI_API_KEY`, `OPENAI_BASE_URL` descomentada, `OPENAI_MODEL=gemini-3.6-flash`). La key no se pega en el chat.

Run: `git status --short`
Expected: `.env` NO aparece (está en `.gitignore`).

- [ ] **Step 4: Correr contra la API real**

Run: `uv run python main.py --provider openai`
Expected: una respuesta real sobre entropía en modo normal con conteo de tokens, y la misma pregunta respondida en streaming, con el texto apareciendo de a poco.

Run: `uv run python main.py`
Expected: OpenAI (vía Gemini) responde; Anthropic imprime `error controlado` (sin key todavía, o `LLMAuthError` si quedó el placeholder `sk-ant-...`). Código de salida 0.

- [ ] **Step 5: Commit**

```bash
git add main.py
git commit -m "Agregar main.py: validación en modo normal y streaming

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: README, verificación final y repositorio en GitHub

**Files:**
- Create: `README.md`
- Modify: `pyproject.toml` (agregar `readme = "README.md"`)

**Interfaces:**
- Consumes: todo lo anterior.
- Produces: el entregable completo en GitHub.

- [ ] **Step 1: Escribir `README.md`**

````markdown
# Unified Async LLM Client

Cliente asíncrono en Python 3.12 para hablar con **OpenAI** y **Anthropic** a través
de una interfaz común, con streaming, validación de datos con Pydantic y manejo
controlado de errores de red, autenticación y límite de tasa.

Pre-entrega 1 del curso: "Cliente de LLM robusto y asíncrono".

## Estructura

```
llm_client/
    schemas.py           # ChatMessage, ModelConfig, ModelResponse (Pydantic)
    errors.py            # jerarquía LLMError: errores controlados, con o sin reintento
    retry.py             # with_retry: reintentos con backoff exponencial
    base.py              # BaseLLMClient: el contrato que cumplen los proveedores
    openai_client.py     # OpenAIClient (AsyncOpenAI)
    anthropic_client.py  # AnthropicClient (AsyncAnthropic)
    manager.py           # AsyncLLMManager: elige el proveedor según la configuración
tests/                   # pytest, sin red y sin keys (los SDKs se reemplazan por fakes)
main.py                  # script de validación: modo normal y streaming
.env.example             # variables de entorno necesarias
pyproject.toml           # dependencias y configuración
```

## Instalación

Requiere Python 3.12.

**Con `uv` (recomendado):**

```bash
uv sync          # crea .venv con Python 3.12 e instala todo, incluidas las deps de test
```

**Con `venv` clásico:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # en Windows: .venv\Scripts\activate
pip install -e .
pip install pytest pytest-asyncio
```

## Variables de entorno

Copiá `.env.example` a `.env` y completá lo que vayas a usar. El `.env` está en
`.gitignore`: nunca se sube.

| Variable | Obligatoria | Default | Para qué |
|---|---|---|---|
| `LLM_PROVIDER` | No | `anthropic` | Proveedor cuando el código no indica uno (`openai` o `anthropic`) |
| `OPENAI_API_KEY` | Para usar OpenAI | | Key de OpenAI (platform.openai.com) |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | Modelo de OpenAI |
| `OPENAI_BASE_URL` | No | | Endpoint alternativo compatible con la API de OpenAI |
| `ANTHROPIC_API_KEY` | Para usar Anthropic | | Key de Anthropic (console.anthropic.com) |
| `ANTHROPIC_MODEL` | No | `claude-haiku-4-5` | Modelo de Anthropic |

Ambas plataformas cobran por uso y requieren cargar crédito. Los modelos por defecto
son los más baratos de cada proveedor: probar el script cuesta fracciones de centavo.

### Probar gratis con Gemini

Google expone un endpoint compatible con la API de OpenAI y una key gratuita en
[Google AI Studio](https://aistudio.google.com/apikey). Con esto, `OpenAIClient`
habla con Gemini sin cambiar código:

```
OPENAI_API_KEY=<tu key de Gemini>
OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
OPENAI_MODEL=gemini-3.6-flash
```

## Correr el script de validación

```bash
uv run python main.py                      # ambos proveedores
uv run python main.py --provider openai    # solo uno
```

(Con el venv clásico activado: `python main.py`.)

El script pregunta "¿Qué es la entropía?" a cada proveedor:

1. **Modo normal**, con ambos proveedores en paralelo (`asyncio.gather`), mostrando
   la respuesta, los tokens consumidos y el tiempo total.
2. **Modo streaming**, un proveedor a la vez, mostrando el texto a medida que llega.

Si falta una key, ese proveedor imprime `error controlado: ...` y el otro sigue.
El script siempre termina con código 0.

## Correr los tests

```bash
uv run pytest -v
```

Los tests no tocan la red ni necesitan keys: los clientes reciben el SDK por
inyección de dependencias y en los tests se les pasa un doble (`tests/fakes.py`)
que responde, entrega un stream o lanza excepciones reales del SDK.

## Uso desde código

```python
import asyncio
from dotenv import load_dotenv
from llm_client import AsyncLLMManager, ChatMessage, LLMError, ModelConfig

async def main():
    load_dotenv()
    manager = AsyncLLMManager(provider="anthropic")   # o AsyncLLMManager() con LLM_PROVIDER
    messages = [ChatMessage(role="user", content="¿Qué es la entropía?")]
    config = ModelConfig(temperature=0.7, max_tokens=200)

    try:
        response = await manager.generate(messages, config)
        print(response.content, response.input_tokens, response.output_tokens)

        async for chunk in manager.stream(messages, config):
            print(chunk, end="", flush=True)
    except LLMError as error:
        print("error controlado:", error)

asyncio.run(main())
```

## Cómo funciona el manejo de errores

Cada cliente atrapa las excepciones de su SDK y las traduce a una jerarquía propia,
así el código que usa el paquete solo necesita `except LLMError`:

| Situación | Excepción | ¿Se reintenta? |
|---|---|---|
| Proveedor desconocido o key faltante | `LLMConfigError` | No |
| Key inválida o sin permisos (401/403) | `LLMAuthError` | No |
| Límite de tasa o cuota (429) | `LLMRateLimitError` | Sí |
| Timeout o conexión caída | `LLMNetworkError` | Sí |
| Error del proveedor (5xx, sobrecarga 529) | `LLMServerError` | Sí |
| Otro error HTTP (400, 404...) | `LLMProviderError` | No |

Los errores reintentables se reintentan hasta 3 veces con espera exponencial (1 s, 2 s).
Los SDKs se construyen con `max_retries=0` para que esta sea la única lógica de reintento.
En streaming solo se reintenta la apertura: un corte a mitad de stream se reporta como
error sin reintentar, porque repetir la llamada duplicaría texto ya mostrado.

## Notas y limitaciones

- `temperature` es opcional. Si se indica, se valida en el rango 0 a 2. Anthropic acepta
  0 a 1, así que su cliente la recorta a 1. Los modelos Claude más nuevos (Opus 5,
  Sonnet 5) rechazan el parámetro: con ellos hay que dejarla en `None`.
- Anthropic recibe el system prompt como parámetro aparte; el cliente extrae los
  mensajes con rol `system` y los concatena con `ModelConfig.system_prompt`.
- Solo texto: sin imágenes, herramientas ni salida estructurada.
- Los modelos de razonamiento de OpenAI (serie o, GPT-5) exigen `max_completion_tokens`
  en vez de `max_tokens` y no están soportados como default.
````

- [ ] **Step 2: Referenciar el README desde `pyproject.toml`**

En `[project]`, agregar debajo de `description`:
```toml
readme = "README.md"
```

- [ ] **Step 3: Verificación final**

Run: `uv sync && uv run pytest -v`
Expected: todos PASSED.

Run: `uv run python main.py`
Expected: la salida descrita en Task 8, código 0.

Run: `git status --short`
Expected: solo `README.md` y `pyproject.toml` modificados; `.env` ausente.

- [ ] **Step 4: Commit**

```bash
git add README.md pyproject.toml
git commit -m "Agregar README con instalación, variables de entorno y uso

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

- [ ] **Step 5: Crear el repositorio en GitHub y subir**

Confirmar con el usuario el nombre del repo y si es público o privado antes de correr:

```bash
gh repo create pre-entrega-1-llm-client --public --source=. --push
```

Expected: `gh` crea el repo en la cuenta autenticada, configura `origin` y sube `main`.

Run: `gh repo view --web`
Expected: se abre el repo en el navegador con el README renderizado y sin `.env`.

---

## Cambios decididos durante la ejecución

Las revisiones por tarea encontraron defectos en el plan (no en la implementación) y se
resolvieron con estas decisiones. El código final del repo es la referencia; acá queda el
porqué de cada desvío respecto de los bloques de código de arriba.

- **Task 5, `errors.py`:** `translate_sdk_error` también importa `httpx2` (con fallback a
  `httpx`) y mapea `httpx.HTTPError` / `httpx.StreamError` a `LLMNetworkError` antes del
  fallback final. Motivo: durante la iteración de un stream los SDKs no envuelven los
  errores de transporte. `httpx2` pasa a ser dependencia directa (`uv add httpx2`).
- **Task 5 y 6, `stream()`:** el `except` que envuelve el `async for` atrapa `Exception`
  (no solo `APIError`) y traduce. Los `except` de `_generate_once` y `_open_stream`
  siguen atrapando solo `APIError`.
- **Task 5, `_generate_once`:** guarda `if not response.choices: raise LLMProviderError(...)`
  antes de leer `choices[0]`; y el parseo completo (hasta el `return ModelResponse`) va en
  un `try/except Exception` que lanza `LLMProviderError("Respuesta inesperada del
  proveedor: ...")` con la excepción original encadenada. Lo mismo en Task 6.
- **Task 6, temperatura:** anthropic 1.4.0 no acepta `temperature` en `messages.create()`
  (es un `TypeError`). Se envía `extra_body={"temperature": min(valor, 1.0)}`, que es lo
  que documenta la guía de migración del SDK para modelos que aún lo aceptan (Haiku 4.5).
  Los tests de temperatura verifican `request["extra_body"]` y que no haya `temperature`
  a nivel superior.
- **Task 5 y 6, tests:** `tests/fakes.py` usa el fallback `httpx2`/`httpx` también para
  OpenAI (openai 3.8.0 usa `httpx2`; `httpx` no está instalado) y agrega
  `transport_error()`. Cada cliente suma: un test de error de transporte a mitad de stream
  (`LLMNetworkError`), uno de `RuntimeError` a mitad de stream (`LLMProviderError`), uno de
  respuesta con forma inesperada (`LLMProviderError`), y
  `test_los_parametros_enviados_existen_en_el_sdk_real`, que valida los kwargs capturados
  por el fake contra la firma real del SDK con `inspect.signature(...).bind(...)`. OpenAI
  suma además el test de `choices` vacío.
- **Task 8, `main.py`:** `run_normal` y `run_stream` agregan un `except Exception` final
  que imprime `error inesperado: ...`, porque una excepción no atrapada dentro de
  `asyncio.gather` cancela el script entero y deja al otro proveedor sin terminar.
- **Task 9, README:** la nota sobre temperatura menciona el envío por `extra_body`.
- **Task 8, `main.py` (tras la prueba real con Gemini):** `max_tokens` pasa de 200 a 4096 y se
  agrega `system_prompt`, porque Gemini 3.6 Flash razona antes de responder y con 200 tokens la
  respuesta llegaba cortada (`finish_reason="length"`); la salida muestra `finish_reason`. El
  modelo gratuito de ejemplo pasa a `gemini-3.6-flash` (Google retiró 2.5 para cuentas nuevas).
