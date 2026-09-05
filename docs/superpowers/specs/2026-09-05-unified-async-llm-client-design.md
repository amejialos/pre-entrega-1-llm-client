# Unified Async LLM Client: diseño

Fecha: 2026-09-05
Entrega: Pre-entrega 1, "Cliente de LLM robusto y asíncrono"

## 1. Objetivo

Un paquete Python 3.12 que permite hablar con OpenAI y con Anthropic a través de una
interfaz común, de forma asíncrona, con soporte de streaming, validación de datos con
Pydantic y manejo controlado de errores de red, autenticación y límite de tasa.

Lo que la consigna pide y dónde queda resuelto:

| Requisito de la consigna | Dónde se resuelve |
|---|---|
| Intercambiabilidad OpenAI / Anthropic bajo una interfaz común | `base.py` (`BaseLLMClient`) + `manager.py` (`AsyncLLMManager`) |
| Todas las llamadas no bloqueantes (`async/await`) | `AsyncOpenAI` y `AsyncAnthropic`; ningún cliente síncrono en el paquete |
| Método que devuelve un generador asíncrono de tokens | `stream()` en cada cliente, con `yield` dentro de `async for` |
| Pydantic para mensajes y configuración (temperatura 0 a 2, `max_tokens`) | `schemas.py` |
| `AsyncLLMManager` que elige proveedor por variable de configuración | `manager.py`, variable `LLM_PROVIDER` |
| Capturar errores de red y rate limit sin crash | `errors.py` + `retry.py` + traducción de excepciones en cada cliente |
| `main.py` que pruebe modo normal y streaming con "¿Qué es la entropía?" | `main.py` |
| `.env.example` y `README.md` con instrucciones | raíz del repo |

## 2. Estructura del repositorio

```
llm_client/
    __init__.py          # re-exporta la API pública del paquete
    schemas.py           # ChatMessage, ModelConfig, ModelResponse (Pydantic)
    errors.py            # jerarquía LLMError
    retry.py             # with_retry: reintentos con backoff exponencial
    base.py              # BaseLLMClient (ABC)
    openai_client.py     # OpenAIClient
    anthropic_client.py  # AnthropicClient
    manager.py           # AsyncLLMManager
tests/
    conftest.py          # fakes compartidos de los SDKs
    test_schemas.py
    test_retry.py
    test_openai_client.py
    test_anthropic_client.py
    test_manager.py
main.py                  # script de validación
pyproject.toml           # metadatos, Python >= 3.12, dependencias
.env.example
.gitignore
README.md
docs/superpowers/specs/  # este documento
```

Regla de diseño: **cada módulo responde una sola pregunta** y se puede entender y probar
sin leer los demás. Los clientes no leen variables de entorno; solo el manager lo hace.

## 3. Componentes

### 3.1 `schemas.py`

```python
Role = Literal["system", "user", "assistant"]
Provider = Literal["openai", "anthropic"]

class ChatMessage(BaseModel):
    role: Role
    content: str                      # min_length=1

class ModelConfig(BaseModel):
    model: str | None = None          # None: el cliente usa su modelo por defecto
    temperature: float | None = None  # None: no se envía; si se envía, 0 <= x <= 2
    max_tokens: int = 1024            # > 0
    system_prompt: str | None = None

class ModelResponse(BaseModel):
    content: str
    model: str
    provider: Provider
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str | None = None
```

Por qué `temperature` es opcional: OpenAI acepta 0 a 2, Anthropic acepta 0 a 1, y los
modelos Claude más nuevos (Opus 5, Sonnet 5) rechazan el parámetro por completo. Si no se
indica, no se envía y funciona en todos lados. La validación 0 a 2 que pide la consigna
se aplica cuando sí se indica.

Los modelos usan `model_config = ConfigDict(extra="forbid")` para que un campo mal
escrito (`temprature`) falle en vez de ignorarse en silencio.

### 3.2 `errors.py`

```python
class LLMError(Exception):
    retryable: bool = False
    def __init__(self, message: str, *, provider: str, original: Exception | None = None)

class LLMConfigError(LLMError): ...      # proveedor desconocido, API key ausente
class LLMAuthError(LLMError): ...        # 401 / 403
class LLMRateLimitError(LLMError):       # 429
    retryable = True
class LLMNetworkError(LLMError):         # timeout, conexión caída
    retryable = True
class LLMServerError(LLMError):          # 5xx y 529 (sobrecarga)
    retryable = True
class LLMProviderError(LLMError): ...    # cualquier otro error HTTP (400, 404, ...)
```

`str(error)` devuelve `"[provider] mensaje"`. `original` conserva la excepción del SDK
para depuración; se encadena con `raise ... from original`.

### 3.3 `retry.py`

```python
async def with_retry(
    operation: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    base_delay: float = 1.0,
) -> T
```

Ejecuta `operation()`. Si lanza un `LLMError` con `retryable = True` y quedan intentos,
espera `base_delay * 2 ** (n - 1)` segundos (1 s, 2 s) y reintenta. Si el error no es
reintentable, o se agotaron los intentos, relanza el mismo error. Registra cada reintento
con `logging` a nivel `WARNING`. No conoce ningún SDK: solo entiende `LLMError`.

### 3.4 `base.py`

```python
class BaseLLMClient(ABC):
    provider: ClassVar[Provider]

    @abstractmethod
    async def generate(self, messages: list[ChatMessage], config: ModelConfig | None = None) -> ModelResponse: ...

    @abstractmethod
    def stream(self, messages: list[ChatMessage], config: ModelConfig | None = None) -> AsyncIterator[str]: ...
```

`generate` es una corrutina (devuelve un valor completo). `stream` es un generador
asíncrono (usa `yield`). Son métodos separados porque Python no permite que una misma
función sea las dos cosas. Si `config` es `None`, el cliente usa `ModelConfig()`.

### 3.5 `openai_client.py`

```python
class OpenAIClient(BaseLLMClient):
    provider = "openai"
    def __init__(self, api_key: str, model: str = "gpt-4o-mini",
                 base_url: str | None = None, client: AsyncOpenAI | None = None)
```

- Construye `AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=0)` salvo que se
  inyecte un `client` (para tests). `max_retries=0` desactiva los reintentos internos del
  SDK para que `with_retry` sea la única lógica de reintento.
- `base_url` permite apuntar al endpoint compatible de Gemini (ver sección 5).
- Conversión de mensajes: `config.system_prompt`, si existe, se inserta como primer
  mensaje con rol `system`. El resto se pasa como `{"role", "content"}`.
- `generate`: `await with_retry(lambda: self._generate_once(messages, config))`.
  `_generate_once` llama a `chat.completions.create(...)` y construye `ModelResponse`
  desde `choices[0].message.content`, `usage.prompt_tokens`, `usage.completion_tokens`,
  `choices[0].finish_reason`, `response.model`.
- `stream`: abre el stream con `with_retry(lambda: self._open_stream(...))`, donde
  `_open_stream` hace `await chat.completions.create(..., stream=True)`. Luego itera
  `async for chunk in stream`, y hace `yield chunk.choices[0].delta.content` cuando
  `choices` no está vacío y `content` no es `None`. Cierra el stream en un `finally`.
- Parámetros enviados: `model`, `messages`, `max_tokens`, y `temperature` solo si no es
  `None`.
- Traducción de errores (`_translate`), en orden de más específico a más general:

| Excepción del SDK `openai` | Nuestra excepción |
|---|---|
| `AuthenticationError`, `PermissionDeniedError` | `LLMAuthError` |
| `RateLimitError` | `LLMRateLimitError` |
| `APIStatusError` con `status_code >= 500` | `LLMServerError` |
| `APIStatusError` (resto: 400, 404, ...) | `LLMProviderError` |
| `APIConnectionError` (incluye `APITimeoutError`) | `LLMNetworkError` |

Los errores que ocurren durante la iteración del stream se traducen igual pero no se
reintentan, porque ya se entregó texto al consumidor.

### 3.6 `anthropic_client.py`

```python
class AnthropicClient(BaseLLMClient):
    provider = "anthropic"
    def __init__(self, api_key: str, model: str = "claude-haiku-4-5",
                 client: AsyncAnthropic | None = None)
```

- Construye `AsyncAnthropic(api_key=api_key, max_retries=0)` salvo inyección.
- Conversión de mensajes: Anthropic no acepta rol `system` en la lista. Los mensajes con
  rol `system` se extraen y se concatenan (separados por línea en blanco) junto con
  `config.system_prompt` en el parámetro `system`. Si no hay nada, no se envía `system`.
- `temperature`: si viene, se recorta a `min(valor, 1.0)` porque Anthropic acepta 0 a 1.
- `generate`: `messages.create(...)`. `content` se arma concatenando los bloques de tipo
  `text` de `response.content`. Tokens desde `usage.input_tokens` y
  `usage.output_tokens`; `finish_reason` desde `stop_reason`.
- `stream`: `await messages.create(..., stream=True)` devuelve un iterador de eventos.
  Se hace `yield event.delta.text` para cada evento `content_block_delta` cuyo
  `delta.type == "text_delta"`. Misma estructura de reintento y cierre que OpenAI.
- Traducción de errores: misma tabla que OpenAI con las clases de `anthropic`
  (`AuthenticationError`, `PermissionDeniedError`, `RateLimitError`, `APIStatusError`,
  `APIConnectionError`). El 529 de sobrecarga es un `APIStatusError` con
  `status_code >= 500`, así que cae en `LLMServerError`.

### 3.7 `manager.py`

```python
class AsyncLLMManager:
    def __init__(self, provider: str | None = None)
    async def generate(self, messages, config=None) -> ModelResponse
    def stream(self, messages, config=None) -> AsyncIterator[str]
    client: BaseLLMClient   # atributo público, útil para tests e inspección
```

Al construirse:

1. `provider = provider or os.environ.get("LLM_PROVIDER", "anthropic")`, normalizado a
   minúsculas.
2. Si no es `"openai"` ni `"anthropic"`: `LLMConfigError`.
3. Lee la key (`OPENAI_API_KEY` / `ANTHROPIC_API_KEY`). Si falta o está vacía:
   `LLMConfigError` con un mensaje que nombra la variable faltante.
4. Lee el modelo (`OPENAI_MODEL` / `ANTHROPIC_MODEL`); si no está, usa el default del
   cliente.
5. Para OpenAI, lee también `OPENAI_BASE_URL` (opcional).
6. Construye el cliente y lo guarda en `self.client`.

`generate` y `stream` solo delegan en `self.client`. El manager no reintenta ni traduce
nada: esa responsabilidad es de los clientes.

### 3.8 `__init__.py`

Re-exporta: `AsyncLLMManager`, `BaseLLMClient`, `OpenAIClient`, `AnthropicClient`,
`ChatMessage`, `ModelConfig`, `ModelResponse`, `LLMError` y sus subclases.

## 4. Flujo de una llamada

Modo normal:

```
main.py
  -> AsyncLLMManager("anthropic")        lee env, construye AnthropicClient
  -> manager.generate(messages, config)  delega
  -> AnthropicClient.generate            with_retry(_generate_once)
     -> _generate_once                   await sdk.messages.create(...)
        -> excepción del SDK             _translate -> LLMError (raise from)
        -> respuesta del SDK             -> ModelResponse
  <- ModelResponse
```

Modo streaming:

```
main.py: async for chunk in manager.stream(...)
  -> AnthropicClient.stream
     -> with_retry(_open_stream)         await sdk.messages.create(..., stream=True)
     -> async for event in stream        yield texto por cada text_delta
        -> excepción a mitad de stream   _translate -> LLMError (sin reintento)
     -> finally: cerrar stream
```

## 5. Configuración

Variables de entorno (todas leídas solo por `AsyncLLMManager`):

| Variable | Obligatoria | Default | Uso |
|---|---|---|---|
| `LLM_PROVIDER` | No | `anthropic` | Proveedor cuando no se pasa explícito |
| `OPENAI_API_KEY` | Si se usa OpenAI | | Key de OpenAI (o de Gemini, ver abajo) |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | Modelo de OpenAI |
| `OPENAI_BASE_URL` | No | (la de OpenAI) | Endpoint alternativo compatible |
| `ANTHROPIC_API_KEY` | Si se usa Anthropic | | Key de Anthropic |
| `ANTHROPIC_MODEL` | No | `claude-haiku-4-5` | Modelo de Anthropic |

Probar el camino de OpenAI gratis con Gemini: Google expone un endpoint compatible con la
API de OpenAI. Con una key gratuita de Google AI Studio, el `.env` queda:

```
OPENAI_API_KEY=<key de Gemini>
OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
OPENAI_MODEL=gemini-2.5-flash
```

`OpenAIClient` no cambia. El README documenta este uso.

## 6. `main.py`

1. `load_dotenv()`.
2. Argumento opcional `--provider {openai,anthropic}`; sin él, prueba ambos.
3. Pregunta: un `ChatMessage(role="user", content="¿Qué es la entropía?")` con
   `ModelConfig(temperature=0.7, max_tokens=200)`.
4. Bloque 1, modo normal: lanza `run_normal(provider)` para cada proveedor con
   `asyncio.gather` y muestra el tiempo total, para evidenciar la concurrencia.
5. Bloque 2, modo streaming: `run_stream(provider)` secuencial, imprimiendo cada
   fragmento con `end=""` y `flush=True`.
6. Cada `run_*` envuelve todo (incluida la construcción del manager) en
   `try/except LLMError` e imprime `error controlado: ...`. Un proveedor que falla no
   afecta al otro. El script siempre termina con código 0.

Sin keys configuradas, la salida son cuatro mensajes de error controlado. Con keys, las
respuestas reales.

## 7. Tests

Sin red, sin keys. Herramientas: `pytest`, `pytest-asyncio` en modo `auto` (no hace
falta decorar cada test).

Estrategia: inyección de dependencias. Los fakes de `tests/conftest.py` imitan la forma
mínima de cada SDK (`chat.completions.create` / `messages.create`) y se configuran para
devolver una respuesta, devolver un stream de fragmentos, o lanzar una secuencia de
excepciones reales del SDK (por ejemplo `openai.RateLimitError`) construidas con
respuestas HTTP falsas. Los fakes registran cuántas veces se los llamó y con qué
argumentos.

| Archivo | Casos |
|---|---|
| `test_schemas.py` | `temperature` 2.5 y -0.1 rechazadas; 0, 1, 2 aceptadas; `None` aceptado. `max_tokens` 0 rechazado. `role` inválido rechazado. `content` vacío rechazado. Campo desconocido rechazado. |
| `test_retry.py` | Éxito al primer intento no espera. Error reintentable: reintenta y devuelve el resultado del segundo intento. Se rinde tras `attempts` y relanza. Error no reintentable: no reintenta. Delays 1, 2 con `base_delay=1` (se parchea `asyncio.sleep`). |
| `test_openai_client.py` | `generate` construye `ModelResponse` correcto. `system_prompt` se inserta como primer mensaje `system`. `temperature=None` no se envía. `stream` entrega los fragmentos en orden e ignora chunks sin contenido. Cada excepción del SDK se traduce a la clase esperada. `RateLimitError` seguido de éxito: se reintenta (con `base_delay=0`). `AuthenticationError`: no se reintenta (una sola llamada). `base_url` se pasa al SDK. |
| `test_anthropic_client.py` | `generate` correcto. Mensajes `system` y `system_prompt` viajan en el parámetro `system`, no en `messages`. `temperature=1.5` se envía como 1.0. `stream` solo emite `text_delta`. Traducción de errores y reintento como en OpenAI. |
| `test_manager.py` | Proveedor desconocido: `LLMConfigError`. Key faltante: `LLMConfigError` que nombra la variable. `LLM_PROVIDER=openai` construye `OpenAIClient`. Argumento explícito gana sobre la variable. `OPENAI_MODEL` y `OPENAI_BASE_URL` llegan al cliente. `generate` y `stream` delegan al cliente (se reemplaza `manager.client` por un fake). |

## 8. README

Secciones: qué es el proyecto; estructura con una línea por archivo; instalación con
`uv` (`uv sync`) y con `venv` clásico (`python3.12 -m venv .venv`, `pip install -e
".[dev]"`); variables de entorno y dónde conseguir cada key (incluido el uso gratuito
con Gemini); cómo correr `main.py` y qué salida esperar con y sin keys; cómo correr los
tests; cómo funciona el manejo de errores y reintentos; limitaciones conocidas.

## 9. Decisiones tomadas y alternativas descartadas

- **Excepciones propias en vez de `ModelResponse` con campo `error`.** Con streaming no
  hay forma limpia de devolver un objeto, y un error dentro de una respuesta vacía se
  ignora fácil. Una jerarquía `LLMError` se atrapa con un solo `except`.
- **`max_retries=0` en los SDKs.** Ambos reintentan solos por defecto. Con dos capas de
  reintento, el comportamiento es impredecible y no se puede testear.
- **Reintento solo al abrir el stream.** Reintentar a mitad de stream duplicaría texto.
- **`create(stream=True)` en ambos SDKs** en vez del helper `messages.stream()` de
  Anthropic. Hace simétrica la estructura de los dos clientes y permite reintentar la
  apertura con la misma función.
- **Sin cliente de Gemini propio.** El endpoint compatible cubre el caso de prueba
  gratuita sin agregar un SDK.
- **Modelos baratos por defecto** (Haiku 4.5, GPT-4o-mini) por costo de prueba.

## 10. Limitaciones conocidas

- Solo contenido de texto: sin imágenes, herramientas ni respuestas estructuradas.
- Sin historial de conversación gestionado: el llamador arma la lista de mensajes.
- `max_tokens` se envía con ese nombre; los modelos de razonamiento de OpenAI (serie o,
  GPT-5) exigen `max_completion_tokens` y no están soportados como default.
- Los modelos Claude que rechazan `temperature` fallan con `LLMProviderError` si se
  indica una temperatura; la solución es dejarla en `None`.
