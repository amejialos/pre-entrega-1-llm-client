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
OPENAI_MODEL=gemini-2.5-flash
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
  0 a 1, así que su cliente la recorta a 1. El SDK 1.x de Anthropic ya no expone
  `temperature` como parámetro tipado, así que el cliente lo envía por `extra_body`.
  Los modelos Claude más nuevos (Opus 5, Sonnet 5) rechazan el parámetro: con ellos
  hay que dejarla en `None`.
- Anthropic recibe el system prompt como parámetro aparte; el cliente extrae los
  mensajes con rol `system` y los concatena con `ModelConfig.system_prompt`.
- Solo texto: sin imágenes, herramientas ni salida estructurada.
- Los modelos de razonamiento de OpenAI (serie o, GPT-5) exigen `max_completion_tokens`
  en vez de `max_tokens` y no están soportados como default.
