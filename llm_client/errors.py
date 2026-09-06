"""Excepciones propias del cliente.

Cada cliente atrapa las excepciones de su SDK y las traduce a una de estas.
Así, quien usa el paquete solo necesita `except LLMError` y nunca depende de
los SDKs. El atributo `retryable` le dice a `with_retry` si vale la pena insistir.
"""

from types import ModuleType


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
