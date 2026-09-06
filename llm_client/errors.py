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
