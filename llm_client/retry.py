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
