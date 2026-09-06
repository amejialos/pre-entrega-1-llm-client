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
