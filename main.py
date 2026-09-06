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
    except Exception as error:  # red de seguridad: un bug no debe tirar abajo al otro proveedor
        print(f"\n[{provider}] error inesperado: {type(error).__name__}: {error}")


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
    except Exception as error:  # red de seguridad: un bug no debe tirar abajo al otro proveedor
        print(f"\n[{provider}] error inesperado: {type(error).__name__}: {error}")


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
