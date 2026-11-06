import hashlib
import math
import re

import httpx

from app.config import settings


def local_embedding(text: str, dimensions: int | None = None) -> list[float]:
    size = dimensions or settings.embedding_dimensions
    vector = [0.0] * size
    tokens = re.findall(r"[\w'-]+", text.lower())
    for token in tokens:
        digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % size
        sign = 1.0 if digest[4] % 2 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


async def embed_texts(texts: list[str]) -> list[list[float]]:
    if settings.embedding_provider != "openai":
        return [local_embedding(text) for text in texts]
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required for the OpenAI embedding provider")
    async with httpx.AsyncClient(timeout=45) as client:
        response = await client.post(
            f"{settings.openai_base_url.rstrip('/')}/embeddings",
            headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            json={"model": settings.embedding_model, "input": texts},
        )
        response.raise_for_status()
        data = response.json()["data"]
        return [item["embedding"] for item in sorted(data, key=lambda item: item["index"])]
