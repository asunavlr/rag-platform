import math
import re

from app.models import Chunk, Document
from app.services.embeddings import local_embedding


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    denominator = math.sqrt(sum(a * a for a in left) * sum(b * b for b in right))
    return numerator / denominator if denominator else 0.0


def lexical_score(query: str, content: str) -> float:
    terms = set(re.findall(r"[\w'-]+", query.lower()))
    words = re.findall(r"[\w'-]+", content.lower())
    if not terms or not words:
        return 0.0
    return sum(1 for term in terms if term in words) / len(terms)


def hybrid_rank(query: str, chunks: list[Chunk], documents: dict[str, Document], top_k: int) -> list[dict]:
    query_vector = local_embedding(query, len(chunks[0].embedding)) if chunks else []
    semantic = sorted(chunks, key=lambda c: cosine_similarity(query_vector, c.embedding), reverse=True)
    lexical = sorted(chunks, key=lambda c: lexical_score(query, c.content), reverse=True)
    semantic_rank = {chunk.id: rank for rank, chunk in enumerate(semantic, 1)}
    lexical_rank = {chunk.id: rank for rank, chunk in enumerate(lexical, 1)}
    fused = []
    for chunk in chunks:
        rrf = 1 / (60 + semantic_rank[chunk.id]) + 1 / (60 + lexical_rank[chunk.id])
        semantic_score = cosine_similarity(query_vector, chunk.embedding)
        lexical = lexical_score(query, chunk.content)
        score = rrf * 1000 + max(semantic_score, 0) * 0.25 + lexical * 0.25
        fused.append({
            "chunk": chunk,
            "document": documents[chunk.document_id],
            "score": score,
        })
    return sorted(fused, key=lambda item: item["score"], reverse=True)[:top_k]
