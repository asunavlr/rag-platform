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


def hybrid_rank(
    query: str,
    chunks: list[Chunk],
    documents: dict[str, Document],
    top_k: int,
    query_vector: list[float] | None = None,
) -> list[dict]:
    if not chunks:
        return []
    vector = query_vector or local_embedding(query, len(chunks[0].embedding))
    semantic = sorted(chunks, key=lambda c: cosine_similarity(vector, c.embedding), reverse=True)
    lexical_order = sorted(chunks, key=lambda c: lexical_score(query, c.content), reverse=True)
    semantic_rank = {chunk.id: rank for rank, chunk in enumerate(semantic, 1)}
    lexical_rank = {chunk.id: rank for rank, chunk in enumerate(lexical_order, 1)}
    fused = []
    for chunk in chunks:
        rrf = 1 / (60 + semantic_rank[chunk.id]) + 1 / (60 + lexical_rank[chunk.id])
        semantic_score = cosine_similarity(vector, chunk.embedding)
        term_score = lexical_score(query, chunk.content)
        score = rrf * 1000 + max(semantic_score, 0) * 0.25 + term_score * 0.25
        fused.append({"chunk": chunk, "document": documents[chunk.document_id], "score": score})
    return sorted(fused, key=lambda item: item["score"], reverse=True)[:top_k]
