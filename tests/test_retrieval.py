from types import SimpleNamespace

from app.services.embeddings import local_embedding
from app.services.retrieval import cosine_similarity, hybrid_rank, lexical_score


def test_local_embedding_is_deterministic_and_normalized():
    first = local_embedding("retrieval augmented generation", 64)
    second = local_embedding("retrieval augmented generation", 64)
    assert first == second
    assert abs(sum(value * value for value in first) - 1) < 1e-6


def test_cosine_similarity():
    assert cosine_similarity([1, 0], [1, 0]) == 1
    assert cosine_similarity([1, 0], [0, 1]) == 0


def test_lexical_score():
    assert lexical_score("refund policy", "The refund policy is documented") == 1


def test_hybrid_rank_returns_relevant_chunk():
    docs = {"d1": SimpleNamespace(id="d1", filename="policy.md")}
    chunks = [
        SimpleNamespace(id="c1", document_id="d1", content="Refund policy allows returns", embedding=local_embedding("refund policy allows returns", 64)),
        SimpleNamespace(id="c2", document_id="d1", content="The weather is sunny", embedding=local_embedding("the weather is sunny", 64)),
    ]
    result = hybrid_rank("refund policy", chunks, docs, 1)
    assert result[0]["chunk"].id == "c1"
