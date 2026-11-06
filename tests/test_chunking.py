import pytest

from app.services.chunking import split_text


def test_empty_text_returns_no_chunks():
    assert split_text("   ") == []


def test_chunks_cover_text_with_overlap():
    text = " ".join(f"word{i}" for i in range(100))
    chunks = split_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(chunks)
    assert chunks[0][-10:] in chunks[1]


def test_invalid_overlap_is_rejected():
    with pytest.raises(ValueError):
        split_text("hello", chunk_size=10, overlap=10)
