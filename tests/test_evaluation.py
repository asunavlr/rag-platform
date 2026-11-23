import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "evaluate.py"
SPEC = importlib.util.spec_from_file_location("rag_evaluate", MODULE_PATH)
assert SPEC and SPEC.loader
evaluate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluate)


def test_score_result_returns_hit_and_reciprocal_rank():
    citations = [
        {"filename": "other.md"},
        {"filename": "policy.md"},
    ]
    assert evaluate.score_result({"policy.md"}, citations) == (1.0, 0.5)


def test_score_result_returns_zero_when_not_found():
    assert evaluate.score_result({"policy.md"}, [{"filename": "other.md"}]) == (0.0, 0.0)


def test_percentile_handles_empty_and_single_value():
    assert evaluate.percentile([], 0.95) == 0.0
    assert evaluate.percentile([12.5], 0.95) == 12.5
