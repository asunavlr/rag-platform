from scripts.evaluate import percentile, score_result


def test_nearest_rank_percentiles():
    assert percentile([10, 20, 30, 40], 0.50) == 20
    assert percentile([10, 20, 30, 40], 0.95) == 40
    assert percentile([], 0.95) == 0


def test_reciprocal_rank_and_hit_rate():
    hit, reciprocal_rank = score_result(
        {"guide.md"}, [{"filename": "other.md"}, {"filename": "guide.md"}]
    )
    assert hit == 1
    assert reciprocal_rank == 0.5
