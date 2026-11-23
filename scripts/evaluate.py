"""Evaluate retrieval quality against a JSONL question set.

Each line: {"question": "...", "expected_filenames": ["policy.md"], "top_k": 5}
Environment: RAG_API_URL, RAG_API_TOKEN, RAG_ORGANIZATION_ID.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any


def percentile(values: list[float], quantile: float) -> float:
    """Nearest-rank percentile; quantile must be in [0, 1]."""
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between 0 and 1")
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, int(len(ordered) * quantile + 0.999999))
    return ordered[min(len(ordered) - 1, rank - 1)]


def score_result(expected: set[str], citations: list[dict[str, Any]]) -> tuple[float, float]:
    """Return hit@k and reciprocal rank using citation filenames."""
    if not expected:
        return 0.0, 0.0
    for rank, citation in enumerate(citations, start=1):
        if citation.get("filename") in expected:
            return 1.0, 1.0 / rank
    return 0.0, 0.0


def query_api(base_url: str, token: str, organization_id: str, question: str, top_k: int) -> dict[str, Any]:
    payload = json.dumps({"question": question, "top_k": top_k}).encode()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/query",
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "X-Organization-ID": organization_id,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())


def main() -> int:
    parser = argparse.ArgumentParser(description="Run retrieval evaluation against a deployed RAG API.")
    parser.add_argument("--dataset", default="eval/sample.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if args.top_k < 1 or args.top_k > 20:
        parser.error("--top-k must be between 1 and 20")

    base_url = os.environ.get("RAG_API_URL", "http://localhost:8000")
    token = os.environ.get("RAG_API_TOKEN")
    organization_id = os.environ.get("RAG_ORGANIZATION_ID")
    if not token or not organization_id:
        parser.error("Set RAG_API_TOKEN and RAG_ORGANIZATION_ID.")

    rows = [json.loads(line) for line in Path(args.dataset).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        parser.error("Dataset is empty.")
    hits: list[float] = []
    reciprocal_ranks: list[float] = []
    latencies: list[float] = []

    for index, row in enumerate(rows, start=1):
        if not isinstance(row.get("question"), str) or not row["question"].strip():
            parser.error(f"Dataset row {index} has no valid question.")
        top_k = int(row.get("top_k", args.top_k))
        if not 1 <= top_k <= 20:
            parser.error(f"Dataset row {index} top_k must be between 1 and 20.")
        started = time.perf_counter()
        result = query_api(base_url, token, organization_id, row["question"], top_k)
        latencies.append((time.perf_counter() - started) * 1000)
        hit, reciprocal_rank = score_result(set(row.get("expected_filenames", [])), result.get("citations", []))
        hits.append(hit)
        reciprocal_ranks.append(reciprocal_rank)

    report = {
        "queries": len(rows),
        "hit_rate_at_k": round(statistics.mean(hits), 4),
        "mrr": round(statistics.mean(reciprocal_ranks), 4),
        "latency_ms_p50": round(percentile(latencies, 0.50), 2),
        "latency_ms_p95": round(percentile(latencies, 0.95), 2),
        "top_k_default": args.top_k,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
