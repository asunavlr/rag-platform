# Retrieval evaluation

A RAG system should be measured against a small, reviewed golden dataset rather than judged from a few manual prompts.

## Run

1. Start the API and create an organization.
2. Upload the documents referenced by the dataset.
3. Export the organization token and ID.

```bash
export RAG_API_URL=http://localhost:8000
export RAG_API_TOKEN=...
export RAG_ORGANIZATION_ID=...
python scripts/evaluate.py --dataset eval/sample.jsonl --top-k 5
```

Each JSONL row contains a question, expected source filenames and optional top-k override. The report includes hit rate at k, mean reciprocal rank (MRR), and p50/p95 request latency.

## Metric definitions
- **Hit rate@k:** fraction of questions where at least one expected filename appears in the top-k citations.
- **MRR:** mean reciprocal rank of the first expected source; zero when no expected source is retrieved.
- **Latency:** end-to-end API request time, including embedding, retrieval and generation.

The sample dataset is illustrative. Replace it with questions and expected sources reviewed by a domain expert. Track dataset version, model/provider, chunking configuration and commit SHA alongside each benchmark. Do not interpret these retrieval metrics as proof of answer faithfulness; add human-reviewed faithfulness and answer-relevance labels before production decisions.
