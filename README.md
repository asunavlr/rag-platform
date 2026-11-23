# RAG Platform

A multi-tenant Retrieval-Augmented Generation (RAG) API built with FastAPI. It ingests documents, chunks and embeds their content, performs hybrid retrieval (semantic + lexical), reranks candidates, and generates grounded answers with source citations.

**Project status:** production-oriented portfolio reference implementation. PostgreSQL uses pgvector for approximate-nearest-neighbour candidate retrieval; SQLite remains available for fast local tests. The default feature-hashing embedding is deterministic and useful only as a zero-cost smoke-test baseline. Configure a semantic embedding provider and a grounded LLM endpoint before evaluating answer quality.

## Architecture

```text
Client
  |  Bearer API token + X-Organization-ID
  v
FastAPI
  |-- Document ingestion -> text extraction -> chunking
  |                         -> embeddings -> persistence
  |-- Query -> semantic retrieval + lexical retrieval
  |          -> reciprocal-rank fusion -> reranking
  |          -> grounded LLM answer + citations
  v
PostgreSQL (documents, chunks, vectors, organizations)
```

## Features

- Tenant-scoped document and chunk access
- PDF, TXT and Markdown ingestion
- Configurable chunk size and overlap
- Pluggable deterministic / OpenAI-compatible embeddings
- Hybrid retrieval using cosine similarity and lexical scoring
- Reciprocal-rank fusion and lightweight reranking
- Answer generation through an OpenAI-compatible chat-completions API
- Citation metadata with document and chunk references
- Health endpoint, structured API schemas, Docker Compose and CI

## Quick start

Requirements: Docker + Docker Compose.

```bash
cp .env.example .env
docker compose up --build
```

API docs: http://localhost:8000/docs

For local development:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

Set `DATABASE_URL` to a PostgreSQL connection string. Docker Compose provisions PostgreSQL with pgvector. SQLite remains supported for lightweight local tests.

## API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/v1/organizations` | Create an organization and receive a development API token |
| POST | `/v1/documents` | Upload PDF, TXT or MD |
| GET | `/v1/documents` | List tenant documents |
| DELETE | `/v1/documents/{id}` | Delete a document and its chunks |
| POST | `/v1/query` | Retrieve context and generate a cited answer |

### Authentication

Create an organization using `POST /v1/organizations`. The response includes a bearer API token for local evaluation. Send it with every tenant-scoped request, together with `X-Organization-ID`.

This bootstrap flow is intentionally simple for a portfolio demo. Replace it with a proper identity provider, user membership model, token rotation and authorization policies before production use.

### Example query

```bash
curl -X POST http://localhost:8000/v1/query \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "X-Organization-ID: YOUR_ORG_ID" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the refund policy?","top_k":5}'
```

## Configuration

See `.env.example`. Set `EMBEDDING_PROVIDER=openai` and `OPENAI_API_KEY` to use an OpenAI-compatible embedding endpoint. Set `LLM_BASE_URL`, `LLM_API_KEY` and `LLM_MODEL` to enable generated answers. Without an LLM configuration, the API returns retrieved context with an explicit extractive fallback.

## Retrieval pipeline

1. Extract text from the uploaded file.
2. Split text into overlapping chunks.
3. Generate one embedding per chunk.
4. Retrieve semantic and lexical candidate rankings.
5. Fuse rankings using reciprocal-rank fusion.
6. Rerank candidates using a combined relevance score.
7. Build a context-limited prompt and return citations.

## Evaluation and limitations

The repository includes deterministic tests and a JSONL evaluation runner reporting hit-rate@k, MRR and p50/p95 latency. The sample dataset is illustrative; replace it with reviewed queries and expected sources. Retrieval metrics do not establish faithfulness or answer correctness.

Remaining production work: real user identity and RBAC, rate limiting, secrets management, audit logging, asynchronous ingestion, malware scanning, deletion guarantees, observability, load testing and human-reviewed faithfulness evaluation. Retrieved documents are untrusted input and must never be treated as system instructions. The organization-token bootstrap endpoint is a demo onboarding flow, not a complete identity system.

## License

MIT


## Retrieval evaluation

A reproducible evaluation harness is available in [docs/evaluation.md](docs/evaluation.md). It runs a JSONL golden dataset against the live API and reports hit-rate@k, MRR, and p50/p95 latency. Configure RAG_API_URL, RAG_API_TOKEN and RAG_ORGANIZATION_ID, then run python scripts/evaluate.py. Replace the illustrative sample questions with domain-reviewed cases before using the metrics for decisions.
