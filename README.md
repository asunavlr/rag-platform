# RAG Platform

A multi-tenant Retrieval-Augmented Generation (RAG) API built with FastAPI. It ingests documents, chunks and embeds their content, performs hybrid retrieval (semantic + lexical), reranks candidates, and generates grounded answers with source citations.

> **Project status:** portfolio MVP. The default embedding provider is deterministic and dependency-free for local development; configure an OpenAI-compatible API for production-quality embeddings and generation. Vectors are stored as JSON and ranked in the application for portability. For high-volume workloads, move vector operations to PostgreSQL + pgvector.

## Architecture

```text
Client
  |  Bearer JWT + X-Organization-ID
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

Set `DATABASE_URL` to a PostgreSQL connection string. SQLite is supported for tests and lightweight local runs.

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

Create an organization using `POST /v1/organizations`. The response includes a bearer token for local evaluation. Send it with every tenant-scoped request, together with `X-Organization-ID`.

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

The repository includes deterministic unit tests for chunking, similarity and ranking. Add a curated question/answer dataset and measure Recall@K, MRR, faithfulness, answer relevance, latency and token cost before deploying.

Security and production work still required: real user identity and RBAC, upload size and malware scanning, rate limiting, secrets management, audit logging, background ingestion workers, deletion guarantees, prompt-injection defenses, pgvector/ANN indexing, observability and load testing. Retrieved documents are untrusted input and must never be treated as system instructions.

## License

MIT
