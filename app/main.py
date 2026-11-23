import io
from contextlib import asynccontextmanager
from pathlib import PurePath

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db, initialize_database
from app.models import Chunk, Document, Organization
from app.schemas import (
    Citation,
    DocumentOut,
    OrganizationCreate,
    OrganizationCreated,
    QueryRequest,
    QueryResponse,
)
from app.security import create_access_token, get_current_organization, hash_token
from app.services.chunking import split_text
from app.services.embeddings import embed_texts
from app.services.generation import generate_answer
from app.services.retrieval import hybrid_rank, retrieve_candidates


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_database()
    yield


app = FastAPI(
    title=settings.app_name,
    version="0.2.0",
    description="Multi-tenant Retrieval-Augmented Generation API with hybrid retrieval.",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "service": "rag-platform"}


@app.get("/ready")
def readiness(db: Session = Depends(get_db)):
    try:
        db.execute(select(1))
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Database is unavailable") from exc
    return {"status": "ready"}


@app.post("/v1/organizations", response_model=OrganizationCreated, status_code=201)
def create_organization(payload: OrganizationCreate, db: Session = Depends(get_db)):
    token = create_access_token()
    organization = Organization(name=payload.name.strip(), token_hash=hash_token(token))
    db.add(organization)
    db.commit()
    db.refresh(organization)
    return OrganizationCreated(
        organization_id=organization.id, name=organization.name, access_token=token
    )


def extract_text(filename: str, content: bytes) -> str:
    suffix = PurePath(filename).suffix.lower()
    if suffix in {".txt", ".md", ".markdown"}:
        try:
            return content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise HTTPException(status_code=422, detail="Text files must use UTF-8 encoding") from exc
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(content))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:
            raise HTTPException(status_code=422, detail="Could not extract text from this PDF") from exc
    raise HTTPException(status_code=415, detail="Supported formats: PDF, TXT, MD")


@app.post("/v1/documents", response_model=DocumentOut, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    organization: Organization = Depends(get_current_organization),
    db: Session = Depends(get_db),
):
    content = await file.read(settings.max_upload_bytes + 1)
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="File exceeds upload limit")
    filename = PurePath((file.filename or "document").replace("\\", "/")).name
    if not filename or filename in {".", ".."} or len(filename) > 255:
        raise HTTPException(status_code=422, detail="A valid filename is required")
    text = extract_text(filename, content)
    chunks_text = split_text(text, settings.chunk_size, settings.chunk_overlap)
    if not chunks_text:
        raise HTTPException(status_code=422, detail="Document contains no extractable text")
    try:
        vectors = await embed_texts(chunks_text)
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Embedding provider request failed") from exc
    if len(vectors) != len(chunks_text) or any(
        len(vector) != settings.embedding_dimensions for vector in vectors
    ):
        raise HTTPException(status_code=502, detail="Embedding provider returned an unexpected vector shape")
    try:
        document = Document(
            organization_id=organization.id,
            filename=filename,
            content_type=file.content_type or "application/octet-stream",
            chunk_count=len(chunks_text),
        )
        db.add(document)
        db.flush()
        for index, (chunk_text, vector) in enumerate(zip(chunks_text, vectors, strict=True)):
            db.add(Chunk(
                organization_id=organization.id,
                document_id=document.id,
                ordinal=index,
                content=chunk_text,
                embedding=vector,
            ))
        db.commit()
        db.refresh(document)
    except Exception:
        db.rollback()
        raise
    return DocumentOut(
        id=document.id, filename=document.filename, status=document.status,
        chunk_count=document.chunk_count, created_at=document.created_at.isoformat(),
    )


@app.get("/v1/documents", response_model=list[DocumentOut])
def list_documents(
    organization: Organization = Depends(get_current_organization),
    db: Session = Depends(get_db),
):
    documents = db.scalars(
        select(Document).where(Document.organization_id == organization.id)
        .order_by(Document.created_at.desc())
    ).all()
    return [
        DocumentOut(
            id=item.id, filename=item.filename, status=item.status,
            chunk_count=item.chunk_count, created_at=item.created_at.isoformat(),
        ) for item in documents
    ]


@app.delete("/v1/documents/{document_id}", status_code=204)
def delete_document(
    document_id: str,
    organization: Organization = Depends(get_current_organization),
    db: Session = Depends(get_db),
):
    document = db.scalar(
        select(Document).where(
            Document.id == document_id, Document.organization_id == organization.id
        )
    )
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    db.delete(document)
    db.commit()


@app.post("/v1/query", response_model=QueryResponse)
async def query_knowledge_base(
    payload: QueryRequest,
    organization: Organization = Depends(get_current_organization),
    db: Session = Depends(get_db),
):
    try:
        query_vector = (await embed_texts([payload.question]))[0]
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Embedding provider request failed") from exc
    candidates = retrieve_candidates(
        db, organization.id, query_vector, candidate_limit=min(payload.top_k * 10, 200)
    )
    if not candidates:
        return QueryResponse(
            answer="No documents have been indexed for this organization.",
            citations=[], retrieval_count=0, generation="none",
        )
    document_ids = {chunk.document_id for chunk in candidates}
    documents = {
        item.id: item for item in db.scalars(
            select(Document).where(
                Document.organization_id == organization.id, Document.id.in_(document_ids)
            )
        ).all()
    }
    ranked = hybrid_rank(payload.question, candidates, documents, payload.top_k, query_vector)
    try:
        answer, generation = await generate_answer(payload.question, ranked)
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Language model provider request failed") from exc
    citations = [
        Citation(
            document_id=item["document"].id,
            filename=item["document"].filename,
            chunk_id=item["chunk"].id,
            chunk_index=item["chunk"].ordinal,
            score=round(item["score"], 6),
            excerpt=item["chunk"].content[:400],
        )
        for item in ranked
    ]
    return QueryResponse(
        answer=answer, citations=citations, retrieval_count=len(ranked), generation=generation
    )
