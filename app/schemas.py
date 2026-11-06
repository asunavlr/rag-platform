from pydantic import BaseModel, Field


class OrganizationCreate(BaseModel):
    name: str = Field(min_length=2, max_length=160)


class OrganizationCreated(BaseModel):
    organization_id: str
    name: str
    access_token: str


class DocumentOut(BaseModel):
    id: str
    filename: str
    status: str
    chunk_count: int
    created_at: str


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=20)


class Citation(BaseModel):
    document_id: str
    filename: str
    chunk_id: str
    chunk_index: int
    score: float
    excerpt: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    retrieval_count: int
    generation: str
