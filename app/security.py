import hashlib
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import Depends, Header, HTTPException
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models import Organization


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create_access_token() -> str:
    return secrets.token_urlsafe(36)


def get_current_organization(
    authorization: str | None = Header(default=None),
    x_organization_id: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> Organization:
    if not authorization or not authorization.startswith("Bearer ") or not x_organization_id:
        raise HTTPException(status_code=401, detail="Bearer token and X-Organization-ID are required")
    token = authorization.removeprefix("Bearer ").strip()
    organization = db.query(Organization).filter_by(id=x_organization_id, token_hash=hash_token(token)).first()
    if not organization:
        raise HTTPException(status_code=401, detail="Invalid credentials or organization")
    return organization
