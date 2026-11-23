from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from alembic import command
from alembic.config import Config

from app.config import settings

connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def initialize_database() -> None:
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", settings.database_url.replace("%", "%%"))
    command.upgrade(config, "head")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
