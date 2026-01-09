from __future__ import annotations

from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from agentic_resume_tailor.db.base import Base
from agentic_resume_tailor.settings import get_settings


def _make_engine():
    """Create the SQLAlchemy engine with SQLite-safe settings."""
    settings = get_settings()
    url = settings.sql_db_url
    connect_args = {}
    if url.startswith("sqlite"):
        connect_args = {"check_same_thread": False}
        parsed = make_url(url)
        db_path = parsed.database
        if db_path and db_path != ":memory:":
            path = Path(db_path)
            if not path.is_absolute():
                path = Path.cwd() / path
            path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(url, future=True, connect_args=connect_args)


engine = _make_engine()
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False)


def init_db() -> None:
    """Create DB tables if they do not exist."""
    from agentic_resume_tailor.db import models  # noqa: F401

    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Yield a DB session for FastAPI dependency injection.

    Returns:
        Generator of results.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
