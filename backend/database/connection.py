"""
SQLAlchemy engine and session management for MySQL.

Provides the shared engine/session used across the app, plus
init_db_schema() (creates the prediction_log table on startup, see
backend/database/models.py) and check_database_connection() (used by
/health/detailed).
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from backend.core.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

# pool_pre_ping avoids handing out dead connections after MySQL restarts.
engine = create_engine(
    settings.sqlalchemy_database_url,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a DB session and always closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Context manager for scripts (outside of FastAPI's DI system)."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def check_database_connection() -> bool:
    """
    Runs a trivial query against MySQL to confirm connectivity.
    Used by the /health endpoint. Never raises -- returns False on failure
    so the health endpoint can report status instead of crashing.
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:  # noqa: BLE001 - we deliberately want to catch all
        logger.warning("Database connectivity check failed: %s", exc)
        return False


def init_db_schema() -> bool:
    """
    Creates any missing tables (currently just prediction_log). Called
    once at API startup. Never raises -- the API must still come up and
    serve predictions even if MySQL is not reachable yet or is still
    starting (Compose depends_on/healthcheck ordering aside, this is a
    second line of defence against startup race conditions).
    """
    from backend.database.models import Base

    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database schema verified/created.")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not create database schema (will retry on next request path): %s", exc)
        return False
