"""
AssureX Database Connection and Session Management
==================================================
Provides SQLAlchemy engine, session maker, declarative base, and dependency generator.
"""

from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from backend.config import DATABASE_URL

# Configure engine with SQLite thread safety if using SQLite
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields an independent database session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
