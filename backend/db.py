"""
PostgreSQL (pgvector) connection using psycopg2 with connection pooling.
Works the same against a local PostgreSQL or AlloyDB.

Uses psycopg2's ThreadedConnectionPool so that concurrent requests reuse
connections instead of opening/closing on every call — too many
simultaneous new TCP connections cause "Connection reset by peer".
"""

import logging
import psycopg2
from psycopg2 import pool, Error
from contextlib import contextmanager
from backend.config import (
    ALLOYDB_HOST,
    ALLOYDB_PORT,
    ALLOYDB_USER,
    ALLOYDB_PASSWORD,
    ALLOYDB_DATABASE,
    ALLOYDB_SSLMODE,
)

logger = logging.getLogger(__name__)

DB_CONFIG = {
    "host": ALLOYDB_HOST,
    "port": ALLOYDB_PORT,
    "database": ALLOYDB_DATABASE,
    "user": ALLOYDB_USER,
    "password": ALLOYDB_PASSWORD,
    "sslmode": ALLOYDB_SSLMODE,
    "connect_timeout": 10,
}

# ── Connection Pool ──────────────────────────────────────────────────────
# minconn=1  — keep at least 1 idle connection warm
# maxconn=10 — never open more than 10 connections
_pool: pool.ThreadedConnectionPool | None = None


def _get_pool() -> pool.ThreadedConnectionPool:
    """Lazy-initialize the thread-safe connection pool."""
    global _pool
    if _pool is None or _pool.closed:
        logger.info("Creating DB connection pool (min=1, max=10)")
        _pool = pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            **DB_CONFIG,
        )
    return _pool


def get_connection():
    """Get a connection from the pool (prefer get_db() context manager)."""
    return _get_pool().getconn()


def release_connection(conn):
    """Return a connection to the pool."""
    try:
        _get_pool().putconn(conn)
    except Exception:
        # Pool may have been closed; just close the connection directly
        try:
            conn.close()
        except Exception:
            pass


@contextmanager
def get_db():
    """
    Context manager for database connections.
    Gets a connection from the pool, auto-commits, and returns it to the pool.
    """
    conn = None
    try:
        conn = get_connection()
        yield conn
        conn.commit()
    except (Exception, Error) as e:
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass
        raise e
    finally:
        if conn:
            release_connection(conn)


def close_pool():
    """Close all connections in the pool. Call on app shutdown."""
    global _pool
    if _pool and not _pool.closed:
        logger.info("Closing DB connection pool")
        _pool.closeall()
        _pool = None


def test_connection():
    """Test that the database connection works."""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            cursor.close()
            return result[0] == 1
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False
