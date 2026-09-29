"""
Verifies the database connectivity check is safe to call with no live
MySQL server (as in CI or this sandbox) — it must return False, never
raise. When a real MySQL is reachable (e.g. inside Docker Compose or
against localhost:3307 as mapped by docker-compose.yml), it should
return True; that positive path is exercised manually as part of
Phase 17 (Final Validation) against the live Docker stack, not here.
"""
from backend.database.connection import check_database_connection


def test_check_database_connection_never_raises():
    result = check_database_connection()
    assert isinstance(result, bool)
