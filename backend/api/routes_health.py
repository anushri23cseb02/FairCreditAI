"""
Health check endpoints.

GET /health         -> simple liveness probe (always fast, no DB call)
GET /health/detailed -> also checks MySQL connectivity
"""
from fastapi import APIRouter

from backend.core.config import get_settings
from backend.database.connection import check_database_connection

router = APIRouter(tags=["health"])
settings = get_settings()


@router.get("/health")
def health() -> dict:
    """Basic liveness check. Matches the exact contract required by the spec."""
    return {"status": "healthy"}


@router.get("/health/detailed")
def health_detailed() -> dict:
    """Extended health check including a real MySQL connectivity probe."""
    db_ok = check_database_connection()
    return {
        "status": "healthy" if db_ok else "degraded",
        "app_name": settings.app_name,
        "app_env": settings.app_env,
        "database": "connected" if db_ok else "unreachable",
    }
