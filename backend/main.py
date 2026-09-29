"""
FairCredit AI - FastAPI backend entrypoint.

Run locally with:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

Docs available at:
    http://localhost:8000/docs
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.routes_health import router as health_router
from backend.api.routes_dataset import router as dataset_router
from backend.api.routes_model import router as model_router
from backend.api.routes_prediction import router as prediction_router
from backend.api.routes_train import router as train_router
from backend.api.routes_credit_card import router as credit_card_router
from backend.core.config import get_settings
from backend.core.logging_config import configure_logging

settings = get_settings()
configure_logging(debug=settings.app_debug)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("%s starting up in '%s' mode", settings.app_name, settings.app_env)
    from backend.database.connection import init_db_schema
    init_db_schema()
    yield
    logger.info("%s shutting down", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    description=(
        "Explainable and Fair Credit Risk Prediction System. Serves "
        "prediction, retraining, explainability, and fairness-audit "
        "endpoints for a Logistic Regression model trained on real HMDA "
        "mortgage data, with prediction history stored in MySQL."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# --- CORS -------------------------------------------------------------
# Only the configured origins (the Streamlit frontend) are allowed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Global exception handling ----------------------------------------
# The application must never crash with an unhandled exception; instead
# it returns a structured JSON error and logs the real cause server-side.
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred. It has been logged.",
        },
    )


# --- Routers ------------------------------------------------------------
app.include_router(health_router)
app.include_router(prediction_router)
app.include_router(model_router)
app.include_router(dataset_router)
app.include_router(train_router)
app.include_router(credit_card_router)


@app.get("/", tags=["root"])
def root() -> dict:
    return {
        "service": settings.app_name,
        "status": "running",
        "docs": "/docs",
        "health": "/health",
    }
