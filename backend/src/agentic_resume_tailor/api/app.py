from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agentic_resume_tailor.api.routes.main import router as main_router
from agentic_resume_tailor.api.runtime import (
    build_runtime_state,
    load_runtime_settings,
    refresh_collection,
)
from agentic_resume_tailor.db.session import init_db
from agentic_resume_tailor.utils.logging import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("API Server starting: Initializing resume DB...")
    init_db()
    for key, value in build_runtime_state().items():
        setattr(app.state, key, value)
    if os.environ.get("ART_SKIP_STARTUP_LOAD"):
        logger.info("API Server startup load skipped.")
        yield
        return
    logger.info("API Server starting: Loading Chroma...")
    refresh_collection(app, load_runtime_settings())
    logger.info("API Server ready.")
    yield


def create_app() -> FastAPI:
    settings = load_runtime_settings()
    app = FastAPI(title="AI Resume Agent API", version="0.3", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"]
        if settings.cors_origins.strip() == "*"
        else [origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(main_router)
    return app
