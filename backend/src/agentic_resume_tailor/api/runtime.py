from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from queue import Queue
from typing import Any, Dict

import chromadb
from chromadb.errors import NotFoundError
from fastapi import HTTPException

from agentic_resume_tailor.settings import (
    get_settings,
    live_settings_fields,
    load_settings,
    restart_required_fields,
)
from agentic_resume_tailor.user_config import (
    get_user_config_path,
    load_user_config,
    save_user_config,
)
from agentic_resume_tailor.utils.embeddings import build_sentence_transformer_ef

logger = logging.getLogger(__name__)

PROGRESS_TTL_S = 1800


@dataclass
class RunProgress:
    queue: Queue[Dict[str, Any]] = field(default_factory=Queue)
    state: Dict[str, Any] = field(default_factory=dict)


def build_runtime_state() -> Dict[str, Any]:
    return {
        "ingest_lock": threading.Lock(),
        "run_progress": {},
        "run_progress_lock": threading.Lock(),
        "collection": None,
        "embedding_fn": None,
    }


def ensure_runtime_state(app: Any) -> None:
    for key, value in build_runtime_state().items():
        if not hasattr(app.state, key):
            setattr(app.state, key, value)


def schedule_progress_cleanup(app: Any, run_id: str) -> None:
    ensure_runtime_state(app)

    def _cleanup() -> None:
        with app.state.run_progress_lock:
            app.state.run_progress.pop(run_id, None)

    timer = threading.Timer(PROGRESS_TTL_S, _cleanup)
    timer.daemon = True
    timer.start()


def get_or_create_progress(app: Any, run_id: str, max_iters: int | None = None) -> RunProgress:
    ensure_runtime_state(app)
    with app.state.run_progress_lock:
        progress = app.state.run_progress.get(run_id)
        if progress is None:
            progress = RunProgress(
                state={
                    "run_id": run_id,
                    "status": "pending",
                    "stage": None,
                    "iteration": None,
                    "max_iters": max_iters,
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
            app.state.run_progress[run_id] = progress
        elif max_iters is not None:
            progress.state.setdefault("max_iters", max_iters)
    return progress


def emit_progress(app: Any, run_id: str, payload: Dict[str, Any]) -> None:
    ensure_runtime_state(app)
    progress = get_or_create_progress(app, run_id)
    event = {"run_id": run_id, **payload}
    if "status" not in event:
        stage = event.get("stage")
        event["status"] = "complete" if stage == "done" else "running"
    event["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    progress.state.update(event)
    progress.queue.put(event)
    if event.get("status") in ("complete", "error"):
        schedule_progress_cleanup(app, run_id)


def format_sse(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def load_runtime_settings() -> Any:
    get_settings.cache_clear()
    return load_settings()


def settings_payload() -> Dict[str, Any]:
    settings = load_runtime_settings()
    payload = settings.model_dump()
    payload.pop("openai_api_key", None)
    payload["config_path"] = get_user_config_path()
    payload["live_fields"] = live_settings_fields()
    payload["restart_required_fields"] = restart_required_fields()
    return payload


def update_settings_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    settings = load_runtime_settings()
    allowed = set(settings.model_fields.keys()) - {"openai_api_key"}
    updates = {key: value for key, value in (payload or {}).items() if key in allowed}
    if updates:
        current = load_user_config()
        save_user_config(None, {**current, **updates})
        get_settings.cache_clear()
    return settings_payload()


def load_collection(settings: Any | None = None):
    settings = settings or load_runtime_settings()
    if os.environ.get("ART_SKIP_CHROMA_LOAD"):
        logger.warning("ART_SKIP_CHROMA_LOAD set; skipping Chroma load.")
        return None, None
    client = chromadb.PersistentClient(path=settings.db_path)
    embedding_fn = build_sentence_transformer_ef(settings.embed_model, disable_progress=True)
    try:
        collection = client.get_collection(
            name=settings.collection_name, embedding_function=embedding_fn
        )
    except NotFoundError:
        logger.warning(
            "Chroma collection '%s' missing; creating empty collection.", settings.collection_name
        )
        collection = client.create_collection(
            name=settings.collection_name,
            embedding_function=embedding_fn,
        )
    logger.info(
        "Loaded Chroma collection '%s' (%s records)",
        settings.collection_name,
        collection.count(),
    )
    return collection, embedding_fn


def refresh_collection(app: Any, settings: Any | None = None) -> None:
    ensure_runtime_state(app)
    app.state.collection, app.state.embedding_fn = load_collection(settings)


def require_collection(app: Any):
    ensure_runtime_state(app)
    if app.state.collection is None or app.state.embedding_fn is None:
        raise HTTPException(
            status_code=503,
            detail="Chroma collection is not loaded; run /admin/ingest first.",
        )
    return app.state.collection, app.state.embedding_fn
