"""Deprecated v3 compatibility wrappers.

Use agentic_resume_tailor.core.loop_controller instead.
"""

from __future__ import annotations

from typing import Any, Dict

from agentic_resume_tailor.core.loop_controller import RunArtifacts, run_loop


class RunArtifactsV3(RunArtifacts):
    """Deprecated. Use RunArtifacts."""


def run_loop_v3(
    jd_text: str,
    *,
    collection: Any,
    embedding_fn: Any,
    static_export: Dict[str, Any],
    settings: Any,
) -> RunArtifacts:
    """Deprecated. Use run_loop."""
    return run_loop(
        jd_text,
        collection=collection,
        embedding_fn=embedding_fn,
        static_export=static_export,
        settings=settings,
    )
