from pathlib import Path

from fastapi.testclient import TestClient

from agentic_resume_tailor.core.agents.scoring_agent import ScoreResult
from agentic_resume_tailor.core.loop_controller import RunArtifacts


def _make_artifacts(tmp_path: Path, run_id: str) -> RunArtifacts:
    pdf_path = tmp_path / f"{run_id}.pdf"
    tex_path = tmp_path / f"{run_id}.tex"
    report_path = tmp_path / f"{run_id}_report.json"

    pdf_path.write_bytes(b"%PDF-1.4\n")
    tex_path.write_text("% tex", encoding="utf-8")
    report_path.write_text("{}", encoding="utf-8")

    score = ScoreResult(
        final_score=90,
        retrieval_score=1.0,
        coverage_bullets_only=1.0,
        coverage_all=1.0,
        length_score=1.0,
        redundancy_penalty=0.0,
        quality_score=0.0,
        must_missing_bullets_only=[],
        nice_missing_bullets_only=[],
        must_missing_all=[],
        nice_missing_all=[],
        length_by_bullet={},
        redundancy_pairs=[],
        boost_terms=[],
        agent_used=False,
        agent_fallback=False,
        agent_model=None,
    )

    return RunArtifacts(
        run_id=run_id,
        selected_ids=[],
        rewritten_bullets={},
        best_score=score,
        iteration_trace=[],
        pdf_path=str(pdf_path),
        tex_path=str(tex_path),
        report_path=str(report_path),
        best_iteration_index=0,
        profile_used=False,
    )


def test_generate_returns_urls_and_writes_artifacts(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ART_SKIP_STARTUP_LOAD", "1")
    monkeypatch.setenv("ART_SKIP_CHROMA_LOAD", "1")

    from agentic_resume_tailor.api import server
    from agentic_resume_tailor.api.routes import main as routes_main

    run_id = "test_run"
    artifacts = _make_artifacts(tmp_path, run_id)

    def fake_run_loop(*_args, **_kwargs):
        return artifacts

    monkeypatch.setattr(routes_main, "run_loop", fake_run_loop)

    client = TestClient(server.app)
    server.app.state.collection = object()
    server.app.state.embedding_fn = object()
    response = client.post("/generate", json={"jd_text": "test jd"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == run_id
    assert payload["pdf_url"] == f"/runs/{run_id}/pdf"
    assert payload["tex_url"] == f"/runs/{run_id}/tex"
    assert payload["report_url"] == f"/runs/{run_id}/report"
    assert Path(artifacts.pdf_path).exists()
    assert Path(artifacts.tex_path).exists()
    assert Path(artifacts.report_path).exists()


def test_generate_uses_live_settings_and_request_rewrite_style(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ART_SKIP_STARTUP_LOAD", "1")
    monkeypatch.setenv("ART_SKIP_CHROMA_LOAD", "1")
    monkeypatch.setenv("USER_SETTINGS_FILE", str(tmp_path / "user_settings.test.json"))

    from agentic_resume_tailor.api import server
    from agentic_resume_tailor.api.routes import main as routes_main

    captured_styles: list[str] = []
    artifacts = _make_artifacts(tmp_path, "style_run")

    def fake_run_loop(_jd_text, *_args, **kwargs):
        captured_styles.append(kwargs["settings"].rewrite_style)
        return artifacts

    async def fake_run_in_threadpool(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(routes_main, "run_loop", fake_run_loop)
    monkeypatch.setattr(routes_main, "run_in_threadpool", fake_run_in_threadpool)

    client = TestClient(server.app)
    server.app.state.collection = object()
    server.app.state.embedding_fn = object()

    settings_resp = client.get("/settings")
    assert settings_resp.status_code == 200
    settings_payload = settings_resp.json()
    assert "live_fields" in settings_payload
    assert "restart_required_fields" in settings_payload
    assert "rewrite_style" in settings_payload["live_fields"]
    assert "db_path" in settings_payload["restart_required_fields"]

    updated_resp = client.put("/settings", json={"rewrite_style": "creative"})
    assert updated_resp.status_code == 200
    assert updated_resp.json()["rewrite_style"] == "creative"

    response_default = client.post("/generate", json={"jd_text": "test jd"})
    assert response_default.status_code == 200

    response_override = client.post(
        "/generate",
        json={"jd_text": "test jd", "rewrite_style": "conservative"},
    )
    assert response_override.status_code == 200
    assert captured_styles == ["creative", "conservative"]
