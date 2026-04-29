from pathlib import Path

from fastapi.testclient import TestClient

from agentic_resume_tailor.core.agents.scoring_agent import ScoreResult
from agentic_resume_tailor.core.loop_controller import RunArtifacts


def test_generate_returns_urls_and_writes_artifacts(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ART_SKIP_STARTUP_LOAD", "1")
    monkeypatch.setenv("ART_SKIP_CHROMA_LOAD", "1")

    from agentic_resume_tailor.api import server

    run_id = "test_run"
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

    artifacts = RunArtifacts(
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

    def fake_run_loop(*_args, **_kwargs):
        return artifacts

    server.COLLECTION = object()
    server.EMB_FN = object()
    monkeypatch.setattr(server, "run_loop", fake_run_loop)

    client = TestClient(server.app)
    response = client.post("/generate", json={"jd_text": "test jd"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == run_id
    assert payload["pdf_url"] == f"/runs/{run_id}/pdf"
    assert payload["tex_url"] == f"/runs/{run_id}/tex"
    assert payload["report_url"] == f"/runs/{run_id}/report"
    assert Path(pdf_path).exists()
    assert Path(tex_path).exists()
    assert Path(report_path).exists()
