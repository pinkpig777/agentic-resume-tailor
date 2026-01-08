from pathlib import Path

from agentic_resume_tailor.core import loop_controller_v3
from agentic_resume_tailor.core.agents.query_agent import QueryPlan, QueryPlanItem
from agentic_resume_tailor.core.agents.scoring_agent import ScoreResultV3
from agentic_resume_tailor.settings import get_settings


class DummyCandidate:
    def __init__(self, bullet_id: str, text: str) -> None:
        self.bullet_id = bullet_id
        self.text_latex = text
        self.meta = {"section": "experience"}
        self.selection_score = 1.0
        self.total_weighted = 1.0
        self.effective_total_weighted = 1.0


def _score(final_score: int) -> ScoreResultV3:
    return ScoreResultV3(
        final_score=final_score,
        retrieval_score=0.5,
        coverage_bullets_only=0.5,
        coverage_all=0.5,
        length_score=1.0,
        redundancy_penalty=0.0,
        quality_score=0.5,
        must_missing_bullets_only=[],
        nice_missing_bullets_only=[],
        must_missing_all=[],
        nice_missing_all=[],
        length_by_bullet={"exp:acme__engineer:b01": 120},
        redundancy_pairs=[],
        boost_terms=[],
        agent_used=False,
        agent_fallback=False,
        agent_model=None,
    )


def test_v3_loop_selects_best_iteration(tmp_path, monkeypatch) -> None:
    candidates = [DummyCandidate("exp:acme__engineer:b01", "Built APIs in Python.")]

    def fake_retrieve(*_args, **_kwargs):
        return candidates

    def fake_select_topk(_cands, max_bullets):
        return [candidates[0].bullet_id][:max_bullets], []

    def fake_render(_settings, _context, run_id):
        pdf_path = tmp_path / f"{run_id}.pdf"
        tex_path = tmp_path / f"{run_id}.tex"
        pdf_path.write_bytes(b"%PDF-1.4\\n")
        tex_path.write_text("% tex", encoding="utf-8")
        return str(pdf_path), str(tex_path)

    def fake_trim(_settings, _run_id, _static, selected_ids, _cands, rewrites, pdf_path):
        tex_path = Path(pdf_path).with_suffix(".tex")
        return pdf_path, str(tex_path), selected_ids, rewrites

    scores = [_score(70), _score(85), _score(80)]

    def fake_score(*_args, **_kwargs):
        return scores.pop(0)

    def fake_build_query_plan(_jd_text, _settings):
        return QueryPlan(
            items=[QueryPlanItem(text="backend api")],
            profile={"must_have": [], "nice_to_have": []},
            profile_used=True,
            profile_summary=None,
            agent_used=False,
            agent_fallback=False,
            agent_model=None,
        )

    monkeypatch.setattr(loop_controller_v3, "build_query_plan", fake_build_query_plan)
    monkeypatch.setattr(loop_controller_v3, "multi_query_retrieve", fake_retrieve)
    monkeypatch.setattr(loop_controller_v3, "select_topk", fake_select_topk)
    monkeypatch.setattr(loop_controller_v3, "_render_pdf", fake_render)
    monkeypatch.setattr(loop_controller_v3, "_trim_to_single_page", fake_trim)
    monkeypatch.setattr(loop_controller_v3, "score_resume", fake_score)

    base = Path(__file__).resolve().parents[2]
    settings = get_settings().model_copy(
        update={
            "use_jd_parser": False,
            "enable_bullet_rewrite": False,
            "max_iters": 3,
            "threshold": 99,
            "output_dir": str(tmp_path),
            "template_dir": str(base / "templates"),
            "skip_pdf": True,
        }
    )

    static_export = {
        "skills": {},
        "experiences": [
            {
                "job_id": "acme__engineer",
                "company": "Acme",
                "role": "Engineer",
                "dates": "",
                "location": "",
                "bullets": [{"id": "b01", "text_latex": "Built APIs in Python."}],
            }
        ],
        "projects": [],
    }

    artifacts = loop_controller_v3.run_loop_v3(
        jd_text="Sample JD",
        collection=None,
        embedding_fn=None,
        static_export=static_export,
        settings=settings,
    )

    assert artifacts.best_iteration_index == 1
    assert artifacts.best_score.final_score == 85
