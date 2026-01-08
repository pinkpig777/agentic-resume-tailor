from pathlib import Path

from agentic_resume_tailor.core import loop_controller_v3
from agentic_resume_tailor.settings import get_settings


class DummyCandidate:
    def __init__(self, bullet_id: str, text: str) -> None:
        self.bullet_id = bullet_id
        self.text_latex = text
        self.meta = {"section": "experience"}
        self.selection_score = 1.0
        self.total_weighted = 1.0
        self.effective_total_weighted = 1.0


def test_v3_report_includes_rewrites(tmp_path, monkeypatch) -> None:
    candidates = [
        DummyCandidate("exp:acme__engineer:b01", "Built APIs in Python."),
        DummyCandidate("exp:acme__engineer:b02", "Improved latency by 20%."),
    ]

    def fake_retrieve(*_args, **_kwargs):
        return candidates

    def fake_render(_settings, _context, run_id):
        pdf_path = tmp_path / f"{run_id}.pdf"
        tex_path = tmp_path / f"{run_id}.tex"
        pdf_path.write_bytes(b"%PDF-1.4\\n")
        tex_path.write_text("% tex", encoding="utf-8")
        return str(pdf_path), str(tex_path)

    def fake_trim(_settings, _run_id, _static, selected_ids, _cands, rewrites, pdf_path):
        tex_path = Path(pdf_path).with_suffix(".tex")
        return pdf_path, str(tex_path), selected_ids, rewrites

    monkeypatch.setattr(loop_controller_v3, "multi_query_retrieve", fake_retrieve)
    monkeypatch.setattr(loop_controller_v3, "_render_pdf", fake_render)
    monkeypatch.setattr(loop_controller_v3, "_trim_to_single_page", fake_trim)

    base = Path(__file__).resolve().parents[2]
    settings = get_settings().model_copy(
        update={
            "use_jd_parser": False,
            "enable_bullet_rewrite": True,
            "max_iters": 1,
            "output_dir": str(tmp_path),
            "template_dir": str(base / "templates"),
            "canon_config": str(base / "config" / "canonicalization.json"),
            "family_config": str(base / "config" / "families.json"),
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
                "bullets": [
                    {"id": "b01", "text_latex": "Built APIs in Python."},
                    {"id": "b02", "text_latex": "Improved latency by 20%."},
                ],
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

    report_path = Path(artifacts.report_path)
    assert report_path.exists()
    payload = report_path.read_text(encoding="utf-8")
    assert '"rewritten_bullets"' in payload
