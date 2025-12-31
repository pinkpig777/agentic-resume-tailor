from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "jd_fixture.json"
EXPECTED_PATH = REPO_ROOT / "tests" / "fixtures" / "expected_output.json"
SRC_PATH = REPO_ROOT / "src"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_report(report: dict) -> dict:
    normalized = json.loads(json.dumps(report))
    normalized["run_id"] = "RUN_ID"
    normalized["created_at"] = "CREATED_AT"
    artifacts = normalized.get("artifacts")
    if isinstance(artifacts, dict):
        artifacts["pdf"] = "RUN_ID.pdf"
        artifacts["tex"] = "RUN_ID.tex"
    _mask_variable_fields(normalized)
    return normalized


def _mask_variable_fields(value: object) -> None:
    if isinstance(value, dict):
        for key, item in list(value.items()):
            if key in {"queries_used"}:
                if isinstance(item, list):
                    has_text = any(isinstance(x, str) and x.strip() for x in item)
                    value[key] = ["<QUERY>"] if has_text else []
                continue
            if key in {"selected_ids"}:
                if isinstance(item, list):
                    value[key] = ["<ID>"] * len(item)
                continue
            if key in {"analysis", "jd_text", "jd_text_preview", "summary"}:
                if isinstance(item, str) and item.strip():
                    value[key] = "<REDACTED>"
                continue
            _mask_variable_fields(item)
    elif isinstance(value, list):
        for item in value:
            _mask_variable_fields(item)


def _ensure_env(tmp_dir: str) -> None:
    settings_path = Path(tmp_dir) / "user_settings.json"
    settings_payload = {
        "db_path": str(REPO_ROOT / "data" / "processed" / "chroma_db"),
        "sql_db_url": f"sqlite:///{tmp_dir}/resume.db",
        "export_file": str(Path(tmp_dir) / "my_experience.json"),
        "template_dir": str(REPO_ROOT / "templates"),
        "output_dir": tmp_dir,
        "use_jd_parser": False,
        "skip_pdf": True,
        "run_id": "RUN_ID",
        "api_url": "http://localhost:8000",
    }
    settings_path.write_text(
        json.dumps(settings_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.environ["USER_SETTINGS_FILE"] = str(settings_path)


def _seed_db_from_json() -> None:
    from agentic_resume_tailor.db.models import (
        Education,
        EducationBullet,
        Experience,
        ExperienceBullet,
        PersonalInfo,
        Project,
        ProjectBullet,
        Skills,
    )
    from agentic_resume_tailor.db.session import SessionLocal, init_db

    data = _load_json(REPO_ROOT / "data" / "my_experience.json")
    init_db()

    with SessionLocal() as db:
        personal = data.get("personal_info", {}) or {}
        db.add(
            PersonalInfo(
                name=str(personal.get("name", "") or ""),
                phone=str(personal.get("phone", "") or ""),
                email=str(personal.get("email", "") or ""),
                linkedin_id=str(personal.get("linkedin_id", "") or ""),
                github_id=str(personal.get("github_id", "") or ""),
                linkedin=str(personal.get("linkedin", "") or ""),
                github=str(personal.get("github", "") or ""),
            )
        )

        skills = data.get("skills", {}) or {}
        db.add(
            Skills(
                languages_frameworks=str(skills.get("languages_frameworks", "") or ""),
                ai_ml=str(skills.get("ai_ml", "") or ""),
                db_tools=str(skills.get("db_tools", "") or ""),
            )
        )

        for idx, edu in enumerate(data.get("education", []) or [], start=1):
            if not isinstance(edu, dict):
                continue
            edu_row = Education(
                school=str(edu.get("school", "") or ""),
                dates=str(edu.get("dates", "") or ""),
                degree=str(edu.get("degree", "") or ""),
                location=str(edu.get("location", "") or ""),
                sort_order=idx,
            )
            db.add(edu_row)
            db.flush()
            for b_idx, bullet in enumerate(edu.get("bullets", []) or [], start=1):
                if not isinstance(bullet, str):
                    continue
                db.add(
                    EducationBullet(
                        education_id=edu_row.id,
                        text_latex=bullet,
                        sort_order=b_idx,
                    )
                )

        for idx, exp in enumerate(data.get("experiences", []) or [], start=1):
            if not isinstance(exp, dict):
                continue
            exp_row = Experience(
                job_id=str(exp.get("job_id") or ""),
                company=str(exp.get("company", "") or ""),
                role=str(exp.get("role", "") or ""),
                dates=str(exp.get("dates", "") or ""),
                location=str(exp.get("location", "") or ""),
                sort_order=idx,
            )
            db.add(exp_row)
            db.flush()
            for b_idx, bullet in enumerate(exp.get("bullets", []) or [], start=1):
                if not isinstance(bullet, dict):
                    continue
                db.add(
                    ExperienceBullet(
                        experience_id=exp_row.id,
                        local_id=str(bullet.get("id") or ""),
                        text_latex=str(bullet.get("text_latex") or ""),
                        sort_order=b_idx,
                    )
                )

        for idx, proj in enumerate(data.get("projects", []) or [], start=1):
            if not isinstance(proj, dict):
                continue
            proj_row = Project(
                project_id=str(proj.get("project_id") or ""),
                name=str(proj.get("name", "") or ""),
                technologies=str(proj.get("technologies", "") or ""),
                sort_order=idx,
            )
            db.add(proj_row)
            db.flush()
            for b_idx, bullet in enumerate(proj.get("bullets", []) or [], start=1):
                if not isinstance(bullet, dict):
                    continue
                db.add(
                    ProjectBullet(
                        project_id=proj_row.id,
                        local_id=str(bullet.get("id") or ""),
                        text_latex=str(bullet.get("text_latex") or ""),
                        sort_order=b_idx,
                    )
                )

        db.commit()


def _run_generate(payload: dict, tmp_dir: str) -> dict:
    _ensure_env(tmp_dir)
    sys.path.insert(0, str(SRC_PATH))

    _seed_db_from_json()

    from agentic_resume_tailor.api import server  # noqa: E402

    client = TestClient(server.app)
    resp = client.post("/generate", json=payload)
    if resp.status_code != 200:
        raise SystemExit(f"Generate failed: HTTP {resp.status_code} {resp.text}")

    report_path = Path(tmp_dir) / "RUN_ID_report.json"
    if not report_path.exists():
        raise SystemExit(f"Expected report not found at {report_path}")

    return _load_json(report_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update expected_output.json with the latest normalized output",
    )
    args = parser.parse_args()

    payload = _load_json(FIXTURE_PATH)

    with tempfile.TemporaryDirectory() as tmp_dir:
        report = _run_generate(payload, tmp_dir)

    normalized = _normalize_report(report)

    if args.update or not EXPECTED_PATH.exists():
        EXPECTED_PATH.write_text(
            json.dumps(normalized, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Updated expected output at {EXPECTED_PATH}")
        return

    expected = _load_json(EXPECTED_PATH)
    if normalized != expected:
        raise SystemExit("Characterization output mismatch. Run with --update to refresh.")

    print("Characterization output matched expected output.")


if __name__ == "__main__":
    main()
