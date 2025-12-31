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
    os.environ["ART_USE_JD_PARSER"] = "0"
    os.environ["ART_RUN_ID"] = "RUN_ID"
    os.environ["ART_SKIP_PDF"] = "1"
    os.environ["ART_OUTPUT_DIR"] = tmp_dir
    os.environ["ART_DB_PATH"] = str(REPO_ROOT / "data" / "processed" / "chroma_db")
    os.environ["ART_DATA_FILE"] = str(REPO_ROOT / "data" / "my_experience.json")
    os.environ["ART_SEED_FROM_JSON"] = "1"
    os.environ["ART_SQL_DB_URL"] = f"sqlite:///{tmp_dir}/resume.db"
    os.environ["ART_TEMPLATE_DIR"] = str(REPO_ROOT / "templates")


def _run_generate(payload: dict, tmp_dir: str) -> dict:
    _ensure_env(tmp_dir)
    sys.path.insert(0, str(SRC_PATH))

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
