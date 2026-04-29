from __future__ import annotations

import copy
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

import jinja2
from fastapi import HTTPException
from pydantic import BaseModel, Field
from pypdf import PdfReader

logger = logging.getLogger(__name__)


class TempAddition(BaseModel):
    parent_type: str
    parent_id: str
    text_latex: str
    temp_id: str | None = None


class TempOverrides(BaseModel):
    edits: Dict[str, str] = Field(default_factory=dict)
    removals: List[str] = Field(default_factory=list)
    additions: List[TempAddition] = Field(default_factory=list)


def normalize_output_pdf_name(name: str | None) -> str | None:
    if not name:
        return None
    filename = os.path.basename(str(name).strip())
    if not filename:
        return None
    if not filename.lower().endswith(".pdf"):
        filename = f"{filename}.pdf"
    return filename


def _output_pdf_alias_path(settings: Any) -> str | None:
    filename = normalize_output_pdf_name(getattr(settings, "output_pdf_name", None))
    if not filename:
        return None
    return os.path.join(settings.output_dir, filename)


def _write_output_pdf_alias(settings: Any, pdf_path: str) -> None:
    alias_path = _output_pdf_alias_path(settings)
    if not alias_path:
        return
    if os.path.abspath(alias_path) == os.path.abspath(pdf_path):
        return
    try:
        shutil.copyfile(pdf_path, alias_path)
    except Exception:
        logger.exception("Failed to write output PDF alias")


def dedupe_ids(ids: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for bid in ids:
        if not bid or bid in seen:
            continue
        seen.add(bid)
        out.append(bid)
    return out


def apply_temp_overrides(
    run_id: str,
    selected_ids: List[str],
    selected_candidates: List[Any],
    temp_overrides: TempOverrides | None,
    *,
    auto_include_additions: bool,
) -> Tuple[List[str], List[Any], Dict[str, Any]]:
    normalized: Dict[str, Any] = {"edits": {}, "removals": [], "additions": []}
    if temp_overrides is None:
        return dedupe_ids(selected_ids), selected_candidates, normalized

    selected_ids = dedupe_ids(selected_ids)
    selected_set = set(selected_ids)

    removals = [bid for bid in temp_overrides.removals if bid in selected_set]
    if removals:
        selected_set.difference_update(removals)
        selected_ids = [bid for bid in selected_ids if bid in selected_set]
        selected_candidates = [
            c for c in selected_candidates if getattr(c, "bullet_id", "") in selected_set
        ]

    additions: List[Dict[str, Any]] = []
    for idx, addition in enumerate(temp_overrides.additions or [], start=1):
        if not addition.text_latex.strip():
            raise HTTPException(status_code=400, detail="temp_additions text_latex is empty")
        temp_id = (addition.temp_id or "").strip() or f"tmp_{run_id}_{idx:03d}"
        prefix = "exp" if addition.parent_type == "experience" else "proj"
        bullet_id = f"{prefix}:{addition.parent_id}:{temp_id}"
        additions.append(
            {
                "temp_id": temp_id,
                "parent_type": addition.parent_type,
                "parent_id": addition.parent_id,
                "text_latex": addition.text_latex,
                "bullet_id": bullet_id,
            }
        )
        if auto_include_additions and bullet_id not in selected_set:
            selected_set.add(bullet_id)
            selected_ids.append(bullet_id)

    edits = {
        bid: text
        for bid, text in (temp_overrides.edits or {}).items()
        if bid in selected_set and isinstance(text, str) and text.strip()
    }

    normalized["removals"] = removals
    normalized["edits"] = edits
    normalized["additions"] = [item for item in additions if item["bullet_id"] in selected_set]
    return selected_ids, selected_candidates, normalized


def filter_temp_overrides_for_report(
    temp_overrides: Dict[str, Any], selected_ids: List[str]
) -> Dict[str, Any]:
    selected_set = set(selected_ids)
    additions = [
        addition
        for addition in (temp_overrides.get("additions", []) or [])
        if addition.get("bullet_id") in selected_set
    ]
    edits = {
        bid: text
        for bid, text in (temp_overrides.get("edits", {}) or {}).items()
        if bid in selected_set
    }
    removals = [
        bid for bid in (temp_overrides.get("removals", []) or []) if bid not in selected_set
    ]
    return {"additions": additions, "edits": edits, "removals": removals}


def has_temp_overrides(temp_overrides: Dict[str, Any]) -> bool:
    return bool(
        temp_overrides.get("additions")
        or temp_overrides.get("edits")
        or temp_overrides.get("removals")
    )


def build_tailored_snapshot(
    static_data: Dict[str, Any],
    selected_ids: List[str],
    *,
    selected_candidates: List[Any] | None = None,
    rewritten_bullets: Dict[str, str] | None = None,
    temp_overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    selected_set = set(selected_ids)
    temp_overrides = temp_overrides or {}
    rewritten_bullets = rewritten_bullets or {}
    temp_edits: Dict[str, str] = temp_overrides.get("edits", {}) or {}
    temp_additions: List[Dict[str, Any]] = temp_overrides.get("additions", []) or []
    tailored = copy.deepcopy(static_data)
    score_map: Dict[str, float] = {}
    for candidate in selected_candidates or []:
        score = getattr(candidate, "selection_score", None)
        if score is None:
            score = getattr(getattr(candidate, "best_hit", None), "weighted", 0.0)
        score_map[getattr(candidate, "bullet_id", "")] = float(score or 0.0)
    order_map = {bid: idx for idx, bid in enumerate(selected_ids)}
    use_order = not score_map

    for addition in temp_additions:
        parent_type = addition.get("parent_type")
        parent_id = addition.get("parent_id")
        temp_id = addition.get("temp_id")
        text_latex = addition.get("text_latex")
        if not parent_type or not parent_id or not temp_id or not text_latex:
            continue
        if parent_type == "experience":
            for exp in tailored.get("experiences", []) or []:
                if exp.get("job_id") == parent_id:
                    exp.setdefault("bullets", []).append({"id": temp_id, "text_latex": text_latex})
                    break
        elif parent_type == "project":
            for proj in tailored.get("projects", []) or []:
                if proj.get("project_id") == parent_id:
                    proj.setdefault("bullets", []).append({"id": temp_id, "text_latex": text_latex})
                    break

    def _rebuild(items: List[Dict[str, Any]], id_field: str, prefix: str) -> List[Dict[str, Any]]:
        rebuilt: List[Dict[str, Any]] = []
        for item in items:
            parent_id = item.get(id_field)
            kept: List[tuple[float, str, str]] = []
            for idx, bullet in enumerate(item.get("bullets", []) or []):
                local_id = bullet.get("id")
                if not parent_id or not local_id:
                    continue
                bullet_id = f"{prefix}:{parent_id}:{local_id}"
                if bullet_id not in selected_set:
                    continue
                score = score_map.get(bullet_id, 0.0)
                tie = local_id or f"idx:{idx:04d}"
                base_text = rewritten_bullets.get(bullet_id, bullet.get("text_latex", ""))
                text = temp_edits.get(bullet_id, base_text)
                order = order_map.get(bullet_id, len(order_map))
                kept.append((order if use_order else score, tie, text))
            if not kept:
                continue
            kept.sort(key=(lambda item: (item[0], item[1])) if use_order else (lambda item: (-item[0], item[1])))
            item["bullets"] = [text for _, _, text in kept]
            rebuilt.append(item)
        return rebuilt

    tailored["experiences"] = _rebuild(tailored.get("experiences", []) or [], "job_id", "exp")
    tailored["projects"] = _rebuild(tailored.get("projects", []) or [], "project_id", "proj")
    return tailored


def render_pdf(settings: Any, context: Dict[str, Any], run_id: str) -> Tuple[str, str]:
    os.makedirs(settings.output_dir, exist_ok=True)
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(settings.template_dir),
        block_start_string="((%",
        block_end_string="%))",
        variable_start_string="<<",
        variable_end_string=">>",
        comment_start_string="((#",
        comment_end_string="#))",
        autoescape=False,
    )

    local_template = os.path.join(settings.template_dir, "resume.local.tex")
    template_name = "resume.local.tex" if os.path.exists(local_template) else "resume.tex"
    template = env.get_template(template_name)
    tex_content = template.render(context)

    tex_path = os.path.join(settings.output_dir, f"{run_id}.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex_content)

    if settings.skip_pdf:
        pdf_path = os.path.join(settings.output_dir, f"{run_id}.pdf")
        with open(pdf_path, "wb") as f:
            f.write(b"")
        _write_output_pdf_alias(settings, pdf_path)
        return pdf_path, tex_path

    try:
        subprocess.run(
            ["tectonic", tex_path, "--outdir", settings.output_dir],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        logger.error("TECTONIC COMPILATION FAILED")
        logger.error("STDOUT (LaTeX Logs): %s", exc.stdout)
        logger.error("STDERR: %s", exc.stderr)
        raise

    pdf_path = os.path.join(settings.output_dir, f"{run_id}.pdf")
    _write_output_pdf_alias(settings, pdf_path)
    return pdf_path, tex_path


def pdf_page_count(path: str) -> int | None:
    try:
        reader = PdfReader(path)
        return len(reader.pages)
    except Exception as exc:
        logger.warning("Failed to read PDF page count: %s", exc)
        return None


def trim_to_single_page(
    settings: Any,
    run_id: str,
    static_data: Dict[str, Any],
    selected_ids: List[str],
    selected_candidates: List[Any],
    pdf_path: str,
    *,
    temp_overrides: Dict[str, Any] | None = None,
    rewritten_bullets: Dict[str, str] | None = None,
) -> Tuple[str, str, List[str], List[Any]]:
    if settings.skip_pdf:
        tex_path = os.path.join(settings.output_dir, f"{run_id}.tex")
        return pdf_path, tex_path, selected_ids, selected_candidates

    score_map: Dict[str, float] = {}
    for candidate in selected_candidates:
        score = getattr(candidate, "selection_score", None)
        if score is None:
            score = getattr(getattr(candidate, "best_hit", None), "weighted", 0.0)
        score_map[getattr(candidate, "bullet_id", "")] = float(score or 0.0)

    page_count = pdf_page_count(pdf_path)
    while page_count is not None and page_count > 1 and len(selected_ids) > 1:
        ranked = [(score_map.get(bid, 0.0), bid) for bid in selected_ids]
        ranked.sort(key=lambda item: (item[0], item[1]))
        drop_id = ranked[0][1] if ranked else ""
        if not drop_id:
            break
        logger.info("Trimming bullet %s to enforce single-page PDF", drop_id)
        selected_ids = [bid for bid in selected_ids if bid != drop_id]
        selected_candidates = [
            candidate
            for candidate in selected_candidates
            if getattr(candidate, "bullet_id", "") != drop_id
        ]
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        tailored = build_tailored_snapshot(
            static_data,
            selected_ids,
            selected_candidates=selected_candidates,
            rewritten_bullets=rewritten_bullets,
            temp_overrides=temp_overrides,
        )
        pdf_path, tex_path = render_pdf(settings, tailored, run_id)
        page_count = pdf_page_count(pdf_path)

    if page_count is not None and page_count > 1:
        logger.warning("PDF still exceeds one page after trimming to %d bullets", len(selected_ids))

    tex_path = os.path.join(settings.output_dir, f"{run_id}.tex")
    return pdf_path, tex_path, selected_ids, selected_candidates


def write_report(settings: Any, run_id: str, report: Dict[str, Any]) -> str:
    report_path = os.path.join(settings.output_dir, f"{run_id}_report.json")
    Path(report_path).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report_path


def update_report_selection(
    settings: Any,
    run_id: str,
    selected_ids: List[str],
    pdf_path: str,
    tex_path: str,
    *,
    rewritten_bullets: Dict[str, str] | None = None,
    temp_overrides: Dict[str, Any] | None = None,
) -> str:
    report_path = os.path.join(settings.output_dir, f"{run_id}_report.json")
    if not os.path.exists(report_path):
        return report_path
    try:
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    except Exception:
        report = {}
    report["selected_ids"] = selected_ids
    report["filtered_selection"] = True
    report.setdefault("artifacts", {})
    report["artifacts"]["pdf"] = os.path.basename(pdf_path)
    report["artifacts"]["tex"] = os.path.basename(tex_path)
    if rewritten_bullets and isinstance(report.get("rewritten_bullets"), list):
        selected_set = set(selected_ids)
        report["rewritten_bullets"] = [
            entry
            for entry in report["rewritten_bullets"]
            if entry.get("bullet_id") in selected_set
        ]
    if temp_overrides and has_temp_overrides(temp_overrides):
        report["temp_additions"] = temp_overrides.get("additions", [])
        report["temp_edits"] = temp_overrides.get("edits", {})
        report["temp_removals"] = temp_overrides.get("removals", [])
    else:
        report.pop("temp_additions", None)
        report.pop("temp_edits", None)
        report.pop("temp_removals", None)
    return write_report(settings, run_id, report)


def process_and_render_artifacts(
    settings: Any,
    run_id: str,
    static_data: Dict[str, Any],
    selected_ids: List[str],
    selected_candidates: List[Any],
    *,
    rewritten_bullets: Dict[str, str] | None = None,
    temp_overrides: Dict[str, Any] | None = None,
    base_report: Dict[str, Any] | None = None,
) -> Tuple[str, str, str, List[str], List[Any]]:
    """Shared service for tailoring, rendering, trimming, and report writing."""
    tailored = build_tailored_snapshot(
        static_data,
        selected_ids,
        selected_candidates=selected_candidates,
        rewritten_bullets=rewritten_bullets,
        temp_overrides=temp_overrides,
    )
    pdf_path, tex_path = render_pdf(settings, tailored, run_id)
    pdf_path, tex_path, final_selected_ids, final_candidates = trim_to_single_page(
        settings,
        run_id,
        static_data,
        selected_ids,
        selected_candidates,
        pdf_path,
        temp_overrides=temp_overrides,
        rewritten_bullets=rewritten_bullets,
    )

    filtered_temp_overrides = None
    if temp_overrides:
        filtered_temp_overrides = filter_temp_overrides_for_report(temp_overrides, final_selected_ids)

    if base_report is not None:
        report = dict(base_report)
        report["selected_ids"] = final_selected_ids
        report.setdefault("artifacts", {})
        report["artifacts"]["pdf"] = os.path.basename(pdf_path)
        report["artifacts"]["tex"] = os.path.basename(tex_path)
        if rewritten_bullets and isinstance(report.get("rewritten_bullets"), list):
            selected_set = set(final_selected_ids)
            report["rewritten_bullets"] = [
                entry for entry in report["rewritten_bullets"]
                if entry.get("bullet_id") in selected_set
            ]
        if filtered_temp_overrides and has_temp_overrides(filtered_temp_overrides):
            report["temp_additions"] = filtered_temp_overrides.get("additions", [])
            report["temp_edits"] = filtered_temp_overrides.get("edits", {})
            report["temp_removals"] = filtered_temp_overrides.get("removals", [])
        else:
            report.pop("temp_additions", None)
            report.pop("temp_edits", None)
            report.pop("temp_removals", None)
        report_path = write_report(settings, run_id, report)
    else:
        report_path = update_report_selection(
            settings,
            run_id,
            final_selected_ids,
            pdf_path,
            tex_path,
            rewritten_bullets=rewritten_bullets,
            temp_overrides=filtered_temp_overrides,
        )

    return pdf_path, tex_path, report_path, final_selected_ids, final_candidates
