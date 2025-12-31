from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from sqlalchemy import func
from sqlalchemy.orm import Session

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
from agentic_resume_tailor.db.utils import (
    ensure_unique_slug,
    make_job_id,
    make_project_id,
    next_bullet_id,
)

SCHEMA_VERSION = "my_experience_v2"


def export_resume_data(session: Session) -> Dict[str, Any]:
    personal = session.query(PersonalInfo).first()
    skills = session.query(Skills).first()

    data: Dict[str, Any] = {
        "personal_info": {
            "name": personal.name if personal else "",
            "phone": personal.phone if personal else "",
            "email": personal.email if personal else "",
            "linkedin_id": personal.linkedin_id if personal else "",
            "github_id": personal.github_id if personal else "",
            "linkedin": personal.linkedin if personal else "",
            "github": personal.github if personal else "",
        },
        "skills": {
            "languages_frameworks": skills.languages_frameworks if skills else "",
            "ai_ml": skills.ai_ml if skills else "",
            "db_tools": skills.db_tools if skills else "",
        },
        "education": [],
        "experiences": [],
        "projects": [],
        "schema_version": SCHEMA_VERSION,
    }

    educations = (
        session.query(Education)
        .order_by(Education.sort_order.asc(), Education.id.asc())
        .all()
    )
    for edu in educations:
        bullets = (
            session.query(EducationBullet)
            .filter(EducationBullet.education_id == edu.id)
            .order_by(EducationBullet.sort_order.asc(), EducationBullet.id.asc())
            .all()
        )
        data["education"].append(
            {
                "school": edu.school or "",
                "dates": edu.dates or "",
                "degree": edu.degree or "",
                "location": edu.location or "",
                "bullets": [b.text_latex or "" for b in bullets],
            }
        )

    experiences = (
        session.query(Experience)
        .order_by(Experience.sort_order.asc(), Experience.id.asc())
        .all()
    )
    for exp in experiences:
        bullets = (
            session.query(ExperienceBullet)
            .filter(ExperienceBullet.experience_id == exp.id)
            .order_by(ExperienceBullet.sort_order.asc(), ExperienceBullet.id.asc())
            .all()
        )
        data["experiences"].append(
            {
                "company": exp.company or "",
                "role": exp.role or "",
                "dates": exp.dates or "",
                "location": exp.location or "",
                "job_id": exp.job_id,
                "bullets": [
                    {"id": b.local_id, "text_latex": b.text_latex or ""} for b in bullets
                ],
            }
        )

    projects = (
        session.query(Project).order_by(Project.sort_order.asc(), Project.id.asc()).all()
    )
    for proj in projects:
        bullets = (
            session.query(ProjectBullet)
            .filter(ProjectBullet.project_id == proj.id)
            .order_by(ProjectBullet.sort_order.asc(), ProjectBullet.id.asc())
            .all()
        )
        data["projects"].append(
            {
                "name": proj.name or "",
                "technologies": proj.technologies or "",
                "project_id": proj.project_id,
                "bullets": [
                    {"id": b.local_id, "text_latex": b.text_latex or ""} for b in bullets
                ],
            }
        )

    return data


def write_resume_json(session: Session, path: str) -> Dict[str, Any]:
    data = export_resume_data(session)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return data


def _normalize_bullets(items: Iterable[Any]) -> List[Tuple[str, str]]:
    existing_ids: List[str] = []
    raw_items = list(items or [])
    for item in raw_items:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            existing_ids.append(item["id"].strip())

    used_ids: set[str] = set()
    normalized: List[Tuple[str, str]] = []

    def allocate_id(text: str) -> str:
        return next_bullet_id(existing_ids + list(used_ids))

    for item in raw_items:
        if isinstance(item, str):
            text = item
            bullet_id = allocate_id(text)
            used_ids.add(bullet_id)
            normalized.append((bullet_id, text))
        elif isinstance(item, dict):
            text = item.get("text_latex") or item.get("text")
            if not isinstance(text, str):
                continue
            bullet_id = str(item.get("id") or "").strip()
            if not bullet_id or bullet_id in used_ids:
                bullet_id = allocate_id(text)
            used_ids.add(bullet_id)
            normalized.append((bullet_id, text))

    return normalized


def _db_has_rows(session: Session) -> bool:
    exp_count = session.query(func.count(Experience.id)).scalar() or 0
    proj_count = session.query(func.count(Project.id)).scalar() or 0
    edu_count = session.query(func.count(Education.id)).scalar() or 0
    info_count = session.query(func.count(PersonalInfo.id)).scalar() or 0
    skills_count = session.query(func.count(Skills.id)).scalar() or 0
    return (exp_count + proj_count + edu_count + info_count + skills_count) > 0


def seed_db_if_empty(session: Session, data_path: str) -> bool:
    if _db_has_rows(session):
        return False

    path = Path(data_path)
    if not path.exists():
        return False

    data = json.loads(path.read_text(encoding="utf-8"))

    personal = data.get("personal_info", {}) or {}
    personal_row = PersonalInfo(
        name=str(personal.get("name", "") or ""),
        phone=str(personal.get("phone", "") or ""),
        email=str(personal.get("email", "") or ""),
        linkedin_id=str(personal.get("linkedin_id", "") or ""),
        github_id=str(personal.get("github_id", "") or ""),
        linkedin=str(personal.get("linkedin", "") or ""),
        github=str(personal.get("github", "") or ""),
    )
    session.add(personal_row)

    skills = data.get("skills", {}) or {}
    skills_row = Skills(
        languages_frameworks=str(skills.get("languages_frameworks", "") or ""),
        ai_ml=str(skills.get("ai_ml", "") or ""),
        db_tools=str(skills.get("db_tools", "") or ""),
    )
    session.add(skills_row)

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
        session.add(edu_row)
        session.flush()
        for b_idx, bullet in enumerate(edu.get("bullets", []) or [], start=1):
            if not isinstance(bullet, str):
                continue
            session.add(
                EducationBullet(
                    education_id=edu_row.id,
                    text_latex=bullet,
                    sort_order=b_idx,
                )
            )

    used_job_ids: set[str] = set()
    for idx, exp in enumerate(data.get("experiences", []) or [], start=1):
        if not isinstance(exp, dict):
            continue
        company = str(exp.get("company", "") or "")
        role = str(exp.get("role", "") or "")
        job_id = str(exp.get("job_id") or make_job_id(company, role))
        job_id = ensure_unique_slug(job_id, used_job_ids)
        used_job_ids.add(job_id)

        exp_row = Experience(
            job_id=job_id,
            company=company,
            role=role,
            dates=str(exp.get("dates", "") or ""),
            location=str(exp.get("location", "") or ""),
            sort_order=idx,
        )
        session.add(exp_row)
        session.flush()

        bullets = _normalize_bullets(exp.get("bullets", []) or [])
        for b_idx, (local_id, text) in enumerate(bullets, start=1):
            session.add(
                ExperienceBullet(
                    experience_id=exp_row.id,
                    local_id=local_id,
                    text_latex=text,
                    sort_order=b_idx,
                )
            )

    used_project_ids: set[str] = set()
    for idx, proj in enumerate(data.get("projects", []) or [], start=1):
        if not isinstance(proj, dict):
            continue
        name = str(proj.get("name", "") or "")
        project_id = str(proj.get("project_id") or make_project_id(name))
        project_id = ensure_unique_slug(project_id, used_project_ids)
        used_project_ids.add(project_id)

        proj_row = Project(
            project_id=project_id,
            name=name,
            technologies=str(proj.get("technologies", "") or ""),
            sort_order=idx,
        )
        session.add(proj_row)
        session.flush()

        bullets = _normalize_bullets(proj.get("bullets", []) or [])
        for b_idx, (local_id, text) in enumerate(bullets, start=1):
            session.add(
                ProjectBullet(
                    project_id=proj_row.id,
                    local_id=local_id,
                    text_latex=text,
                    sort_order=b_idx,
                )
            )

    session.commit()
    return True
