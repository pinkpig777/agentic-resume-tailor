from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

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

SCHEMA_VERSION = "my_experience_v2"


def export_resume_data(session: Session) -> Dict[str, Any]:
    """
    Export the resume profile from the DB in the normalized JSON schema.

    Includes personal_info, skills, education, experiences, projects, and stable bullet ids.
    Orders entries by sort_order then id for consistent display.
    """
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
    """
    Export resume data and write it to a JSON file.

    Ensures the parent directory exists and returns the exported dict.
    """
    data = export_resume_data(session)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return data
