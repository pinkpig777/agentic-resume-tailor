from __future__ import annotations

from typing import Any, Dict

from agentic_resume_tailor.db.models import Education, Experience, PersonalInfo, Project, Skills


def experience_to_dict(exp: Experience) -> Dict[str, Any]:
    bullets = sorted(exp.bullets, key=lambda bullet: (bullet.sort_order, bullet.id))
    return {
        "job_id": exp.job_id,
        "company": exp.company,
        "role": exp.role,
        "dates": exp.dates,
        "location": exp.location,
        "sort_order": exp.sort_order,
        "bullets": [
            {"id": bullet.local_id, "text_latex": bullet.text_latex, "sort_order": bullet.sort_order}
            for bullet in bullets
        ],
    }


def project_to_dict(proj: Project) -> Dict[str, Any]:
    bullets = sorted(proj.bullets, key=lambda bullet: (bullet.sort_order, bullet.id))
    return {
        "project_id": proj.project_id,
        "name": proj.name,
        "technologies": proj.technologies,
        "sort_order": proj.sort_order,
        "bullets": [
            {"id": bullet.local_id, "text_latex": bullet.text_latex, "sort_order": bullet.sort_order}
            for bullet in bullets
        ],
    }


def education_to_dict(edu: Education) -> Dict[str, Any]:
    bullets = sorted(edu.bullets, key=lambda bullet: (bullet.sort_order, bullet.id))
    return {
        "id": edu.id,
        "school": edu.school,
        "dates": edu.dates,
        "degree": edu.degree,
        "location": edu.location,
        "sort_order": edu.sort_order,
        "bullets": [bullet.text_latex for bullet in bullets],
    }


def personal_info_to_dict(info: PersonalInfo | None) -> Dict[str, str]:
    return {
        "name": info.name if info else "",
        "phone": info.phone if info else "",
        "email": info.email if info else "",
        "linkedin_id": info.linkedin_id if info else "",
        "github_id": info.github_id if info else "",
        "linkedin": info.linkedin if info else "",
        "github": info.github if info else "",
    }


def skills_to_dict(skills: Skills | None) -> Dict[str, str]:
    return {
        "languages_frameworks": skills.languages_frameworks if skills else "",
        "ai_ml": skills.ai_ml if skills else "",
        "db_tools": skills.db_tools if skills else "",
    }
