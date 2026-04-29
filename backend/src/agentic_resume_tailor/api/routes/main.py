from __future__ import annotations

import os
import time
from queue import Empty
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from sqlalchemy import func
from sqlalchemy.orm import Session, selectinload

from agentic_resume_tailor.api.runtime import (
    emit_progress,
    format_sse,
    get_or_create_progress,
    load_runtime_settings,
    refresh_collection,
    require_collection,
    settings_payload,
    update_settings_payload,
)
from agentic_resume_tailor.api.schemas import (
    BulletCreate,
    BulletUpdate,
    EducationCreate,
    EducationUpdate,
    ExperienceCreate,
    ExperienceUpdate,
    GenerateRequest,
    GenerateResponse,
    GenerateV3Request,
    PersonalInfoUpdate,
    ProjectCreate,
    ProjectUpdate,
    RenderSelectionRequest,
    SkillsUpdate,
)
from agentic_resume_tailor.api.serializers import (
    education_to_dict,
    experience_to_dict,
    personal_info_to_dict,
    project_to_dict,
    skills_to_dict,
)
from agentic_resume_tailor.core.artifacts import (
    apply_temp_overrides,
    normalize_output_pdf_name,
    process_and_render_artifacts,
)
from agentic_resume_tailor.core.loop_controller import generate_run_id, run_loop
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
from agentic_resume_tailor.db.session import SessionLocal, get_db
from agentic_resume_tailor.db.sync import export_resume_data, write_resume_json
from agentic_resume_tailor.db.utils import (
    ensure_unique_slug,
    make_job_id,
    make_project_id,
    next_bullet_id,
    next_sort_order,
)

router = APIRouter()


def _load_static_data() -> Dict[str, Any]:
    with SessionLocal() as db:
        return export_resume_data(db)


def _next_sort_order_for(query) -> int:
    return next_sort_order([query.scalar()])


def _export_latest(db: Session, settings: Any) -> None:
    write_resume_json(db, settings.export_file)


def _maybe_auto_reingest(request: Request, settings: Any) -> None:
    if not getattr(settings, "auto_reingest_on_save", False):
        return
    if not request.app.state.ingest_lock.acquire(blocking=False):
        return
    try:
        from agentic_resume_tailor import ingest as ingest_module

        ingest_module.ingest(json_path=settings.export_file, settings=settings)
        refresh_collection(request.app, settings)
    finally:
        request.app.state.ingest_lock.release()


def _generation_settings(req: GenerateRequest, settings: Any) -> Any:
    overrides = {}
    for field in (
        "max_bullets",
        "per_query_k",
        "final_k",
        "max_iters",
        "threshold",
        "alpha",
        "must_weight",
        "boost_weight",
        "boost_top_n_missing",
    ):
        value = getattr(req, field)
        if value is not None:
            overrides[field] = value
    if req.enable_bullet_rewrite is not None:
        overrides["enable_bullet_rewrite"] = req.enable_bullet_rewrite
    if req.rewrite_style is not None:
        overrides["rewrite_style"] = req.rewrite_style
    return settings.model_copy(update=overrides)


@router.get("/health")
def health():
    settings = load_runtime_settings()
    return {
        "status": "ok",
        "collection": settings.collection_name,
        "embed_model": settings.embed_model,
    }


@router.get("/settings")
def get_user_settings():
    return settings_payload()


@router.put("/settings")
def update_user_settings(payload: Dict[str, Any]):
    return update_settings_payload(payload)


@router.get("/personal_info")
def get_personal_info(db: Session = Depends(get_db)):
    return personal_info_to_dict(db.query(PersonalInfo).first())


@router.put("/personal_info")
def update_personal_info(payload: PersonalInfoUpdate, request: Request, db: Session = Depends(get_db)):
    settings = load_runtime_settings()
    info = db.query(PersonalInfo).first()
    if info is None:
        info = PersonalInfo()
        db.add(info)
    for field in ("name", "phone", "email", "linkedin_id", "github_id", "linkedin", "github"):
        value = getattr(payload, field)
        if value is not None:
            setattr(info, field, value)
    db.commit()
    _export_latest(db, settings)
    _maybe_auto_reingest(request, settings)
    db.refresh(info)
    return personal_info_to_dict(info)


@router.get("/skills")
def get_skills(db: Session = Depends(get_db)):
    return skills_to_dict(db.query(Skills).first())


@router.put("/skills")
def update_skills(payload: SkillsUpdate, request: Request, db: Session = Depends(get_db)):
    settings = load_runtime_settings()
    skills = db.query(Skills).first()
    if skills is None:
        skills = Skills()
        db.add(skills)
    for field in ("languages_frameworks", "ai_ml", "db_tools"):
        value = getattr(payload, field)
        if value is not None:
            setattr(skills, field, value)
    db.commit()
    _export_latest(db, settings)
    _maybe_auto_reingest(request, settings)
    db.refresh(skills)
    return skills_to_dict(skills)


@router.get("/education")
def list_education(db: Session = Depends(get_db)):
    educations = (
        db.query(Education)
        .options(selectinload(Education.bullets))
        .order_by(Education.sort_order.asc(), Education.id.asc())
        .all()
    )
    return [education_to_dict(edu) for edu in educations]


@router.post("/education")
def create_education(payload: EducationCreate, request: Request, db: Session = Depends(get_db)):
    settings = load_runtime_settings()
    sort_order = payload.sort_order
    if sort_order is None:
        sort_order = _next_sort_order_for(db.query(func.max(Education.sort_order)))
    edu = Education(
        school=payload.school,
        dates=payload.dates,
        degree=payload.degree,
        location=payload.location,
        sort_order=sort_order,
    )
    db.add(edu)
    db.flush()
    for idx, bullet in enumerate(payload.bullets, start=1):
        if bullet:
            db.add(EducationBullet(education_id=edu.id, text_latex=bullet, sort_order=idx))
    db.commit()
    _export_latest(db, settings)
    _maybe_auto_reingest(request, settings)
    edu = (
        db.query(Education)
        .options(selectinload(Education.bullets))
        .filter(Education.id == edu.id)
        .first()
    )
    return education_to_dict(edu)


@router.put("/education/{education_id}")
def update_education(
    education_id: int,
    payload: EducationUpdate,
    request: Request,
    db: Session = Depends(get_db),
):
    settings = load_runtime_settings()
    edu = (
        db.query(Education)
        .options(selectinload(Education.bullets))
        .filter(Education.id == education_id)
        .first()
    )
    if edu is None:
        raise HTTPException(status_code=404, detail="Education entry not found")
    for field in ("school", "dates", "degree", "location", "sort_order"):
        value = getattr(payload, field)
        if value is not None:
            setattr(edu, field, value)
    if payload.bullets is not None:
        db.query(EducationBullet).filter(EducationBullet.education_id == edu.id).delete()
        for idx, bullet in enumerate(payload.bullets, start=1):
            if bullet:
                db.add(EducationBullet(education_id=edu.id, text_latex=bullet, sort_order=idx))
    db.commit()
    _export_latest(db, settings)
    _maybe_auto_reingest(request, settings)
    edu = (
        db.query(Education)
        .options(selectinload(Education.bullets))
        .filter(Education.id == edu.id)
        .first()
    )
    return education_to_dict(edu)


@router.delete("/education/{education_id}")
def delete_education(education_id: int, request: Request, db: Session = Depends(get_db)):
    settings = load_runtime_settings()
    edu = db.query(Education).filter(Education.id == education_id).first()
    if edu is None:
        raise HTTPException(status_code=404, detail="Education entry not found")
    db.delete(edu)
    db.commit()
    _export_latest(db, settings)
    _maybe_auto_reingest(request, settings)
    return {"status": "deleted", "id": education_id}


@router.get("/experiences")
def list_experiences(db: Session = Depends(get_db)):
    experiences = (
        db.query(Experience)
        .options(selectinload(Experience.bullets))
        .order_by(Experience.sort_order.asc(), Experience.id.asc())
        .all()
    )
    return [experience_to_dict(exp) for exp in experiences]


@router.get("/experiences/{job_id}")
def get_experience(job_id: str, db: Session = Depends(get_db)):
    exp = (
        db.query(Experience)
        .options(selectinload(Experience.bullets))
        .filter(Experience.job_id == job_id)
        .first()
    )
    if exp is None:
        raise HTTPException(status_code=404, detail="Experience not found")
    return experience_to_dict(exp)


@router.post("/experiences")
def create_experience(payload: ExperienceCreate, request: Request, db: Session = Depends(get_db)):
    settings = load_runtime_settings()
    job_id = make_job_id(payload.company, payload.role)
    if db.query(Experience).filter(Experience.job_id == job_id).first():
        raise HTTPException(status_code=409, detail="Experience with job_id already exists")
    sort_order = payload.sort_order
    if sort_order is None:
        sort_order = _next_sort_order_for(db.query(func.max(Experience.sort_order)))
    exp = Experience(
        job_id=job_id,
        company=payload.company,
        role=payload.role,
        dates=payload.dates,
        location=payload.location,
        sort_order=sort_order,
    )
    db.add(exp)
    db.flush()
    existing_ids: list[str] = []
    for idx, bullet in enumerate(payload.bullets, start=1):
        if not bullet:
            continue
        local_id = next_bullet_id(existing_ids)
        existing_ids.append(local_id)
        db.add(
            ExperienceBullet(
                experience_id=exp.id,
                local_id=local_id,
                text_latex=bullet,
                sort_order=idx,
            )
        )
    db.commit()
    _export_latest(db, settings)
    _maybe_auto_reingest(request, settings)
    exp = (
        db.query(Experience)
        .options(selectinload(Experience.bullets))
        .filter(Experience.id == exp.id)
        .first()
    )
    return experience_to_dict(exp)


@router.put("/experiences/{job_id}")
def update_experience(
    job_id: str,
    payload: ExperienceUpdate,
    request: Request,
    db: Session = Depends(get_db),
):
    settings = load_runtime_settings()
    exp = (
        db.query(Experience)
        .options(selectinload(Experience.bullets))
        .filter(Experience.job_id == job_id)
        .first()
    )
    if exp is None:
        raise HTTPException(status_code=404, detail="Experience not found")
    exp_id = exp.id
    for field in ("company", "role", "dates", "location", "sort_order"):
        value = getattr(payload, field)
        if value is not None:
            setattr(exp, field, value)
    new_job_id = make_job_id(exp.company, exp.role)
    if new_job_id != exp.job_id:
        conflict = (
            db.query(Experience)
            .filter(Experience.job_id == new_job_id, Experience.id != exp.id)
            .first()
        )
        if conflict:
            raise HTTPException(status_code=409, detail="job_id collision for updated experience")
        exp.job_id = new_job_id
    db.commit()
    _export_latest(db, settings)
    _maybe_auto_reingest(request, settings)
    exp = (
        db.query(Experience)
        .options(selectinload(Experience.bullets))
        .filter(Experience.id == exp_id)
        .first()
    )
    return experience_to_dict(exp)


@router.delete("/experiences/{job_id}")
def delete_experience(job_id: str, request: Request, db: Session = Depends(get_db)):
    settings = load_runtime_settings()
    exp = db.query(Experience).filter(Experience.job_id == job_id).first()
    if exp is None:
        raise HTTPException(status_code=404, detail="Experience not found")
    db.delete(exp)
    db.commit()
    _export_latest(db, settings)
    _maybe_auto_reingest(request, settings)
    return {"status": "deleted", "job_id": job_id}


@router.get("/experiences/{job_id}/bullets")
def list_experience_bullets(job_id: str, db: Session = Depends(get_db)):
    exp = db.query(Experience).filter(Experience.job_id == job_id).first()
    if exp is None:
        raise HTTPException(status_code=404, detail="Experience not found")
    bullets = (
        db.query(ExperienceBullet)
        .filter(ExperienceBullet.experience_id == exp.id)
        .order_by(ExperienceBullet.sort_order.asc(), ExperienceBullet.id.asc())
        .all()
    )
    return [
        {"id": bullet.local_id, "text_latex": bullet.text_latex, "sort_order": bullet.sort_order}
        for bullet in bullets
    ]


@router.post("/experiences/{job_id}/bullets")
def create_experience_bullet(
    job_id: str,
    payload: BulletCreate,
    request: Request,
    db: Session = Depends(get_db),
):
    settings = load_runtime_settings()
    exp = db.query(Experience).filter(Experience.job_id == job_id).first()
    if exp is None:
        raise HTTPException(status_code=404, detail="Experience not found")
    existing_ids = [
        row[0]
        for row in db.query(ExperienceBullet.local_id)
        .filter(ExperienceBullet.experience_id == exp.id)
        .all()
    ]
    local_id = next_bullet_id(existing_ids)
    sort_order = payload.sort_order
    if sort_order is None:
        sort_order = _next_sort_order_for(
            db.query(func.max(ExperienceBullet.sort_order)).filter(
                ExperienceBullet.experience_id == exp.id
            )
        )
    bullet = ExperienceBullet(
        experience_id=exp.id,
        local_id=local_id,
        text_latex=payload.text_latex,
        sort_order=sort_order,
    )
    db.add(bullet)
    db.commit()
    _export_latest(db, settings)
    _maybe_auto_reingest(request, settings)
    return {"id": local_id, "text_latex": bullet.text_latex, "sort_order": bullet.sort_order}


@router.put("/experiences/{job_id}/bullets/{local_id}")
def update_experience_bullet(
    job_id: str,
    local_id: str,
    payload: BulletUpdate,
    request: Request,
    db: Session = Depends(get_db),
):
    settings = load_runtime_settings()
    bullet = (
        db.query(ExperienceBullet)
        .join(Experience, Experience.id == ExperienceBullet.experience_id)
        .filter(Experience.job_id == job_id, ExperienceBullet.local_id == local_id)
        .first()
    )
    if bullet is None:
        raise HTTPException(status_code=404, detail="Experience bullet not found")
    if payload.text_latex is not None:
        bullet.text_latex = payload.text_latex
    if payload.sort_order is not None:
        bullet.sort_order = payload.sort_order
    db.commit()
    _export_latest(db, settings)
    _maybe_auto_reingest(request, settings)
    return {"id": bullet.local_id, "text_latex": bullet.text_latex, "sort_order": bullet.sort_order}


@router.delete("/experiences/{job_id}/bullets/{local_id}")
def delete_experience_bullet(
    job_id: str,
    local_id: str,
    request: Request,
    db: Session = Depends(get_db),
):
    settings = load_runtime_settings()
    bullet = (
        db.query(ExperienceBullet)
        .join(Experience, Experience.id == ExperienceBullet.experience_id)
        .filter(Experience.job_id == job_id, ExperienceBullet.local_id == local_id)
        .first()
    )
    if bullet is None:
        raise HTTPException(status_code=404, detail="Experience bullet not found")
    db.delete(bullet)
    db.commit()
    _export_latest(db, settings)
    _maybe_auto_reingest(request, settings)
    return {"status": "deleted", "id": local_id}


@router.get("/projects")
def list_projects(db: Session = Depends(get_db)):
    projects = (
        db.query(Project)
        .options(selectinload(Project.bullets))
        .order_by(Project.sort_order.asc(), Project.id.asc())
        .all()
    )
    return [project_to_dict(proj) for proj in projects]


@router.get("/projects/{project_id}")
def get_project(project_id: str, db: Session = Depends(get_db)):
    proj = (
        db.query(Project)
        .options(selectinload(Project.bullets))
        .filter(Project.project_id == project_id)
        .first()
    )
    if proj is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return project_to_dict(proj)


@router.post("/projects")
def create_project(payload: ProjectCreate, request: Request, db: Session = Depends(get_db)):
    settings = load_runtime_settings()
    base_id = make_project_id(payload.name)
    existing_ids = [row[0] for row in db.query(Project.project_id).all()]
    project_id = ensure_unique_slug(base_id, existing_ids)
    sort_order = payload.sort_order
    if sort_order is None:
        sort_order = _next_sort_order_for(db.query(func.max(Project.sort_order)))
    proj = Project(
        project_id=project_id,
        name=payload.name,
        technologies=payload.technologies,
        sort_order=sort_order,
    )
    db.add(proj)
    db.flush()
    existing_ids = []
    for idx, bullet in enumerate(payload.bullets, start=1):
        if not bullet:
            continue
        local_id = next_bullet_id(existing_ids)
        existing_ids.append(local_id)
        db.add(
            ProjectBullet(
                project_id=proj.id,
                local_id=local_id,
                text_latex=bullet,
                sort_order=idx,
            )
        )
    db.commit()
    _export_latest(db, settings)
    _maybe_auto_reingest(request, settings)
    proj = (
        db.query(Project)
        .options(selectinload(Project.bullets))
        .filter(Project.id == proj.id)
        .first()
    )
    return project_to_dict(proj)


@router.put("/projects/{project_id}")
def update_project(
    project_id: str,
    payload: ProjectUpdate,
    request: Request,
    db: Session = Depends(get_db),
):
    settings = load_runtime_settings()
    proj = (
        db.query(Project)
        .options(selectinload(Project.bullets))
        .filter(Project.project_id == project_id)
        .first()
    )
    if proj is None:
        raise HTTPException(status_code=404, detail="Project not found")
    proj_id = proj.id
    old_name = proj.name
    for field in ("name", "technologies", "sort_order"):
        value = getattr(payload, field)
        if value is not None:
            setattr(proj, field, value)
    if payload.name is not None and proj.name != old_name:
        new_base = make_project_id(proj.name)
        if new_base != proj.project_id:
            existing_ids = [
                row[0] for row in db.query(Project.project_id).filter(Project.id != proj.id).all()
            ]
            proj.project_id = ensure_unique_slug(new_base, existing_ids)
    db.commit()
    _export_latest(db, settings)
    _maybe_auto_reingest(request, settings)
    proj = (
        db.query(Project)
        .options(selectinload(Project.bullets))
        .filter(Project.id == proj_id)
        .first()
    )
    return project_to_dict(proj)


@router.delete("/projects/{project_id}")
def delete_project(project_id: str, request: Request, db: Session = Depends(get_db)):
    settings = load_runtime_settings()
    proj = db.query(Project).filter(Project.project_id == project_id).first()
    if proj is None:
        raise HTTPException(status_code=404, detail="Project not found")
    db.delete(proj)
    db.commit()
    _export_latest(db, settings)
    _maybe_auto_reingest(request, settings)
    return {"status": "deleted", "project_id": project_id}


@router.get("/projects/{project_id}/bullets")
def list_project_bullets(project_id: str, db: Session = Depends(get_db)):
    proj = db.query(Project).filter(Project.project_id == project_id).first()
    if proj is None:
        raise HTTPException(status_code=404, detail="Project not found")
    bullets = (
        db.query(ProjectBullet)
        .filter(ProjectBullet.project_id == proj.id)
        .order_by(ProjectBullet.sort_order.asc(), ProjectBullet.id.asc())
        .all()
    )
    return [
        {"id": bullet.local_id, "text_latex": bullet.text_latex, "sort_order": bullet.sort_order}
        for bullet in bullets
    ]


@router.post("/projects/{project_id}/bullets")
def create_project_bullet(
    project_id: str,
    payload: BulletCreate,
    request: Request,
    db: Session = Depends(get_db),
):
    settings = load_runtime_settings()
    proj = db.query(Project).filter(Project.project_id == project_id).first()
    if proj is None:
        raise HTTPException(status_code=404, detail="Project not found")
    existing_ids = [
        row[0]
        for row in db.query(ProjectBullet.local_id).filter(ProjectBullet.project_id == proj.id).all()
    ]
    local_id = next_bullet_id(existing_ids)
    sort_order = payload.sort_order
    if sort_order is None:
        sort_order = _next_sort_order_for(
            db.query(func.max(ProjectBullet.sort_order)).filter(ProjectBullet.project_id == proj.id)
        )
    bullet = ProjectBullet(
        project_id=proj.id,
        local_id=local_id,
        text_latex=payload.text_latex,
        sort_order=sort_order,
    )
    db.add(bullet)
    db.commit()
    _export_latest(db, settings)
    _maybe_auto_reingest(request, settings)
    return {"id": local_id, "text_latex": bullet.text_latex, "sort_order": bullet.sort_order}


@router.put("/projects/{project_id}/bullets/{local_id}")
def update_project_bullet(
    project_id: str,
    local_id: str,
    payload: BulletUpdate,
    request: Request,
    db: Session = Depends(get_db),
):
    settings = load_runtime_settings()
    bullet = (
        db.query(ProjectBullet)
        .join(Project, Project.id == ProjectBullet.project_id)
        .filter(Project.project_id == project_id, ProjectBullet.local_id == local_id)
        .first()
    )
    if bullet is None:
        raise HTTPException(status_code=404, detail="Project bullet not found")
    if payload.text_latex is not None:
        bullet.text_latex = payload.text_latex
    if payload.sort_order is not None:
        bullet.sort_order = payload.sort_order
    db.commit()
    _export_latest(db, settings)
    _maybe_auto_reingest(request, settings)
    return {"id": bullet.local_id, "text_latex": bullet.text_latex, "sort_order": bullet.sort_order}


@router.delete("/projects/{project_id}/bullets/{local_id}")
def delete_project_bullet(
    project_id: str,
    local_id: str,
    request: Request,
    db: Session = Depends(get_db),
):
    settings = load_runtime_settings()
    bullet = (
        db.query(ProjectBullet)
        .join(Project, Project.id == ProjectBullet.project_id)
        .filter(Project.project_id == project_id, ProjectBullet.local_id == local_id)
        .first()
    )
    if bullet is None:
        raise HTTPException(status_code=404, detail="Project bullet not found")
    db.delete(bullet)
    db.commit()
    _export_latest(db, settings)
    _maybe_auto_reingest(request, settings)
    return {"status": "deleted", "id": local_id}


@router.post("/admin/export")
def export_resume(request: Request, reingest: bool = False, db: Session = Depends(get_db)):
    settings = load_runtime_settings()
    write_resume_json(db, settings.export_file)
    reingested = False
    if reingest:
        from agentic_resume_tailor import ingest as ingest_module

        ingest_module.ingest(json_path=settings.export_file, settings=settings)
        refresh_collection(request.app, settings)
        reingested = True
    return {"status": "ok", "path": settings.export_file, "reingested": reingested}


@router.post("/admin/ingest")
def ingest_resume(request: Request, db: Session = Depends(get_db)):
    settings = load_runtime_settings()
    if not request.app.state.ingest_lock.acquire(blocking=False):
        return JSONResponse(
            {"status": "error", "count": 0, "elapsed_s": 0.0, "error": "ingest already running"},
            status_code=409,
        )

    start = time.time()
    try:
        write_resume_json(db, settings.export_file)
        from agentic_resume_tailor import ingest as ingest_module

        count = ingest_module.ingest(json_path=settings.export_file, settings=settings)
        refresh_collection(request.app, settings)
        elapsed = time.time() - start
        return {"status": "ok", "count": count, "elapsed_s": round(elapsed, 2)}
    except Exception as exc:
        elapsed = time.time() - start
        return JSONResponse(
            {
                "status": "error",
                "count": 0,
                "elapsed_s": round(elapsed, 2),
                "error": str(exc),
            },
            status_code=500,
        )
    finally:
        request.app.state.ingest_lock.release()


@router.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest, request: Request) -> GenerateResponse:
    jd_text = (req.jd_text or "").strip()
    if not jd_text:
        return JSONResponse({"error": "jd_text is empty"}, status_code=400)

    settings = _generation_settings(req, load_runtime_settings())
    static_data = _load_static_data()
    run_id = req.run_id or generate_run_id(settings)
    get_or_create_progress(request.app, run_id, max_iters=settings.max_iters)
    collection, embedding_fn = require_collection(request.app)
    try:
        artifacts = await run_in_threadpool(
            run_loop,
            jd_text,
            collection=collection,
            embedding_fn=embedding_fn,
            static_export=static_data,
            settings=settings,
            run_id=run_id,
            progress_cb=lambda payload: emit_progress(request.app, run_id, payload),
        )
    except Exception as exc:
        emit_progress(request.app, run_id, {"stage": "error", "status": "error", "message": str(exc)})
        raise

    return GenerateResponse(
        run_id=artifacts.run_id,
        profile_used=artifacts.profile_used,
        best_iteration_index=artifacts.best_iteration_index,
        pdf_url=f"/runs/{artifacts.run_id}/pdf",
        tex_url=f"/runs/{artifacts.run_id}/tex",
        report_url=f"/runs/{artifacts.run_id}/report",
    )


@router.post("/generate_v3", response_model=GenerateResponse)
async def generate_v3(req: GenerateV3Request, request: Request) -> GenerateResponse:
    return await generate(req, request)


@router.post("/runs/{run_id}/render")
def render_selected(run_id: str, payload: RenderSelectionRequest, request: Request):
    settings = load_runtime_settings()
    selected_ids, selected_candidates, temp_overrides = apply_temp_overrides(
        run_id,
        payload.selected_ids or [],
        [],
        payload.temp_overrides,
        auto_include_additions=False,
    )
    if not selected_ids:
        return JSONResponse({"error": "selected_ids is empty"}, status_code=400)

    static_data = _load_static_data()
    rewritten_bullets = payload.rewritten_bullets or {}
    pdf_path, tex_path, report_path, final_selected_ids, final_candidates = process_and_render_artifacts(
        settings,
        run_id,
        static_data,
        selected_ids,
        selected_candidates,
        rewritten_bullets=rewritten_bullets if rewritten_bullets else None,
        temp_overrides=temp_overrides,
    )
    return {
        "status": "ok",
        "run_id": run_id,
        "pdf_url": f"/runs/{run_id}/pdf",
        "tex_url": f"/runs/{run_id}/tex",
        "report_url": f"/runs/{run_id}/report",
    }


@router.get("/runs/{run_id}/events")
def stream_run_events(run_id: str, request: Request):
    progress = get_or_create_progress(request.app, run_id)

    def event_stream():
        yield format_sse(progress.state)
        while True:
            try:
                event = progress.queue.get(timeout=10)
            except Empty:
                yield "event: ping\ndata: {}\n\n"
                continue
            yield format_sse(event)
            if event.get("status") in ("complete", "error"):
                break

    headers = {"Cache-Control": "no-cache", "Connection": "keep-alive"}
    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)


@router.get("/runs/{run_id}/pdf")
def get_pdf(run_id: str):
    settings = load_runtime_settings()
    path = os.path.join(settings.output_dir, f"{run_id}.pdf")
    if not os.path.exists(path):
        return JSONResponse({"error": "pdf not found"}, status_code=404)
    filename = normalize_output_pdf_name(settings.output_pdf_name) or "tailored_resume.pdf"
    headers = {"Content-Disposition": f'inline; filename="{filename}"'}
    return FileResponse(path, media_type="application/pdf", headers=headers)


@router.get("/runs/{run_id}/tex")
def get_tex(run_id: str):
    settings = load_runtime_settings()
    path = os.path.join(settings.output_dir, f"{run_id}.tex")
    if not os.path.exists(path):
        return JSONResponse({"error": "tex not found"}, status_code=404)
    return FileResponse(path, media_type="application/x-tex", filename="tailored_resume.tex")


@router.get("/runs/{run_id}/report")
def get_report(run_id: str):
    settings = load_runtime_settings()
    path = os.path.join(settings.output_dir, f"{run_id}_report.json")
    if not os.path.exists(path):
        return JSONResponse({"error": "report not found"}, status_code=404)
    return FileResponse(path, media_type="application/json", filename="resume_report.json")
