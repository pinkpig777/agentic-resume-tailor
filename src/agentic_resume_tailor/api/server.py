import copy
import json
import logging
import os
import subprocess
import threading
import time
from typing import Any, Dict, List, Tuple

import chromadb
import jinja2
import uvicorn
from chromadb.utils import embedding_functions
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pypdf import PdfReader
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session, selectinload

from agentic_resume_tailor.core.jd_utils import fallback_queries_from_jd, try_parse_jd
from agentic_resume_tailor.core.loop_controller import LoopConfig, run_loop
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
from agentic_resume_tailor.db.session import SessionLocal, get_db, init_db
from agentic_resume_tailor.db.sync import export_resume_data, write_resume_json
from agentic_resume_tailor.db.utils import (
    ensure_unique_slug,
    make_job_id,
    make_project_id,
    next_bullet_id,
    next_sort_order,
)
from agentic_resume_tailor.user_config import get_user_config_path, load_user_config, save_user_config
from agentic_resume_tailor.settings import get_settings
from agentic_resume_tailor.utils.logging import configure_logging

configure_logging()
logger = logging.getLogger(__name__)
settings = get_settings()

INGEST_LOCK = threading.Lock()
USER_CONFIG = load_user_config()

# -----------------------------
# Configuration (env-driven)
# -----------------------------
DB_PATH = settings.db_path
EXPORT_FILE = USER_CONFIG.get("export_file", settings.export_file)
TEMPLATE_DIR = settings.template_dir
OUTPUT_DIR = settings.output_dir
SKIP_PDF_RENDER = settings.skip_pdf

COLLECTION_NAME = settings.collection_name
EMBED_MODEL = settings.embed_model

DEFAULT_MAX_BULLETS = settings.max_bullets
DEFAULT_PER_QUERY_K = settings.per_query_k
DEFAULT_FINAL_K = settings.final_k

DEFAULT_MAX_ITERS = settings.max_iters
DEFAULT_THRESHOLD = settings.threshold
DEFAULT_ALPHA = settings.alpha
DEFAULT_MUST_WEIGHT = settings.must_weight

# set to "http://localhost:8501" if you want strict
CORS_ORIGINS = settings.cors_origins


# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI(title="AI Resume Agent API", version="0.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"]
    if CORS_ORIGINS.strip() == "*"
    else [o.strip() for o in CORS_ORIGINS.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    jd_text: str = Field(min_length=1)

    # Optional overrides
    max_bullets: int = Field(default=DEFAULT_MAX_BULLETS, ge=4, le=32)
    per_query_k: int = Field(default=DEFAULT_PER_QUERY_K, ge=1, le=50)
    final_k: int = Field(default=DEFAULT_FINAL_K, ge=5, le=200)

    max_iters: int = Field(default=DEFAULT_MAX_ITERS, ge=1, le=6)
    threshold: int = Field(default=DEFAULT_THRESHOLD, ge=0, le=100)

    alpha: float = Field(default=DEFAULT_ALPHA, ge=0.0, le=1.0)
    must_weight: float = Field(default=DEFAULT_MUST_WEIGHT, ge=0.0, le=1.0)

    # Boosting behavior
    boost_weight: float = Field(default=settings.boost_weight, ge=0.1, le=3.0)
    boost_top_n_missing: int = Field(default=settings.boost_top_n_missing, ge=1, le=20)


class GenerateResponse(BaseModel):
    run_id: str
    profile_used: bool
    best_iteration_index: int
    pdf_url: str
    tex_url: str
    report_url: str


class PersonalInfoUpdate(BaseModel):
    name: str | None = None
    phone: str | None = None
    email: str | None = None
    linkedin_id: str | None = None
    github_id: str | None = None
    linkedin: str | None = None
    github: str | None = None


class SkillsUpdate(BaseModel):
    languages_frameworks: str | None = None
    ai_ml: str | None = None
    db_tools: str | None = None


class ExperienceCreate(BaseModel):
    company: str = Field(min_length=1)
    role: str = Field(min_length=1)
    dates: str = ""
    location: str = ""
    sort_order: int | None = None
    bullets: List[str] = Field(default_factory=list)


class ExperienceUpdate(BaseModel):
    company: str | None = None
    role: str | None = None
    dates: str | None = None
    location: str | None = None
    sort_order: int | None = None


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1)
    technologies: str = ""
    sort_order: int | None = None
    bullets: List[str] = Field(default_factory=list)


class ProjectUpdate(BaseModel):
    name: str | None = None
    technologies: str | None = None
    sort_order: int | None = None


class BulletCreate(BaseModel):
    text_latex: str = Field(min_length=1)
    sort_order: int | None = None


class BulletUpdate(BaseModel):
    text_latex: str | None = None
    sort_order: int | None = None


class EducationCreate(BaseModel):
    school: str = Field(min_length=1)
    dates: str = ""
    degree: str = ""
    location: str = ""
    bullets: List[str] = Field(default_factory=list)
    sort_order: int | None = None


class EducationUpdate(BaseModel):
    school: str | None = None
    dates: str | None = None
    degree: str | None = None
    location: str | None = None
    bullets: List[str] | None = None
    sort_order: int | None = None


def _ensure_dirs() -> None:
    """Ensure output directories exist.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _load_static_data() -> Dict[str, Any]:
    """Load the resume snapshot from the SQL database.

    Returns:
        Dictionary result.
    """
    with SessionLocal() as db:
        return export_resume_data(db)


def _load_collection():
    """Load the Chroma collection and embedding function.
    """
    client = chromadb.PersistentClient(path=DB_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=ef)
    logger.info("Loaded Chroma collection '%s' (%s records)", COLLECTION_NAME, collection.count())
    return collection, ef


def _reload_static_data() -> Dict[str, Any]:
    """Refresh the in-memory resume snapshot.

    Returns:
        Dictionary result.
    """
    global STATIC_DATA
    STATIC_DATA = _load_static_data()
    return STATIC_DATA


def _reload_collection() -> None:
    """Refresh the in-memory Chroma collection.
    """
    global COLLECTION, EMB_FN
    COLLECTION, EMB_FN = _load_collection()


def _export_latest(db: Session) -> None:
    """Export the latest DB state to the resume JSON file.

    Args:
        db: Database session.
    """
    write_resume_json(db, EXPORT_FILE)


def _get_user_setting(key: str, default: Any) -> Any:
    """Fetch a user setting with a default fallback.

    Args:
        key: The key value.
        default: The default value.

    Returns:
        Result value.
    """
    return USER_CONFIG.get(key, default)


def _maybe_auto_reingest() -> None:
    """Auto re-ingest Chroma when enabled and idle.
    """
    if not _get_user_setting("auto_reingest_on_save", settings.auto_reingest_on_save):
        return
    if not INGEST_LOCK.acquire(blocking=False):
        logger.info("Auto re-ingest skipped; ingest already running.")
        return
    try:
        from agentic_resume_tailor import ingest as ingest_module

        ingest_module.ingest(json_path=EXPORT_FILE)
        _reload_collection()
    finally:
        INGEST_LOCK.release()


def _next_sort_order_for(query) -> int:
    """Compute the next sort order from a query result.

    Args:
        query: The query value.

    Returns:
        Integer result.
    """
    max_order = query.scalar()
    return next_sort_order([max_order])


def _experience_to_dict(exp: Experience) -> Dict[str, Any]:
    """Serialize an experience model for API responses.

    Args:
        exp: The exp value.

    Returns:
        Dictionary result.
    """
    bullets = sorted(exp.bullets, key=lambda b: (b.sort_order, b.id))
    return {
        "job_id": exp.job_id,
        "company": exp.company,
        "role": exp.role,
        "dates": exp.dates,
        "location": exp.location,
        "sort_order": exp.sort_order,
        "bullets": [
            {"id": b.local_id, "text_latex": b.text_latex, "sort_order": b.sort_order}
            for b in bullets
        ],
    }


def _project_to_dict(proj: Project) -> Dict[str, Any]:
    """Serialize a project model for API responses.

    Args:
        proj: The proj value.

    Returns:
        Dictionary result.
    """
    bullets = sorted(proj.bullets, key=lambda b: (b.sort_order, b.id))
    return {
        "project_id": proj.project_id,
        "name": proj.name,
        "technologies": proj.technologies,
        "sort_order": proj.sort_order,
        "bullets": [
            {"id": b.local_id, "text_latex": b.text_latex, "sort_order": b.sort_order}
            for b in bullets
        ],
    }


def _education_to_dict(edu: Education) -> Dict[str, Any]:
    """Serialize an education model for API responses.

    Args:
        edu: The edu value.

    Returns:
        Dictionary result.
    """
    bullets = sorted(edu.bullets, key=lambda b: (b.sort_order, b.id))
    return {
        "id": edu.id,
        "school": edu.school,
        "dates": edu.dates,
        "degree": edu.degree,
        "location": edu.location,
        "sort_order": edu.sort_order,
        "bullets": [b.text_latex for b in bullets],
    }


def _personal_info_to_dict(info: PersonalInfo | None) -> Dict[str, str]:
    """Serialize personal info for API responses.

    Args:
        info: The info value.

    Returns:
        Dictionary result.
    """
    return {
        "name": info.name if info else "",
        "phone": info.phone if info else "",
        "email": info.email if info else "",
        "linkedin_id": info.linkedin_id if info else "",
        "github_id": info.github_id if info else "",
        "linkedin": info.linkedin if info else "",
        "github": info.github if info else "",
    }


def _skills_to_dict(skills: Skills | None) -> Dict[str, str]:
    """Serialize skills for API responses.

    Args:
        skills: The skills value.

    Returns:
        Dictionary result.
    """
    return {
        "languages_frameworks": skills.languages_frameworks if skills else "",
        "ai_ml": skills.ai_ml if skills else "",
        "db_tools": skills.db_tools if skills else "",
    }


def select_and_rebuild(
    static_data: Dict[str, Any],
    selected_ids: List[str],
    selected_candidates: List[Any] | None = None,
) -> Dict[str, Any]:
    """Build a tailored resume snapshot with selected bullets only.

    Args:
        static_data: Exported resume data snapshot.
        selected_ids: Selected bullet identifiers.
        selected_candidates: Selected candidate bullets (optional).

    Returns:
        Dictionary result.
    """
    selected_set = set(selected_ids)
    tailored = copy.deepcopy(static_data)
    score_map: Dict[str, float] = {}
    for c in selected_candidates or []:
        score = getattr(c, "selection_score", None)
        if score is None:
            score = getattr(getattr(c, "best_hit", None), "weighted", 0.0)
        score_map[getattr(c, "bullet_id", "")] = float(score or 0.0)

    # Experiences
    new_exps = []
    for exp in tailored.get("experiences", []) or []:
        job_id = exp.get("job_id")
        kept_bullets: List[tuple[float, str, str]] = []
        for idx, b in enumerate(exp.get("bullets", []) or []):
            local_id = b.get("id")
            if not job_id or not local_id:
                continue
            bid = f"exp:{job_id}:{local_id}"
            if bid in selected_set:
                score = score_map.get(bid, 0.0)
                tie = local_id or f"idx:{idx:04d}"
                kept_bullets.append((score, tie, b.get("text_latex", "")))
        if kept_bullets:
            kept_bullets.sort(key=lambda item: (-item[0], item[1]))
            exp["bullets"] = [text for _, _, text in kept_bullets]
            new_exps.append(exp)

    # Projects
    new_projs = []
    for proj in tailored.get("projects", []) or []:
        project_id = proj.get("project_id")
        kept_bullets: List[tuple[float, str, str]] = []
        for idx, b in enumerate(proj.get("bullets", []) or []):
            local_id = b.get("id")
            if not project_id or not local_id:
                continue
            bid = f"proj:{project_id}:{local_id}"
            if bid in selected_set:
                score = score_map.get(bid, 0.0)
                tie = local_id or f"idx:{idx:04d}"
                kept_bullets.append((score, tie, b.get("text_latex", "")))
        if kept_bullets:
            kept_bullets.sort(key=lambda item: (-item[0], item[1]))
            proj["bullets"] = [text for _, _, text in kept_bullets]
            new_projs.append(proj)

    tailored["experiences"] = new_exps
    tailored["projects"] = new_projs
    return tailored


def render_pdf(context: Dict[str, Any], run_id: str) -> Tuple[str, str]:
    """Render a resume context to LaTeX/PDF artifacts.

    Args:
        context: The context value.
        run_id: Run identifier.

    Returns:
        Tuple of results.
    """
    _ensure_dirs()

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(TEMPLATE_DIR),
        block_start_string="((%",
        block_end_string="%))",
        variable_start_string="<<",
        variable_end_string=">>",
        comment_start_string="((#",
        comment_end_string="#))",
        autoescape=False,
    )

    local_template = os.path.join(TEMPLATE_DIR, "resume.local.tex")
    template_name = "resume.local.tex" if os.path.exists(local_template) else "resume.tex"
    template = env.get_template(template_name)
    tex_content = template.render(context)

    tex_path = os.path.join(OUTPUT_DIR, f"{run_id}.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex_content)

    if SKIP_PDF_RENDER:
        pdf_path = os.path.join(OUTPUT_DIR, f"{run_id}.pdf")
        with open(pdf_path, "wb") as f:
            f.write(b"")
        return pdf_path, tex_path

    try:
        subprocess.run(
            ["tectonic", tex_path, "--outdir", OUTPUT_DIR],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error("TECTONIC COMPILATION FAILED")
        logger.error("STDOUT (LaTeX Logs): %s", e.stdout)
        logger.error("STDERR: %s", e.stderr)
        raise

    pdf_path = os.path.join(OUTPUT_DIR, f"{run_id}.pdf")
    return pdf_path, tex_path


def _pdf_page_count(path: str) -> int | None:
    """Return the page count for a PDF path, if readable.

    Args:
        path: Filesystem path.

    Returns:
        Page count or None if unreadable.
    """
    try:
        reader = PdfReader(path)
        return len(reader.pages)
    except Exception as exc:
        logger.warning("Failed to read PDF page count: %s", exc)
        return None


def _trim_to_single_page(
    run_id: str,
    static_data: Dict[str, Any],
    selected_ids: List[str],
    selected_candidates: List[Any],
    pdf_path: str,
) -> Tuple[str, str, List[str], List[Any]]:
    """Trim lowest-weight bullets until the PDF is single-page.

    Args:
        run_id: Run identifier.
        static_data: Exported resume data snapshot.
        selected_ids: Selected bullet identifiers.
        selected_candidates: Selected candidate bullets.
        pdf_path: Current PDF path.

    Returns:
        Tuple of PDF path, TeX path, selected ids, selected candidates.
    """
    if SKIP_PDF_RENDER:
        tex_path = os.path.join(OUTPUT_DIR, f"{run_id}.tex")
        return pdf_path, tex_path, selected_ids, selected_candidates

    score_map: Dict[str, float] = {}
    for c in selected_candidates:
        score = getattr(c, "selection_score", None)
        if score is None:
            score = getattr(getattr(c, "best_hit", None), "weighted", 0.0)
        score_map[getattr(c, "bullet_id", "")] = float(score or 0.0)

    page_count = _pdf_page_count(pdf_path)
    while page_count is not None and page_count > 1 and len(selected_ids) > 1:
        ranked = [(score_map.get(bid, 0.0), bid) for bid in selected_ids]
        ranked.sort(key=lambda item: (item[0], item[1]))
        drop_id = ranked[0][1] if ranked else ""
        if not drop_id:
            break
        logger.info("Trimming bullet %s to enforce single-page PDF", drop_id)
        selected_ids = [bid for bid in selected_ids if bid != drop_id]
        selected_candidates = [
            c for c in selected_candidates if getattr(c, "bullet_id", "") != drop_id
        ]
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        tailored = select_and_rebuild(static_data, selected_ids, selected_candidates)
        pdf_path, tex_path = render_pdf(tailored, run_id)
        page_count = _pdf_page_count(pdf_path)

    if page_count is not None and page_count > 1:
        logger.warning("PDF still exceeds one page after trimming to %d bullets", len(selected_ids))

    tex_path = os.path.join(OUTPUT_DIR, f"{run_id}.tex")
    return pdf_path, tex_path, selected_ids, selected_candidates


def _run_id() -> str:
    """Generate a run id for artifacts.

    Returns:
        String result.
    """
    override = settings.run_id
    if override:
        return override
    # sortable and unique enough for local usage
    return time.strftime("%Y%m%d_%H%M%S") + "_" + str(int(time.time() * 1000))[-6:]


# -----------------------------
# Startup
# -----------------------------
logger.info("API Server starting: Initializing resume DB...")
init_db()
logger.info("API Server starting: Loading data + Chroma...")
STATIC_DATA = _load_static_data()
COLLECTION, EMB_FN = _load_collection()
logger.info("API Server ready.")


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    """Return API health metadata.
    """
    return {"status": "ok", "collection": COLLECTION_NAME, "embed_model": EMBED_MODEL}


@app.get("/settings")
def get_user_settings():
    """Return merged settings with user overrides.
    """
    base = settings.model_dump()
    base.pop("openai_api_key", None)
    merged = {**base, **USER_CONFIG}
    merged["config_path"] = get_user_config_path()
    return merged


@app.put("/settings")
def update_user_settings(payload: Dict[str, Any]):
    """Persist user settings updates to JSON.

    Args:
        payload: Request payload.
    """
    allowed = set(settings.model_fields.keys()) - {"openai_api_key"}
    updates = {k: v for k, v in (payload or {}).items() if k in allowed}
    if not updates:
        return get_user_settings()

    global USER_CONFIG, EXPORT_FILE
    USER_CONFIG = save_user_config(None, {**USER_CONFIG, **updates})
    if "export_file" in USER_CONFIG:
        EXPORT_FILE = USER_CONFIG["export_file"]
    return get_user_settings()


@app.get("/personal_info")
def get_personal_info(db: Session = Depends(get_db)):
    """Return the personal info record.

    Args:
        db: Database session (optional).
    """
    info = db.query(PersonalInfo).first()
    return _personal_info_to_dict(info)


@app.put("/personal_info")
def update_personal_info(payload: PersonalInfoUpdate, db: Session = Depends(get_db)):
    """Create or update the personal info record.

    Args:
        payload: Request payload.
        db: Database session (optional).
    """
    info = db.query(PersonalInfo).first()
    if info is None:
        info = PersonalInfo()
        db.add(info)
    for field in ("name", "phone", "email", "linkedin_id", "github_id", "linkedin", "github"):
        value = getattr(payload, field)
        if value is not None:
            setattr(info, field, value)
    db.commit()
    _export_latest(db)
    _maybe_auto_reingest()
    db.refresh(info)
    return _personal_info_to_dict(info)


@app.get("/skills")
def get_skills(db: Session = Depends(get_db)):
    """Return the skills record.

    Args:
        db: Database session (optional).
    """
    skills = db.query(Skills).first()
    return _skills_to_dict(skills)


@app.put("/skills")
def update_skills(payload: SkillsUpdate, db: Session = Depends(get_db)):
    """Create or update the skills record.

    Args:
        payload: Request payload.
        db: Database session (optional).
    """
    skills = db.query(Skills).first()
    if skills is None:
        skills = Skills()
        db.add(skills)
    for field in ("languages_frameworks", "ai_ml", "db_tools"):
        value = getattr(payload, field)
        if value is not None:
            setattr(skills, field, value)
    db.commit()
    _export_latest(db)
    _maybe_auto_reingest()
    db.refresh(skills)
    return _skills_to_dict(skills)


@app.get("/education")
def list_education(db: Session = Depends(get_db)):
    """List education entries.

    Args:
        db: Database session (optional).
    """
    educations = (
        db.query(Education)
        .options(selectinload(Education.bullets))
        .order_by(Education.sort_order.asc(), Education.id.asc())
        .all()
    )
    return [_education_to_dict(edu) for edu in educations]


@app.post("/education")
def create_education(payload: EducationCreate, db: Session = Depends(get_db)):
    """Create a new education entry.

    Args:
        payload: Request payload.
        db: Database session (optional).
    """
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
        if not bullet:
            continue
        db.add(EducationBullet(education_id=edu.id, text_latex=bullet, sort_order=idx))

    db.commit()
    _export_latest(db)
    _maybe_auto_reingest()
    edu = (
        db.query(Education)
        .options(selectinload(Education.bullets))
        .filter(Education.id == edu.id)
        .first()
    )
    return _education_to_dict(edu)


@app.put("/education/{education_id}")
def update_education(education_id: int, payload: EducationUpdate, db: Session = Depends(get_db)):
    """Update an education entry.

    Args:
        education_id: Education identifier.
        payload: Request payload.
        db: Database session (optional).
    """
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
            if not bullet:
                continue
            db.add(EducationBullet(education_id=edu.id, text_latex=bullet, sort_order=idx))

    db.commit()
    _export_latest(db)
    _maybe_auto_reingest()
    edu = (
        db.query(Education)
        .options(selectinload(Education.bullets))
        .filter(Education.id == edu.id)
        .first()
    )
    return _education_to_dict(edu)


@app.delete("/education/{education_id}")
def delete_education(education_id: int, db: Session = Depends(get_db)):
    """Delete an education entry.

    Args:
        education_id: Education identifier.
        db: Database session (optional).
    """
    edu = db.query(Education).filter(Education.id == education_id).first()
    if edu is None:
        raise HTTPException(status_code=404, detail="Education entry not found")
    db.delete(edu)
    db.commit()
    _export_latest(db)
    _maybe_auto_reingest()
    return {"status": "deleted", "id": education_id}


@app.get("/experiences")
def list_experiences(db: Session = Depends(get_db)):
    """List experience entries.

    Args:
        db: Database session (optional).
    """
    experiences = (
        db.query(Experience)
        .options(selectinload(Experience.bullets))
        .order_by(Experience.sort_order.asc(), Experience.id.asc())
        .all()
    )
    return [_experience_to_dict(exp) for exp in experiences]


@app.get("/experiences/{job_id}")
def get_experience(job_id: str, db: Session = Depends(get_db)):
    """Return a single experience by job_id.

    Args:
        job_id: Job identifier.
        db: Database session (optional).
    """
    exp = (
        db.query(Experience)
        .options(selectinload(Experience.bullets))
        .filter(Experience.job_id == job_id)
        .first()
    )
    if exp is None:
        raise HTTPException(status_code=404, detail="Experience not found")
    return _experience_to_dict(exp)


@app.post("/experiences")
def create_experience(payload: ExperienceCreate, db: Session = Depends(get_db)):
    """Create a new experience entry.

    Args:
        payload: Request payload.
        db: Database session (optional).
    """
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

    existing_ids: List[str] = []
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
    _export_latest(db)
    _maybe_auto_reingest()
    exp = (
        db.query(Experience)
        .options(selectinload(Experience.bullets))
        .filter(Experience.id == exp.id)
        .first()
    )
    return _experience_to_dict(exp)


@app.put("/experiences/{job_id}")
def update_experience(job_id: str, payload: ExperienceUpdate, db: Session = Depends(get_db)):
    """Update an experience entry.

    Args:
        job_id: Job identifier.
        payload: Request payload.
        db: Database session (optional).
    """
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
    _export_latest(db)
    _maybe_auto_reingest()
    exp = (
        db.query(Experience)
        .options(selectinload(Experience.bullets))
        .filter(Experience.id == exp_id)
        .first()
    )
    return _experience_to_dict(exp)


@app.delete("/experiences/{job_id}")
def delete_experience(job_id: str, db: Session = Depends(get_db)):
    """Delete an experience entry.

    Args:
        job_id: Job identifier.
        db: Database session (optional).
    """
    exp = db.query(Experience).filter(Experience.job_id == job_id).first()
    if exp is None:
        raise HTTPException(status_code=404, detail="Experience not found")
    db.delete(exp)
    db.commit()
    _export_latest(db)
    _maybe_auto_reingest()
    return {"status": "deleted", "job_id": job_id}


@app.get("/experiences/{job_id}/bullets")
def list_experience_bullets(job_id: str, db: Session = Depends(get_db)):
    """List bullets for an experience.

    Args:
        job_id: Job identifier.
        db: Database session (optional).
    """
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
        {"id": b.local_id, "text_latex": b.text_latex, "sort_order": b.sort_order}
        for b in bullets
    ]


@app.post("/experiences/{job_id}/bullets")
def create_experience_bullet(job_id: str, payload: BulletCreate, db: Session = Depends(get_db)):
    """Create a bullet under an experience.

    Args:
        job_id: Job identifier.
        payload: Request payload.
        db: Database session (optional).
    """
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
    _export_latest(db)
    _maybe_auto_reingest()
    return {"id": local_id, "text_latex": bullet.text_latex, "sort_order": bullet.sort_order}


@app.put("/experiences/{job_id}/bullets/{local_id}")
def update_experience_bullet(
    job_id: str, local_id: str, payload: BulletUpdate, db: Session = Depends(get_db)
):
    """Update a bullet under an experience.

    Args:
        job_id: Job identifier.
        local_id: Local bullet identifier.
        payload: Request payload.
        db: Database session (optional).
    """
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
    _export_latest(db)
    _maybe_auto_reingest()
    return {"id": bullet.local_id, "text_latex": bullet.text_latex, "sort_order": bullet.sort_order}


@app.delete("/experiences/{job_id}/bullets/{local_id}")
def delete_experience_bullet(job_id: str, local_id: str, db: Session = Depends(get_db)):
    """Delete a bullet under an experience.

    Args:
        job_id: Job identifier.
        local_id: Local bullet identifier.
        db: Database session (optional).
    """
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
    _export_latest(db)
    _maybe_auto_reingest()
    return {"status": "deleted", "id": local_id}


@app.get("/projects")
def list_projects(db: Session = Depends(get_db)):
    """List project entries.

    Args:
        db: Database session (optional).
    """
    projects = (
        db.query(Project)
        .options(selectinload(Project.bullets))
        .order_by(Project.sort_order.asc(), Project.id.asc())
        .all()
    )
    return [_project_to_dict(proj) for proj in projects]


@app.get("/projects/{project_id}")
def get_project(project_id: str, db: Session = Depends(get_db)):
    """Return a single project by project_id.

    Args:
        project_id: Project identifier.
        db: Database session (optional).
    """
    proj = (
        db.query(Project)
        .options(selectinload(Project.bullets))
        .filter(Project.project_id == project_id)
        .first()
    )
    if proj is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return _project_to_dict(proj)


@app.post("/projects")
def create_project(payload: ProjectCreate, db: Session = Depends(get_db)):
    """Create a new project entry.

    Args:
        payload: Request payload.
        db: Database session (optional).
    """
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
    _export_latest(db)
    _maybe_auto_reingest()
    proj = (
        db.query(Project)
        .options(selectinload(Project.bullets))
        .filter(Project.id == proj.id)
        .first()
    )
    return _project_to_dict(proj)


@app.put("/projects/{project_id}")
def update_project(project_id: str, payload: ProjectUpdate, db: Session = Depends(get_db)):
    """Update a project entry.

    Args:
        project_id: Project identifier.
        payload: Request payload.
        db: Database session (optional).
    """
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
                row[0]
                for row in db.query(Project.project_id)
                .filter(Project.id != proj.id)
                .all()
            ]
            proj.project_id = ensure_unique_slug(new_base, existing_ids)

    db.commit()
    _export_latest(db)
    _maybe_auto_reingest()
    proj = (
        db.query(Project)
        .options(selectinload(Project.bullets))
        .filter(Project.id == proj_id)
        .first()
    )
    return _project_to_dict(proj)


@app.delete("/projects/{project_id}")
def delete_project(project_id: str, db: Session = Depends(get_db)):
    """Delete a project entry.

    Args:
        project_id: Project identifier.
        db: Database session (optional).
    """
    proj = db.query(Project).filter(Project.project_id == project_id).first()
    if proj is None:
        raise HTTPException(status_code=404, detail="Project not found")
    db.delete(proj)
    db.commit()
    _export_latest(db)
    _maybe_auto_reingest()
    return {"status": "deleted", "project_id": project_id}


@app.get("/projects/{project_id}/bullets")
def list_project_bullets(project_id: str, db: Session = Depends(get_db)):
    """List bullets for a project.

    Args:
        project_id: Project identifier.
        db: Database session (optional).
    """
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
        {"id": b.local_id, "text_latex": b.text_latex, "sort_order": b.sort_order}
        for b in bullets
    ]


@app.post("/projects/{project_id}/bullets")
def create_project_bullet(project_id: str, payload: BulletCreate, db: Session = Depends(get_db)):
    """Create a bullet under a project.

    Args:
        project_id: Project identifier.
        payload: Request payload.
        db: Database session (optional).
    """
    proj = db.query(Project).filter(Project.project_id == project_id).first()
    if proj is None:
        raise HTTPException(status_code=404, detail="Project not found")

    existing_ids = [
        row[0]
        for row in db.query(ProjectBullet.local_id)
        .filter(ProjectBullet.project_id == proj.id)
        .all()
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
    _export_latest(db)
    _maybe_auto_reingest()
    return {"id": local_id, "text_latex": bullet.text_latex, "sort_order": bullet.sort_order}


@app.put("/projects/{project_id}/bullets/{local_id}")
def update_project_bullet(
    project_id: str, local_id: str, payload: BulletUpdate, db: Session = Depends(get_db)
):
    """Update a bullet under a project.

    Args:
        project_id: Project identifier.
        local_id: Local bullet identifier.
        payload: Request payload.
        db: Database session (optional).
    """
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
    _export_latest(db)
    _maybe_auto_reingest()
    return {"id": bullet.local_id, "text_latex": bullet.text_latex, "sort_order": bullet.sort_order}


@app.delete("/projects/{project_id}/bullets/{local_id}")
def delete_project_bullet(project_id: str, local_id: str, db: Session = Depends(get_db)):
    """Delete a bullet under a project.

    Args:
        project_id: Project identifier.
        local_id: Local bullet identifier.
        db: Database session (optional).
    """
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
    _export_latest(db)
    _maybe_auto_reingest()
    return {"status": "deleted", "id": local_id}


@app.post("/admin/export")
def export_resume(reingest: bool = False, db: Session = Depends(get_db)):
    """Export DB to JSON and optionally re-ingest Chroma.

    Args:
        reingest: The reingest value (optional).
        db: Database session (optional).
    """
    write_resume_json(db, EXPORT_FILE)
    _reload_static_data()

    reingested = False
    if reingest:
        from agentic_resume_tailor import ingest as ingest_module

        ingest_module.ingest(json_path=EXPORT_FILE)
        _reload_collection()
        reingested = True

    return {"status": "ok", "path": EXPORT_FILE, "reingested": reingested}


@app.post("/admin/ingest")
def ingest_resume(db: Session = Depends(get_db)):
    """Ingest the exported resume JSON into Chroma.

    Args:
        db: Database session (optional).
    """
    if not INGEST_LOCK.acquire(blocking=False):
        return JSONResponse(
            {"status": "error", "count": 0, "elapsed_s": 0.0, "error": "ingest already running"},
            status_code=409,
        )

    start = time.time()
    try:
        write_resume_json(db, EXPORT_FILE)
        from agentic_resume_tailor import ingest as ingest_module

        count = ingest_module.ingest(json_path=EXPORT_FILE)
        _reload_collection()
        _reload_static_data()
        elapsed = time.time() - start
        return {"status": "ok", "count": count, "elapsed_s": round(elapsed, 2)}
    except Exception as exc:
        logger.exception("Chroma ingest failed")
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
        INGEST_LOCK.release()


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    """Run the resume generation pipeline.

    Args:
        req: Request payload.

    Returns:
        Result value.
    """
    jd_text = (req.jd_text or "").strip()
    if not jd_text:
        return JSONResponse({"error": "jd_text is empty"}, status_code=400)

    run_id = _run_id()
    static_data = _load_static_data()

    profile = try_parse_jd(jd_text)
    base_profile_or_queries = profile if profile is not None else fallback_queries_from_jd(jd_text)

    cfg = LoopConfig(
        max_iters=req.max_iters,
        threshold=req.threshold,
        per_query_k=req.per_query_k,
        final_k=req.final_k,
        max_bullets=req.max_bullets,
        alpha=req.alpha,
        must_weight=req.must_weight,
        boost_weight=req.boost_weight,
        boost_top_n_missing=req.boost_top_n_missing,
    )

    loop = run_loop(
        jd_text=jd_text,
        static_data=static_data,
        collection=COLLECTION,
        embedding_fn=EMB_FN,
        base_profile_or_queries=base_profile_or_queries,
        cfg=cfg,
    )

    # Build final resume and render artifacts from best iteration
    selected_ids = list(loop.best_selected_ids)
    selected_candidates = list(loop.best_selected_candidates or [])
    tailored_data = select_and_rebuild(static_data, selected_ids, selected_candidates)
    pdf_path, tex_path = render_pdf(tailored_data, run_id)
    pdf_path, tex_path, selected_ids, selected_candidates = _trim_to_single_page(
        run_id,
        static_data,
        selected_ids,
        selected_candidates,
        pdf_path,
    )

    report = {
        "run_id": run_id,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "profile_used": profile is not None,
        "loop_config": cfg.__dict__,
        "best_iteration_index": loop.best_iteration_index,
        "selected_ids": selected_ids,
        "iterations": loop.iterations,
        "artifacts": {
            "pdf": os.path.basename(pdf_path),
            "tex": os.path.basename(tex_path),
        },
    }

    # Attach best score summary if available
    if loop.best_hybrid is not None:
        report["best_score"] = {
            "final_score": loop.best_hybrid.final_score,
            "retrieval_score": loop.best_hybrid.retrieval_score,
            "coverage_bullets_only": loop.best_hybrid.coverage_bullets_only,
            "coverage_all": loop.best_hybrid.coverage_all,
            "must_missing_bullets_only": loop.best_hybrid.must_missing_bullets_only,
            "nice_missing_bullets_only": loop.best_hybrid.nice_missing_bullets_only,
            "must_missing_all": loop.best_hybrid.must_missing_all,
            "nice_missing_all": loop.best_hybrid.nice_missing_all,
        }

    report_path = os.path.join(OUTPUT_DIR, f"{run_id}_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return GenerateResponse(
        run_id=run_id,
        profile_used=profile is not None,
        best_iteration_index=loop.best_iteration_index,
        pdf_url=f"/runs/{run_id}/pdf",
        tex_url=f"/runs/{run_id}/tex",
        report_url=f"/runs/{run_id}/report",
    )


@app.get("/runs/{run_id}/pdf")
def get_pdf(run_id: str):
    """Serve a rendered PDF artifact.

    Args:
        run_id: Run identifier.
    """
    path = os.path.join(OUTPUT_DIR, f"{run_id}.pdf")
    if not os.path.exists(path):
        return JSONResponse({"error": "pdf not found"}, status_code=404)
    return FileResponse(path, media_type="application/pdf", filename="tailored_resume.pdf")


@app.get("/runs/{run_id}/tex")
def get_tex(run_id: str):
    """Serve a rendered TeX artifact.

    Args:
        run_id: Run identifier.
    """
    path = os.path.join(OUTPUT_DIR, f"{run_id}.tex")
    if not os.path.exists(path):
        return JSONResponse({"error": "tex not found"}, status_code=404)
    return FileResponse(path, media_type="application/x-tex", filename="tailored_resume.tex")


@app.get("/runs/{run_id}/report")
def get_report(run_id: str):
    """Serve a run report artifact.

    Args:
        run_id: Run identifier.
    """
    path = os.path.join(OUTPUT_DIR, f"{run_id}_report.json")
    if not os.path.exists(path):
        return JSONResponse({"error": "report not found"}, status_code=404)
    return FileResponse(path, media_type="application/json", filename="resume_report.json")


def main() -> None:
    """Run the API server entrypoint.
    """
    uvicorn.run(app, host="0.0.0.0", port=settings.port)


if __name__ == "__main__":
    main()
