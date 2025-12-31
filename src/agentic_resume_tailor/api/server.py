import copy
import json
import logging
import os
import subprocess
import time
from typing import Any, Dict, List, Tuple

import chromadb
import jinja2
import uvicorn
from chromadb.utils import embedding_functions
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from agentic_resume_tailor.core.loop_controller import LoopConfig, run_loop
from agentic_resume_tailor.settings import get_settings
from agentic_resume_tailor.utils.logging import configure_logging

configure_logging()
logger = logging.getLogger(__name__)
settings = get_settings()

# -----------------------------
# Configuration (env-driven)
# -----------------------------
DB_PATH = settings.db_path
DATA_FILE = settings.data_file
TEMPLATE_DIR = settings.template_dir
OUTPUT_DIR = settings.output_dir
SKIP_PDF_RENDER = settings.skip_pdf

COLLECTION_NAME = settings.collection_name
EMBED_MODEL = settings.embed_model

USE_JD_PARSER = settings.use_jd_parser

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


def _ensure_dirs() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _load_static_data() -> Dict[str, Any]:
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=ef)
    logger.info("Loaded Chroma collection '%s' (%s records)", COLLECTION_NAME, collection.count())
    return collection, ef


def try_parse_jd(jd_text: str):
    """
    Optional Node 1 parser (OpenAI). If it fails, we fall back to local multi-query.
    """
    if not USE_JD_PARSER:
        return None

    try:
        from agentic_resume_tailor import jd_parser

        if not hasattr(jd_parser, "parse_job_description"):
            raise RuntimeError("jd_parser.parse_job_description not found")

        model = settings.jd_model
        try:
            return jd_parser.parse_job_description(jd_text, model=model)
        except TypeError:
            return jd_parser.parse_job_description(jd_text)

    except Exception as e:
        logger.warning("JD parser failed. Reason: %s", e)
        return None


def fallback_queries_from_jd(jd_text: str, max_q: int = 6) -> List[str]:
    """
    Minimal heuristic fallback.
    Produces embedding-friendly queries from bullet lines + a condensed full query.
    """
    lines = [ln.strip() for ln in jd_text.splitlines() if ln.strip()]
    bulletish = [
        ln.lstrip("-•* ").strip() for ln in lines if ln.strip().startswith(("-", "•", "*"))
    ]

    out: List[str] = []
    for b in bulletish:
        if len(b) >= 12:
            out.append(b)

    condensed = " ".join(lines[:20])
    condensed = " ".join(condensed.split())
    if condensed and condensed not in out:
        out.insert(0, condensed)

    # de-dupe keep order
    seen = set()
    deduped: List[str] = []
    for q in out:
        qn = q.lower()
        if qn not in seen:
            seen.add(qn)
            deduped.append(q)
        if len(deduped) >= max_q:
            break

    return deduped[:max_q] if deduped else [jd_text.strip()]


def select_and_rebuild(
    static_data: Dict[str, Any],
    selected_ids: List[str],
    selected_candidates: List[Any] | None = None,
) -> Dict[str, Any]:
    """
    Rebuild resume data from my_experience.json, keeping ONLY bullets whose deterministic ids survive.
    Convert bullets to list[str] of text_latex because the LaTeX template expects strings.
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
    """
    Render the tailored resume to PDF using the LaTeX template.
    Returns (pdf_path, tex_path).
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

    template = env.get_template("resume.tex")
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


def _run_id() -> str:
    override = settings.run_id
    if override:
        return override
    # sortable and unique enough for local usage
    return time.strftime("%Y%m%d_%H%M%S") + "_" + str(int(time.time() * 1000))[-6:]


# -----------------------------
# Startup
# -----------------------------
logger.info("API Server starting: Loading data + Chroma...")
STATIC_DATA = _load_static_data()
COLLECTION, EMB_FN = _load_collection()
logger.info("API Server ready.")


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok", "collection": COLLECTION_NAME, "embed_model": EMBED_MODEL}


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    jd_text = (req.jd_text or "").strip()
    if not jd_text:
        return JSONResponse({"error": "jd_text is empty"}, status_code=400)

    run_id = _run_id()

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
        static_data=STATIC_DATA,
        collection=COLLECTION,
        embedding_fn=EMB_FN,
        base_profile_or_queries=base_profile_or_queries,
        cfg=cfg,
    )

    # Build final resume and render artifacts from best iteration
    tailored_data = select_and_rebuild(
        STATIC_DATA,
        loop.best_selected_ids,
        loop.best_selected_candidates,
    )
    pdf_path, tex_path = render_pdf(tailored_data, run_id)

    report = {
        "run_id": run_id,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "profile_used": profile is not None,
        "loop_config": cfg.__dict__,
        "best_iteration_index": loop.best_iteration_index,
        "selected_ids": loop.best_selected_ids,
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
    path = os.path.join(OUTPUT_DIR, f"{run_id}.pdf")
    if not os.path.exists(path):
        return JSONResponse({"error": "pdf not found"}, status_code=404)
    return FileResponse(path, media_type="application/pdf", filename="tailored_resume.pdf")


@app.get("/runs/{run_id}/tex")
def get_tex(run_id: str):
    path = os.path.join(OUTPUT_DIR, f"{run_id}.tex")
    if not os.path.exists(path):
        return JSONResponse({"error": "tex not found"}, status_code=404)
    return FileResponse(path, media_type="application/x-tex", filename="tailored_resume.tex")


@app.get("/runs/{run_id}/report")
def get_report(run_id: str):
    path = os.path.join(OUTPUT_DIR, f"{run_id}_report.json")
    if not os.path.exists(path):
        return JSONResponse({"error": "report not found"}, status_code=404)
    return FileResponse(path, media_type="application/json", filename="resume_report.json")


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=settings.port)


if __name__ == "__main__":
    main()
