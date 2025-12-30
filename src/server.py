import copy
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import jinja2
import uvicorn

from retrieval import multi_query_retrieve
from selection import select_topk
from loop_controller import LoopConfig, run_loop
from keyword_matcher import extract_profile_keywords


# -----------------------------
# Configuration (env-driven)
# -----------------------------
DB_PATH = os.environ.get("ART_DB_PATH", "/app/data/processed/chroma_db")
DATA_FILE = os.environ.get("ART_DATA_FILE", "/app/data/my_experience.json")
TEMPLATE_DIR = os.environ.get("ART_TEMPLATE_DIR", "/app/templates")
OUTPUT_DIR = os.environ.get("ART_OUTPUT_DIR", "/app/output")

COLLECTION_NAME = os.environ.get("ART_COLLECTION", "resume_experience")
EMBED_MODEL = os.environ.get("ART_EMBED_MODEL", "BAAI/bge-small-en-v1.5")

USE_JD_PARSER = os.environ.get("ART_USE_JD_PARSER", "1") == "1"

DEFAULT_MAX_BULLETS = int(os.environ.get("ART_MAX_BULLETS", "16"))
DEFAULT_PER_QUERY_K = int(os.environ.get("ART_PER_QUERY_K", "10"))
DEFAULT_FINAL_K = int(os.environ.get("ART_FINAL_K", "30"))

DEFAULT_MAX_ITERS = int(os.environ.get("ART_MAX_ITERS", "3"))
DEFAULT_THRESHOLD = int(os.environ.get("ART_SCORE_THRESHOLD", "80"))
DEFAULT_ALPHA = float(os.environ.get("ART_SCORE_ALPHA", "0.7"))
DEFAULT_MUST_WEIGHT = float(os.environ.get("ART_MUST_WEIGHT", "0.8"))

# set to "http://localhost:8501" if you want strict
CORS_ORIGINS = os.environ.get("ART_CORS_ORIGINS", "*")


# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI(title="AI Resume Agent API", version="0.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if CORS_ORIGINS.strip() == "*" else [o.strip()
                                                             for o in CORS_ORIGINS.split(",") if o.strip()],
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
    boost_weight: float = Field(default=float(
        os.environ.get("ART_BOOST_WEIGHT", "1.6")), ge=0.1, le=3.0)
    boost_top_n_missing: int = Field(default=int(
        os.environ.get("ART_BOOST_TOP_N", "6")), ge=1, le=20)


def _ensure_dirs() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _load_static_data() -> Dict[str, Any]:
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL)
    collection = client.get_collection(
        name=COLLECTION_NAME, embedding_function=ef)
    print(
        f"✅ Loaded Chroma collection '{COLLECTION_NAME}' ({collection.count()} records)")
    return collection, ef


def try_parse_jd(jd_text: str):
    """
    Optional Node 1 parser (OpenAI). If it fails, we fall back to local multi-query.
    """
    if not USE_JD_PARSER:
        return None

    try:
        import jd_parser  # src/jd_parser.py
        if not hasattr(jd_parser, "parse_job_description"):
            raise RuntimeError("jd_parser.parse_job_description not found")

        model = os.environ.get("ART_JD_MODEL", "gpt-4.1-nano-2025-04-14")
        try:
            return jd_parser.parse_job_description(jd_text, model=model)
        except TypeError:
            return jd_parser.parse_job_description(jd_text)

    except Exception as e:
        print(f"⚠️ JD parser failed. Reason: {e}")
        return None


def fallback_queries_from_jd(jd_text: str, max_q: int = 6) -> List[str]:
    """
    Minimal heuristic fallback.
    Produces embedding-friendly queries from bullet lines + a condensed full query.
    """
    lines = [ln.strip() for ln in jd_text.splitlines() if ln.strip()]
    bulletish = [ln.lstrip("-•* ").strip()
                 for ln in lines if ln.strip().startswith(("-", "•", "*"))]

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


def select_and_rebuild(static_data: Dict[str, Any], selected_ids: List[str]) -> Dict[str, Any]:
    """
    Rebuild resume data from my_experience.json, keeping ONLY bullets whose deterministic ids survive.
    Convert bullets to list[str] of text_latex because the LaTeX template expects strings.
    """
    selected_set = set(selected_ids)
    tailored = copy.deepcopy(static_data)

    # Experiences
    new_exps = []
    for exp in tailored.get("experiences", []) or []:
        job_id = exp.get("job_id")
        kept_texts: List[str] = []
        for b in exp.get("bullets", []) or []:
            local_id = b.get("id")
            if not job_id or not local_id:
                continue
            bid = f"exp:{job_id}:{local_id}"
            if bid in selected_set:
                kept_texts.append(b.get("text_latex", ""))
        if kept_texts:
            exp["bullets"] = kept_texts
            new_exps.append(exp)

    # Projects
    new_projs = []
    for proj in tailored.get("projects", []) or []:
        project_id = proj.get("project_id")
        kept_texts: List[str] = []
        for b in proj.get("bullets", []) or []:
            local_id = b.get("id")
            if not project_id or not local_id:
                continue
            bid = f"proj:{project_id}:{local_id}"
            if bid in selected_set:
                kept_texts.append(b.get("text_latex", ""))
        if kept_texts:
            proj["bullets"] = kept_texts
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

    try:
        subprocess.run(
            ["tectonic", tex_path, "--outdir", OUTPUT_DIR],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 40, file=sys.stderr)
        print("❌ TECTONIC COMPILATION FAILED", file=sys.stderr)
        print("=" * 40, file=sys.stderr)
        print("STDOUT (LaTeX Logs):", file=sys.stderr)
        print(e.stdout, file=sys.stderr)
        print("-" * 20, file=sys.stderr)
        print("STDERR:", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        print("=" * 40, file=sys.stderr)
        raise

    pdf_path = os.path.join(OUTPUT_DIR, f"{run_id}.pdf")
    return pdf_path, tex_path


def _run_id() -> str:
    # sortable and unique enough for local usage
    return time.strftime("%Y%m%d_%H%M%S") + "_" + str(int(time.time() * 1000))[-6:]


# -----------------------------
# Startup
# -----------------------------
print("⚙️ API Server starting: Loading data + Chroma...")
STATIC_DATA = _load_static_data()
COLLECTION, EMB_FN = _load_collection()
print("✅ API Server ready.")


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok", "collection": COLLECTION_NAME, "embed_model": EMBED_MODEL}


@app.post("/generate")
async def generate(req: GenerateRequest):
    jd_text = (req.jd_text or "").strip()
    if not jd_text:
        return JSONResponse({"error": "jd_text is empty"}, status_code=400)

    run_id = _run_id()

    profile = try_parse_jd(jd_text)
    base_profile_or_queries = profile if profile is not None else fallback_queries_from_jd(
        jd_text)

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
    tailored_data = select_and_rebuild(STATIC_DATA, loop.best_selected_ids)
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

    return {
        "run_id": run_id,
        "profile_used": profile is not None,
        "best_iteration_index": loop.best_iteration_index,
        "pdf_url": f"/runs/{run_id}/pdf",
        "tex_url": f"/runs/{run_id}/tex",
        "report_url": f"/runs/{run_id}/report",
    }


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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
