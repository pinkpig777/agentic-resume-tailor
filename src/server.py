import copy
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse
import jinja2
import uvicorn

from retrieval import multi_query_retrieve
from selection import select_topk
from keyword_matcher import extract_profile_keywords, match_keywords_against_bullets
from scorer import score as hybrid_score


# --- CONFIGURATION ---
MAX_BULLETS_ON_PAGE = int(os.environ.get("ART_MAX_BULLETS", "16"))

DB_PATH = os.environ.get("ART_DB_PATH", "/app/data/processed/chroma_db")
DATA_FILE = os.environ.get("ART_DATA_FILE", "/app/data/my_experience.json")
TEMPLATE_DIR = os.environ.get("ART_TEMPLATE_DIR", "/app/templates")
OUTPUT_DIR = os.environ.get("ART_OUTPUT_DIR", "/app/output")

COLLECTION_NAME = os.environ.get("ART_COLLECTION", "resume_experience")
EMBED_MODEL = os.environ.get("ART_EMBED_MODEL", "BAAI/bge-small-en-v1.5")

USE_JD_PARSER = os.environ.get("ART_USE_JD_PARSER", "1") == "1"


app = FastAPI()


class Timer:
    def __init__(self, name: str):
        self.name = name
        self.start = time.time()

    def stop(self):
        elapsed = time.time() - self.start
        print(f"⏱️  [TIME] {self.name}: {elapsed:.2f}s")
        return elapsed


def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_static_data() -> Dict[str, Any]:
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL)
    collection = client.get_collection(
        name=COLLECTION_NAME, embedding_function=ef)
    print(
        f"✅ Loaded Chroma collection '{COLLECTION_NAME}' ({collection.count()} records)")
    return collection, ef


def try_parse_jd(jd_text: str):
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


def build_skills_pseudo_bullet(static_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Skills are NOT embedded in Chroma.
    We add a pseudo bullet for coverage_all computations only (scoring/explainability).
    """
    skills = static_data.get("skills", {}) or {}
    parts = []
    for k in ["languages_frameworks", "ai_ml", "db_tools"]:
        v = skills.get(k)
        if v:
            parts.append(str(v))
    txt = " | ".join(parts).strip()
    if not txt:
        return None
    return {"bullet_id": "__skills__", "text_latex": txt, "meta": {"section": "skills"}}


def render_pdf(context: Dict[str, Any]) -> str:
    """
    Render the tailored resume to PDF using the LaTeX template.
    Returns output PDF path.
    """
    ensure_dirs()
    t_render = Timer("Render & Compile PDF")

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

    try:
        print("📝 Rendering LaTeX template...")
        template = env.get_template("resume.tex")
        tex_content = template.render(context)

        tex_path = os.path.join(OUTPUT_DIR, "tailored_resume.tex")
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(tex_content)

    except Exception as e:
        print(f"❌ Template render failed: {e}", file=sys.stderr)
        raise

    try:
        print(f"📄 Compiling PDF via Tectonic: {tex_path}")
        subprocess.run(
            ["tectonic", tex_path, "--outdir", OUTPUT_DIR],
            check=True,
            capture_output=True,
            text=True,
        )
        pdf_path = os.path.join(OUTPUT_DIR, "tailored_resume.pdf")
        print("✅ PDF Compiled Successfully!")
        t_render.stop()
        return pdf_path

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

    except FileNotFoundError:
        print("❌ 'tectonic' not found in PATH.", file=sys.stderr)
        raise


def select_and_rebuild(static_data: Dict[str, Any], selected_ids: List[str]) -> Dict[str, Any]:
    """
    Rebuild resume data from my_experience.json, keeping ONLY bullets whose deterministic ids survive.
    Convert bullets to list[str] of text_latex because your LaTeX template expects strings.
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


# --- Startup ---
print("⚙️ Server Starting: Loading data + Chroma...")
STATIC_DATA = load_static_data()
COLLECTION, EMB_FN = load_collection()
print("✅ Server Ready.")


@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
      <head>
        <title>Resume Agent</title>
        <style>
          body { font-family: -apple-system, system-ui, sans-serif; max-width: 900px; margin: 40px auto; padding: 20px; line-height: 1.6; }
          textarea { width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 6px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
          button { background: #2563eb; color: white; border: none; padding: 12px 24px; border-radius: 6px; cursor: pointer; font-size: 16px; margin-top: 12px;}
          button:hover { background: #1d4ed8; }
          .hint { color: #555; font-size: 14px; }
        </style>
      </head>
      <body>
        <h1>🚀 AI Resume Tailor</h1>
        <p class="hint">Paste a Job Description. This uses local Chroma retrieval + 16-bullet selection. Optional JD parser if OPENAI_API_KEY is set.</p>
        <form action="/generate" method="post">
          <textarea name="jd_text" rows="18" placeholder="Paste Job Description here..."></textarea>
          <br />
          <button type="submit">Generate PDF</button>
        </form>
      </body>
    </html>
    """


@app.post("/generate")
async def generate_resume(request: Request):
    ensure_dirs()
    form_data = await request.form()
    jd_text = (form_data.get("jd_text") or "").strip()
    if not jd_text:
        return HTMLResponse("jd_text is empty", status_code=400)

    t_all = Timer("Total /generate")

    # Node 1 (optional)
    profile = try_parse_jd(jd_text)

    # Node 2 retrieval
    jd_parser_result = profile if profile is not None else fallback_queries_from_jd(
        jd_text)
    cands = multi_query_retrieve(
        collection=COLLECTION,
        embedding_fn=EMB_FN,
        jd_parser_result=jd_parser_result,
        per_query_k=int(os.environ.get("ART_PER_QUERY_K", "10")),
        final_k=int(os.environ.get("ART_FINAL_K", "30")),
    )

    # Node 3 selection
    selected_ids, _ = select_topk(cands, max_bullets=MAX_BULLETS_ON_PAGE)

    # Rebuild resume data (strictly from my_experience.json)
    tailored_data = select_and_rebuild(STATIC_DATA, selected_ids)

    # Optional scoring + report
    report: Dict[str, Any] = {
        "selected_ids": selected_ids,
        "candidate_count": len(cands),
        "profile_used": profile is not None,
    }

    if profile is not None:
        pk = extract_profile_keywords(profile)

        selected_set = set(selected_ids)
        selected_candidates = [c for c in cands if c.bullet_id in selected_set]
        selected_bullets = [{"bullet_id": c.bullet_id, "text_latex": c.text_latex,
                             "meta": c.meta} for c in selected_candidates]
        all_bullets = [{"bullet_id": c.bullet_id,
                        "text_latex": c.text_latex, "meta": c.meta} for c in cands]

        skills_b = build_skills_pseudo_bullet(STATIC_DATA)
        all_bullets_plus_skills = all_bullets + \
            ([skills_b] if skills_b else [])

        must_evs_bullets_only = match_keywords_against_bullets(
            pk["must_have"], selected_bullets)
        nice_evs_bullets_only = match_keywords_against_bullets(
            pk["nice_to_have"], selected_bullets)

        must_evs_all = match_keywords_against_bullets(
            pk["must_have"], all_bullets_plus_skills)
        nice_evs_all = match_keywords_against_bullets(
            pk["nice_to_have"], all_bullets_plus_skills)

        hybrid = hybrid_score(
            selected_candidates=selected_candidates,
            all_candidates=cands,
            profile_keywords=pk,
            must_evs_all=must_evs_all,
            nice_evs_all=nice_evs_all,
            must_evs_bullets_only=must_evs_bullets_only,
            nice_evs_bullets_only=nice_evs_bullets_only,
            alpha=float(os.environ.get("ART_SCORE_ALPHA", "0.7")),
            must_weight=float(os.environ.get("ART_MUST_WEIGHT", "0.8")),
        )

        report.update({
            "hybrid": {
                "final_score": hybrid.final_score,
                "retrieval_score": hybrid.retrieval_score,
                "coverage_bullets_only": hybrid.coverage_bullets_only,
                "coverage_all": hybrid.coverage_all,
                "must_missing_bullets_only": hybrid.must_missing_bullets_only,
                "nice_missing_bullets_only": hybrid.nice_missing_bullets_only,
                "must_missing_all": hybrid.must_missing_all,
                "nice_missing_all": hybrid.nice_missing_all,
            }
        })

    report_path = os.path.join(OUTPUT_DIR, "resume_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Node 5/6 render
    pdf_path = render_pdf(tailored_data)

    t_all.stop()
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename="tailored_resume.pdf",
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
