# src/server.py
import copy
import json
import os
import subprocess
import time
import sys

import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse
import jinja2
from jinja2 import TemplateError
import uvicorn

from loop_controller import run_loop

# --- CONFIG ---
MAX_BULLETS_ON_PAGE = int(os.environ.get("ART_MAX_BULLETS", "16"))
DB_PATH = "/app/data/processed/chroma_db"
DATA_FILE = "/app/data/my_experience.json"
TEMPLATE_DIR = "/app/templates"
OUTPUT_DIR = "/app/output"

EMBED_MODEL = os.environ.get("ART_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
COLLECTION_NAME = os.environ.get("ART_COLLECTION", "resume_experience")

USE_JD_PARSER = os.environ.get("ART_USE_JD_PARSER", "1") == "1"
THRESHOLD = int(os.environ.get("ART_THRESHOLD", "80"))
MAX_ITERS = int(os.environ.get("ART_MAX_ITERS", "3"))
ALPHA = float(os.environ.get("ART_SCORE_ALPHA", "0.7"))
PER_QUERY_K = int(os.environ.get("ART_PER_QUERY_K", "10"))
FINAL_K = int(os.environ.get("ART_FINAL_K", "30"))

app = FastAPI()


class Timer:
    def __init__(self, name: str):
        self.name = name
        self.start = time.time()

    def stop(self):
        elapsed = time.time() - self.start
        print(f"⏱️  [TIME] {self.name}: {elapsed:.2f}s")
        return elapsed


print("⚙️  Server Starting: Loading data + ChromaDB...")

with open(DATA_FILE, "r", encoding="utf-8") as f:
    static_data = json.load(f)

client = chromadb.PersistentClient(path=DB_PATH)
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL)
collection = client.get_collection(name=COLLECTION_NAME, embedding_function=ef)

print(
    f"✅ Loaded collection '{COLLECTION_NAME}' with {collection.count()} records")


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
        print(f"⚠️ JD parser failed; continuing without profile. Reason: {e}")
        return None


def bullet_id_for_exp(job: dict, b: dict) -> str:
    job_id = job.get("job_id")
    bid = b.get("id")
    if not job_id or not bid:
        raise ValueError("experience is missing job_id or bullet.id")
    return f"exp:{job_id}:{bid}"


def bullet_id_for_proj(proj: dict, b: dict) -> str:
    proj_id = proj.get("project_id")
    bid = b.get("id")
    if not proj_id or not bid:
        raise ValueError("project is missing project_id or bullet.id")
    return f"proj:{proj_id}:{bid}"


def rebuild_resume_with_selected_bullets(full_data: dict, selected_ids: list[str]) -> dict:
    """
    Filters experiences/projects based on selected bullet_ids, then converts bullets to list[str] (text_latex)
    so your LaTeX template can render easily.
    """
    selected_set = set(selected_ids)
    tailored = copy.deepcopy(full_data)

    # experiences
    final_exps = []
    for job in tailored.get("experiences", []):
        kept = []
        for b in job.get("bullets", []):
            if not isinstance(b, dict):
                continue
            bid = bullet_id_for_exp(job, b)
            if bid in selected_set:
                kept.append(b.get("text_latex", ""))
        kept = [x for x in kept if x]
        if kept:
            job["bullets"] = kept
            final_exps.append(job)

    # projects
    final_projs = []
    for proj in tailored.get("projects", []):
        kept = []
        for b in proj.get("bullets", []):
            if not isinstance(b, dict):
                continue
            bid = bullet_id_for_proj(proj, b)
            if bid in selected_set:
                kept.append(b.get("text_latex", ""))
        kept = [x for x in kept if x]
        if kept:
            proj["bullets"] = kept
            final_projs.append(proj)

    tailored["experiences"] = final_exps
    tailored["projects"] = final_projs
    return tailored


def render_pdf(context: dict) -> str:
    """
    Renders tailored_resume.tex and compiles to tailored_resume.pdf.
    Returns PDF path.
    """
    t = Timer("Render & Compile PDF")

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(TEMPLATE_DIR),
        block_start_string="((%",
        block_end_string="%))",
        variable_start_string="<<",
        variable_end_string=">>",
        comment_start_string="((#",
        comment_end_string="#))",
    )

    try:
        template = env.get_template("resume.tex")
        tex_content = template.render(context)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        tex_path = os.path.join(OUTPUT_DIR, "tailored_resume.tex")
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(tex_content)
    except TemplateError as e:
        print(f"❌ Jinja2 Template Error: {e}", file=sys.stderr)
        raise
    except IOError as e:
        print(f"❌ File Write Error: {e}", file=sys.stderr)
        raise

    try:
        subprocess.run(
            ["tectonic", tex_path],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 40, file=sys.stderr)
        print("❌ TECTONIC COMPILATION FAILED", file=sys.stderr)
        print("=" * 40, file=sys.stderr)
        print("STDOUT:", file=sys.stderr)
        print(e.stdout, file=sys.stderr)
        print("STDERR:", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        print("=" * 40, file=sys.stderr)
        raise

    pdf_path = os.path.join(OUTPUT_DIR, "tailored_resume.pdf")
    t.stop()
    return pdf_path


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return """
    <html>
      <head>
        <title>Resume Agent</title>
        <style>
          body { font-family: -apple-system, system-ui, sans-serif; max-width: 900px; margin: 40px auto; padding: 20px; line-height: 1.6; }
          textarea { width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 6px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
          button { background: #2563eb; color: white; border: none; padding: 12px 24px; border-radius: 6px; cursor: pointer; font-size: 16px; margin-top: 10px;}
          button:hover { background: #1d4ed8; }
        </style>
      </head>
      <body>
        <h1>AI Resume Tailor</h1>
        <p>Paste a JD. The system does multi-query retrieval + rerank + selection + optional retry loop.</p>
        <form action="/generate" method="post">
          <textarea name="jd_text" rows="18" placeholder="Paste Job Description here..."></textarea>
          <br/>
          <button type="submit">Generate PDF</button>
        </form>
      </body>
    </html>
    """


@app.post("/generate")
async def generate_resume(request: Request):
    form_data = await request.form()
    jd_text = str(form_data["jd_text"]).strip()
    if not jd_text:
        return HTMLResponse("JD is empty", status_code=400)

    profile = try_parse_jd(jd_text)

    best, history = run_loop(
        jd_text=jd_text,
        collection=collection,
        embedding_fn=ef,
        profile=profile,
        per_query_k=PER_QUERY_K,
        final_k=FINAL_K,
        max_bullets=MAX_BULLETS_ON_PAGE,
        threshold=THRESHOLD,
        max_iters=MAX_ITERS,
        alpha=ALPHA,
    )

    print("✅ Loop finished. Best iteration:", best.iter_idx)
    if best.score is not None:
        print("   score:", best.score.final_score,
              "| missing must:", best.must_missing[:8])

    tailored_data = rebuild_resume_with_selected_bullets(
        static_data, best.selected_ids)
    pdf_path = render_pdf(tailored_data)

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename="tailored_resume.pdf",
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
