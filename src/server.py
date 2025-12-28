import copy
import json
import os
import subprocess
import time

import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import jinja2
import uvicorn

# --- CONFIGURATION ---
MAX_BULLETS_ON_PAGE = 16       # Fits comfortably on one page
DB_PATH = "/app/data/processed/chroma_db"
DATA_FILE = "/app/data/my_experience.json"
TEMPLATE_DIR = "/app/templates"
OUTPUT_DIR = "/app/output"

app = FastAPI()
templates = Jinja2Templates(directory="/app/templates")


class Timer:
    def __init__(self, name: str):
        self.name = name
        self.start = time.time()

    def stop(self):
        elapsed = time.time() - self.start
        print(f"⏱️  [TIME] {self.name}: {elapsed:.2f}s")
        return elapsed

# --- 1. GLOBAL STARTUP (Runs Once) ---
print("⚙️  Server Starting: Loading Brain...")

# Load JSON Data
with open(DATA_FILE, "r") as f:
    static_data = json.load(f)

# Load ChromaDB (The "Brain")
client = chromadb.PersistentClient(path=DB_PATH)
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2")
collection = client.get_collection(
    name="resume_experience", embedding_function=ef)

print("✅  Brain Loaded! Ready for requests.")


# --- 2. CORE ALGORITHMS ---

def render_pdf(context):
    """
    Render the tailored resume to PDF using the LaTeX template.
    """
    t_render = Timer("Render & Compile PDF")

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(TEMPLATE_DIR),
        block_start_string='((%',
        block_end_string='%))',
        variable_start_string='<<',
        variable_end_string='>>',
        comment_start_string='((#',
        comment_end_string='#))'
    )
    template = env.get_template('resume.tex')
    tex_content = template.render(context)

    tex_path = os.path.join(OUTPUT_DIR, "tailored_resume.tex")
    with open(tex_path, "w") as f:
        f.write(tex_content)

    print("📄 Compiling PDF...")
    subprocess.run(["tectonic", tex_path], check=True)
    t_render.stop()

def get_bullet_scores(jd_text, collection):
    """
    Scoring Engine: Compares JD against ALL experience bullets.
    Returns: { "bullet_text": relevance_score (0.0 - 1.0) }
    """
    # Fetch broadly to catch all potential matches
    results = collection.query(
        query_texts=[jd_text],
        n_results=100,  # Fetch everything relevant
        include=["documents", "distances"]
    )

    scores = {}
    if results['documents']:
        for i, doc in enumerate(results['documents'][0]):
            dist = results['distances'][0][i]
            # Convert Distance (0=perfect, 2=bad) to Score (1.0=perfect, 0.0=bad)
            # This formula gives a score of ~1.0 for exact matches, ~0.0 for unrelated
            score = max(0, (1.5 - dist) / 1.5)
            scores[doc] = score

    return scores


def optimize_resume(full_data, bullet_scores):
    """
    The Logic: "Fill First, Trim Later"
    1. Collect ALL bullets.
    2. If total > MAX_BULLETS, trim the lowest scoring ones.
    3. Reconstruct the resume chronologically.
    """
    tailored = copy.deepcopy(full_data)
    all_candidates = []

    # --- PHASE A: COLLECT EVERYTHING ---
    def collect_bullets(section_list, section_type):
        for item in section_list:
            for b in item.get('bullets', []):
                # Default 0.0 if AI didn't pick it up
                s = bullet_scores.get(b, 0.0)
                all_candidates.append({
                    'text': b,
                    'score': s,
                    'section_type': section_type
                })

    collect_bullets(tailored.get('experiences', []), 'exp')
    collect_bullets(tailored.get('projects', []), 'proj')

    # --- PHASE B: CAPACITY CHECK ---
    current_count = len(all_candidates)
    print(f"📊 Total Bullets Available: {current_count}")

    if current_count <= MAX_BULLETS_ON_PAGE:
        print("✅ Under capacity. Keeping EVERYTHING (no cuts).")
        survivors = all_candidates
    else:
        print(
            f"✂️  Over capacity ({current_count} > {MAX_BULLETS_ON_PAGE}). Trimming weak bullets...")
        # Sort by Score (High -> Low)
        all_candidates.sort(key=lambda x: x['score'], reverse=True)
        # Keep Top N
        survivors = all_candidates[:MAX_BULLETS_ON_PAGE]

    # Create lookup set for fast filtering
    surviving_texts = set(c['text'] for c in survivors)

    # --- PHASE C: RECONSTRUCTION (Chronological) ---
    final_experiences = []
    final_projects = []

    # 1. Rebuild Experience
    for job in tailored.get('experiences', []):
        # Filter bullets
        job['bullets'] = [b for b in job['bullets'] if b in surviving_texts]

        # Keep job if it has bullets
        if job['bullets']:
            final_experiences.append(job)

    # 2. Rebuild Projects
    for proj in tailored.get('projects', []):
        proj['bullets'] = [b for b in proj['bullets'] if b in surviving_texts]

        if proj['bullets']:
            final_projects.append(proj)

    # --- PHASE D: SAFETY NET (Current Job) ---
    # Ensure most recent job exists, even if irrelevant
    if tailored.get('experiences') and (not final_experiences or final_experiences[0]['company'] != tailored['experiences'][0]['company']):
        print("⚠️  Warning: Most recent job was dropped. Forcing it back...")
        recent_job = tailored['experiences'][0]
        # Keep its top bullet (or generic one)
        if recent_job['bullets']:
            recent_job['bullets'] = [recent_job['bullets'][0]]
        final_experiences.insert(0, recent_job)

    tailored['experiences'] = final_experiences
    tailored['projects'] = final_projects

    print(
        f"✅ Final Resume: {len(final_experiences)} Jobs, {len(final_projects)} Projects.")
    return tailored


# --- 3. WEB ROUTES ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return """
    <html>
        <head>
            <title>Resume Agent</title>
            <style>
                body { font-family: -apple-system, system-ui, sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; line-height: 1.6; }
                textarea { width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 4px; font-family: monospace; }
                button { background: #2563eb; color: white; border: none; padding: 12px 24px; border-radius: 4px; cursor: pointer; font-size: 16px; margin-top: 10px;}
                button:hover { background: #1d4ed8; }
            </style>
        </head>
        <body>
            <h1>🚀 AI Resume Tailor</h1>
            <p>Paste the Job Description below. The AI will rank your experience and generate a one-page PDF.</p>
            <form action="/generate" method="post">
                <textarea name="jd_text" rows="15" placeholder="Paste Job Description here..."></textarea>
                <br>
                <button type="submit">Generate Optimized PDF</button>
            </form>
        </body>
    </html>
    """


@app.post("/generate")
async def generate_resume(request: Request):
    form_data = await request.form()
    jd_text = form_data['jd_text']

    # 1. Score
    scores = get_bullet_scores(jd_text, collection)

    # 2. Optimize (Fill & Trim)
    tailored_data = optimize_resume(static_data, scores)

    # 3. Render
    # Note: We assume render_pdf writes to /app/output/tailored_resume.pdf
    render_pdf(tailored_data)

    return FileResponse(
        "/app/output/tailored_resume.pdf",
        media_type='application/pdf',
        filename="tailored_resume.pdf"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
