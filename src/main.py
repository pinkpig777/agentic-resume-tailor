import time
import os
import copy
import json
import jinja2
import subprocess
import chromadb
from chromadb.utils import embedding_functions
from jd_parser import parse_job_description

# --- TIMING UTILITY ---


class Timer:
    def __init__(self, name):
        self.name = name
        self.start = time.time()

    def stop(self):
        elapsed = time.time() - self.start
        print(f"⏱️  [TIME] {self.name}: {elapsed:.2f}s")
        return elapsed


# --- CONFIGURATION ---
TEMPLATE_DIR = "/app/templates"
OUTPUT_DIR = "/app/output"
DATA_FILE = "/app/data/my_experience.json"
DB_PATH = "/app/data/processed/chroma_db"


def load_data():
    t = Timer("Load JSON Data")
    with open(DATA_FILE, "r") as f:
        data = json.load(f)
    t.stop()
    return data


def get_relevant_bullets_whitelist(jd_text, collection):
    t_total = Timer("Full RAG Pipeline")

    t_ai = Timer("OpenAI JD Analysis")
    print("🤖 Analyzing JD...")
    requirements = parse_job_description(jd_text)
    t_ai.stop()

    whitelist = set()

    t_db = Timer("ChromaDB Search")
    for query in requirements.experience_queries:
        results = collection.query(
            query_texts=[query],
            n_results=2,
            include=["documents"]
        )
        for doc in results['documents'][0]:
            whitelist.add(doc)
    t_db.stop()

    t_total.stop()
    return whitelist


# def reconstruct_resume_data(full_data, whitelist):
#     return full_data  # (Logic omitted for brevity, it's instant)
def reconstruct_resume_data(full_data, whitelist):
    """
    Creates a tailored copy of the resume data.
    """
    # 1. CRITICAL: Make a deep copy so we don't ruin the global 'static_data'
    #    This ensures every request starts with a fresh, full resume.
    tailored = copy.deepcopy(full_data)

    print(f"🎨 Filtering Resume... Whitelist size: {len(whitelist)}")

    # 2. Filter Experiences
    for job in tailored.get('experiences', []):
        original_bullets = job.get('bullets', [])

        # Keep bullet IF it matches a whitelist string
        # (We use simple string matching here. In prod, you might use IDs)
        filtered_bullets = [b for b in original_bullets if b in whitelist]

        # FALLBACK: If the AI picked NOTHING for this job, but it's in your history:
        # Option A: Keep 1 generic bullet so the job isn't empty?
        # Option B: Keep it empty?
        # Let's go with Option A: if whitelist missed it, keep the first bullet.
        if not filtered_bullets and original_bullets:
            filtered_bullets.append(original_bullets[0])

        job['bullets'] = filtered_bullets

    # 3. Filter Projects (Same logic)
    for project in tailored.get('projects', []):
        original_bullets = project.get('bullets', [])
        filtered_bullets = [b for b in original_bullets if b in whitelist]

        if not filtered_bullets and original_bullets:
            filtered_bullets.append(original_bullets[0])

        project['bullets'] = filtered_bullets

    return tailored


def render_pdf(context):
    t_render = Timer("Render & Compile PDF")

    # Jinja Setup
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


def main():
    t_global = Timer("Total Execution")

    # 1. Setup
    print("⚙️  Loading Libraries & DB...")
    t_setup = Timer("Import & DB Connection")
    client = chromadb.PersistentClient(path=DB_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2")
    collection = client.get_collection(
        name="resume_experience", embedding_function=ef)
    full_data = load_data()
    t_setup.stop()

    # 2. Input
    target_job_description = """
    We are looking for a Python Engineer with Computer Vision experience.
    Must know PyTorch, OpenCV, and Real-time video processing.
    """

    # 3. Logic
    whitelist = get_relevant_bullets_whitelist(
        target_job_description, collection)
    tailored_data = reconstruct_resume_data(full_data, whitelist)

    # 4. Render
    render_pdf(tailored_data)

    t_global.stop()


if __name__ == "__main__":
    main()
