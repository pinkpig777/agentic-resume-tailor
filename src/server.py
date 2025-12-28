from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import shutil
import os
import chromadb
from chromadb.utils import embedding_functions
import copy
# Import our existing logic
from main import load_data, get_relevant_bullets_whitelist, reconstruct_resume_data, render_pdf

app = FastAPI()
templates = Jinja2Templates(directory="/app/templates")

# --- GLOBAL STATE (Loaded Once!) ---
print("⚙️  Server Starting: Loading Brain...")
DB_PATH = "/app/data/processed/chroma_db"
client = chromadb.PersistentClient(path=DB_PATH)
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2")
collection = client.get_collection(
    name="resume_experience", embedding_function=ef)
static_data = load_data()
print("✅  Brain Loaded! Ready for requests.")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # Simple HTML Form
    return """
    <html>
        <head><title>AI Resume Agent</title></head>
        <body style="font-family: sans-serif; max-width: 800px; margin: 40px auto; padding: 20px;">
            <h1>📄 AI Resume Generator</h1>
            <form action="/generate" method="post">
                <textarea name="jd_text" rows="10" style="width: 100%;" placeholder="Paste Job Description here..."></textarea><br><br>
                <button type="submit" style="padding: 10px 20px; font-size: 16px; cursor: pointer;">Generate Resume</button>
            </form>
        </body>
    </html>
    """


@app.post("/generate")
async def generate_resume(request: Request):
    form_data = await request.form()
    jd_text = form_data['jd_text']

    # 1. Logic (Uses pre-loaded 'collection' and 'static_data')
    whitelist = get_relevant_bullets_whitelist(jd_text, collection)
    tailored_data = reconstruct_resume_data(static_data, whitelist)

    # 2. Render
    render_pdf(tailored_data)

    # 3. Return the PDF
    return FileResponse("/app/output/tailored_resume.pdf", media_type='application/pdf', filename="tailored_resume.pdf")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
