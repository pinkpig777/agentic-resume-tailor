# Agentic Resume Tailor (ART)

Local, privacy-first resume tailoring agent. ART keeps your profile on disk, stores bullets in a local ChromaDB vector store, retrieves the most relevant bullets for a job description (JD), and renders a single-page LaTeX PDF via Tectonic.

This repo has two runtimes:

- FastAPI backend (`src/server.py`): API endpoints, agent loop, rendering, artifact/report generation.
- Streamlit UI (`src/app.py`): Generate, Resume Editor, and Settings pages.

---

## Quickstart (Docker Compose)

```bash
docker compose up --build
```

Open:

- API health: `http://localhost:8000/health`
- Streamlit UI: `http://localhost:8501`

Then:

1) Open **Resume Editor**, create your profile.
2) Click **Re-ingest ChromaDB**.
3) Open **Generate**, paste a JD, and create a tailored resume.

---

## Local run (Python)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/server.py
streamlit run src/app.py
```

Optional: create a `.env` with `OPENAI_API_KEY=...` for the JD parser.

---

## Documentation

For full details (data workflow, settings, API, schema, repo layout, diagram, troubleshooting), see `ARCHITECTURE.md`.
