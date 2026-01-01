# Agentic Resume Tailor (ART)

Local, privacy-first resume tailoring agent. ART keeps your profile on disk, stores bullets in a local ChromaDB vector store, retrieves the most relevant bullets for a job description (JD), and renders a single-page LaTeX PDF via Tectonic.

This repo has two runtimes:

- FastAPI backend (`src/server.py`): API endpoints, agent loop, rendering, artifact/report generation.
- Streamlit UI (`src/app.py`): Generate, Resume Editor, and Settings pages.

---

![Resume Editor](docs/images/figure.jpg)

## Highlights

- 🚀 Generate a single-page PDF and LaTeX source from any JD.
- 🗃️ Edit your profile end-to-end with DB-backed CRUD (personal info, education, experiences, projects).
- 🔁 Agentic loop boosts missing must-have keywords and blends retrieval + coverage scores.
- 🧾 Explainability reports show selected IDs, scores, and keyword evidence.
- 🔄 One-click export and re-ingest keeps Chroma in sync.

## User Guide

This guide covers day-to-day usage. For setup and deployment, see `ARCHITECTURE.md`.

### 1) Open the app

- Streamlit UI: `http://localhost:8501`
- API health: `http://localhost:8000/health`

### 2) Build your profile (Resume Editor)

1. Open **Resume Editor** in the sidebar.
2. Fill out **Personal Info** and **Skills**.
3. Add **Education**, **Work Experience**, and **Projects**.
4. Add bullets under each experience/project.
5. Click **Re-ingest ChromaDB** to refresh retrieval after edits.

Notes:

- Bullet IDs are stable (e.g., `b01`, `b02`) and never renumbered.
- Bullet text is LaTeX-ready; the system does not rewrite it.

### 3) Adjust defaults (Settings)

- Open **Settings** to change default generation and ingest behavior.
- Settings are loaded from `config/user_settings.json` and overridden by:
  - `config/user_settings.local.json` for local runs
  - `config/user_settings.docker.json` for Docker/Compose runs
- If you enable **Auto re-ingest on save**, the vector store refreshes after each edit.
- Use **Advanced tuning** to adjust the quantitative bullet bonus (per-hit and cap).
- The JD parser model is selected from a dropdown of current OpenAI models (or override in
  `config/user_settings.json`).

### 4) Generate a tailored resume (Generate)

1. Open **Generate**.
2. Paste a JD and click **Generate**.
3. Review the report, then download the PDF and report JSON.

Outputs:

- PDF: `output/<run_id>.pdf`
- TeX: `output/<run_id>.tex`
- Report: `output/<run_id>_report.json`

### 5) Exported JSON artifact

- `data/my_experience.json` is an exported artifact (for backup/inspection).
- The SQL database remains the source of truth.

---

## Need setup details?

See `ARCHITECTURE.md` for deployment steps, system diagrams, DB schema, and API details.

Template tip: create `templates/resume.local.tex` to override the default `templates/resume.tex`
without committing your personal edits.

---

## Local development (uv)

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# run unit tests
uv run pytest

# run the API
uv run python src/server.py

# run the UI
uv run streamlit run src/app.py
```
