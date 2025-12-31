# Agentic Resume Tailor (ART)

Local, privacy-first resume tailoring agent. ART stores only your own bullets in a local ChromaDB vector store, retrieves the most relevant bullets for a job description (JD), and renders a single-page LaTeX PDF via Tectonic.

This repo supports two runtimes:

- FastAPI backend (`src/server.py`): API endpoints, agent loop, rendering, artifact/report generation.
- Streamlit UI (`src/app.py`): frontend that calls the API, includes the Resume Editor, and visualizes reports.

---

## What it does

- Local vector store (ChromaDB): embeds each bullet with `BAAI/bge-small-en-v1.5`.
- Deterministic provenance: each bullet is stored with a stable `bullet_id`.
- Multi-query retrieval: uses JD parser output (`experience_queries`) or fallback queries, merges + dedupes by `bullet_id`, then reranks.
- Selection (Top-K): selects the top `N` bullets (default `16`).
- Agentic loop: iteratively boosts missing must-have keywords into the next retrieval pass (no OpenAI needed for boosting).
- Keyword matching: canonicalization + family matching for explainability and coverage scoring.
- Hybrid scoring: blends retrieval strength + keyword coverage per iteration.
- Quant bonus: adds a small, bounded boost for bullets with quantitative results.
- Within-experience ordering: sorts selected bullets inside each job/project by relevance (ties by bullet id).
- LaTeX rendering: Jinja2 -> `.tex` -> Tectonic -> `.pdf`.
- Report: writes `output/<run_id>_report.json` with queries used, selected bullet IDs, missing keywords, scores, and iteration history.
- Resume Editor: CRUD experiences, projects, and bullets with a one-click re-ingest to Chroma.

---

## Repo map

- `data/`
  - `raw_experience_data_example.json` - legacy JSON sample (not used by default)
  - `my_experience.json` - JSON export artifact (written on saves and ingest)
  - `processed/chroma_db/` - local ChromaDB store
  - `processed/resume.db` - SQLite CRUD store (default)
- `config/user_settings.json` - user-editable app settings (local defaults)
- `config/user_settings.docker.json` - Docker-friendly settings (api_url points to `http://api:8000`)
- `script/`
  - `convert_experience_json.py` - normalize raw data and assign stable IDs
  - `test_query.py` - manual retrieval/loop debug runner
  - `test_render.py` - render a PDF from template using sample JSON
- `config/`
  - `canonicalization.json` - alias/canonical rules
  - `families.json` - family taxonomy (generic -> satisfied_by)
- `src/`
  - `agentic_resume_tailor/` - src-layout package
    - `api/server.py` - FastAPI backend (API-only, writes artifacts + report)
    - `db/` - SQLAlchemy models + export/seed helpers for CRUD
    - `ui/app.py` - Streamlit UI (calls backend, visualizes report, downloads PDF)
    - `core/` - retrieval/selection/scoring pipeline
    - `ingest.py` - upserts bullets into Chroma using deterministic `bullet_id`
    - `jd_parser.py` - optional OpenAI JD parser (Target Profile v1)
    - `core/jd_utils.py` - shared JD parsing + fallback query helpers
    - `settings.py` - pydantic-settings config loader
    - `utils/logging.py` - log configuration helpers
  - `server.py`, `app.py`, `ingest.py` - thin wrappers for backward-compatible entrypoints
- `tests/`
  - `characterization/run_generate_characterization.py` - black-box generate test
  - `fixtures/` - characterization fixtures and expected output
  - `unit/` - fast unit tests for core modules
- `templates/resume.tex` - Jinja2 LaTeX template with `<< >>` and `((% %))` delimiters
- `output/` - generated artifacts (`<run_id>.pdf`, `<run_id>.tex`, `<run_id>_report.json`,
  `my_experience.json`)

---

## Data workflow (DB-first)

- The SQL database is the source of truth (created on first launch).
- The Resume Editor writes directly to the DB via CRUD endpoints.
- Re-ingest always exports the DB to `data/my_experience.json`, then ingests Chroma.
- `data/my_experience.json` is an exported artifact for inspection/backups, not the primary store.
- App settings are stored in `config/user_settings.json` and editable in the Resume Editor.

### Export format (`my_experience.json`)

The export file uses the same schema as before:

```json
{ "id": "b01", "text_latex": "..." }
```

Notes:
- Use `$|$` inside `role` to separate primary title from team/focus. The primary title is used to create a stable `job_id`.
- Bullets are LaTeX-ready and are never rewritten by the system.
- The template expects `personal_info`, `skills`, `education`, `experiences`, and `projects` to exist (use empty lists when needed).

### Legacy JSON tools (not on the happy path)

`script/convert_experience_json.py` still exists to normalize old JSON, but the app does not
auto-import JSON on startup. Use the Resume Editor to populate the DB.

### `bullet_id` convention

- Experience bullets: `exp:<job_id>:<bullet_local_id>`
- Project bullets: `proj:<project_id>:<bullet_local_id>`

Examples:

- `exp:saturnai__ai_software_engineer:b03`
- `proj:zapmail_ai_driven_email_automation_platform:b02`

---

## Database-backed CRUD

- CRUD endpoints are available at `/personal_info`, `/skills`, `/experiences`, `/projects`, and `/education`.
- Use `POST /admin/ingest` to export the DB, rebuild Chroma, and refresh in-memory data.
- Use `POST /admin/export` to regenerate `data/my_experience.json` without re-ingesting.

## Database schema (SQLite/Postgres)

Tables and relations (SQLAlchemy models in `src/agentic_resume_tailor/db/models.py`):

- `personal_info` (single row)
  - `id` (PK), `name`, `phone`, `email`, `linkedin_id`, `github_id`, `linkedin`, `github`
- `skills` (single row)
  - `id` (PK), `languages_frameworks`, `ai_ml`, `db_tools`
- `education`
  - `id` (PK), `school`, `dates`, `degree`, `location`, `sort_order`
- `education_bullets`
  - `id` (PK), `education_id` (FK → `education.id`), `text_latex`, `sort_order`
- `experiences`
  - `id` (PK), `job_id` (unique), `company`, `role`, `dates`, `location`, `sort_order`
- `experience_bullets`
  - `id` (PK), `experience_id` (FK → `experiences.id`), `local_id` (`b01`...), `text_latex`,
    `sort_order`
  - unique constraint: (`experience_id`, `local_id`)
- `projects`
  - `id` (PK), `project_id` (unique), `name`, `technologies`, `sort_order`
- `project_bullets`
  - `id` (PK), `project_id` (FK → `projects.id`), `local_id` (`b01`...), `text_latex`,
    `sort_order`
  - unique constraint: (`project_id`, `local_id`)

Notes:
- `job_id` and `project_id` are deterministic slugs; bullet `local_id` is stable and never derived from text.
- Ordering is controlled by `sort_order`; deletes never renumber IDs.

## Resume Editor (UI)

- Open the Streamlit app and switch to **Resume Editor** in the sidebar.
- Create, edit, and delete personal info, skills, education, experiences/projects, and bullets.
- Click **Re-ingest ChromaDB** after edits so retrieval reflects the latest data.

---

## Environment

Create a `.env` in the repo root (optional, for secrets only):

```env
OPENAI_API_KEY=YOUR_OPENAI_API_KEY
```

If you run locally, load `.env` with `python -m dotenv run -- <command>` or export variables in your shell.
The app reads `OPENAI_API_KEY` from `.env` or the process environment.

All app settings live in `config/user_settings.json`. Keys map to the Settings fields:

```json
{
  "db_path": "data/processed/chroma_db",
  "sql_db_url": "sqlite:///data/processed/resume.db",
  "export_file": "data/my_experience.json",
  "auto_reingest_on_save": false,
  "template_dir": "templates",
  "output_dir": "output",
  "collection_name": "resume_experience",
  "embed_model": "BAAI/bge-small-en-v1.5",
  "use_jd_parser": true,
  "max_bullets": 16,
  "per_query_k": 10,
  "final_k": 30,
  "max_iters": 3,
  "threshold": 80,
  "alpha": 0.7,
  "must_weight": 0.8,
  "boost_weight": 1.6,
  "boost_top_n_missing": 6,
  "cors_origins": "*",
  "skip_pdf": false,
  "run_id": null,
  "jd_model": "gpt-4.1-nano-2025-04-14",
  "canon_config": "config/canonicalization.json",
  "family_config": "config/families.json",
  "api_url": "http://localhost:8000",
  "log_level": "INFO",
  "log_json": false,
  "port": 8000
}
```

---

## Prerequisites

- Docker, or Python 3.10+ with `pip`
- Internet access for the initial embedding model download (cached afterwards)
- Keep `data/*.json` and `.env` private (gitignored)

---

## Quickstart (Docker Compose, recommended)

1) Start API + UI:

```bash
docker compose up --build
```

Open:

- API health: `http://localhost:8000/health`
- Streamlit UI: `http://localhost:8501`

2) In the UI, open **Resume Editor**, create your profile, then click **Re-ingest ChromaDB**.

Note: Compose uses `config/user_settings.docker.json`. Edit that file to change settings.

Stop:

```bash
docker compose down
```

---

## Docker (no Compose)

### 0) Build the image

```bash
docker build -t resume-agent .
```

### 1) Run the FastAPI backend (API)

```bash
docker run --rm -p 8000:8000 \
  --env-file .env \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/output:/app/output" \
  -v "$(pwd)/src:/app/src" \
  -v "$(pwd)/templates:/app/templates" \
  -v "$(pwd)/config:/app/config" \
  -v "$(pwd)/.cache_docker:/root/.cache" \
  resume-agent python src/server.py
```

Health check:

```bash
curl -sS http://localhost:8000/health
```

### 2) Run the Streamlit UI (separate container)

```bash
docker run --rm -p 8501:8501 \
  --env-file .env \
  -v "$(pwd)/src:/app/src" \
  resume-agent streamlit run /app/src/app.py --server.address=0.0.0.0 --server.port=8501
```

Open Streamlit: `http://localhost:8501`

Then open **Resume Editor**, create your profile, and click **Re-ingest ChromaDB**.

Note: set `api_url` in `config/user_settings.json` if the UI needs a non-default API URL.

---

## Local run (Python)

If you run outside Docker, update `config/user_settings.json` with local paths:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Example settings are shown in the Environment section above.

python src/server.py
streamlit run src/app.py
```

Then open **Resume Editor**, create your profile, and click **Re-ingest ChromaDB**.

Note: Tectonic must be installed on your machine to render PDFs locally.

---

## Development

Format + lint:

```bash
ruff format .
ruff check --fix .
```

Tests:

```bash
# characterization (black-box) test
python tests/characterization/run_generate_characterization.py

# update expected output if intentional behavior changes
python tests/characterization/run_generate_characterization.py --update

# unit tests
python -m unittest discover -s tests/unit
```

---

## API usage

### `GET /health`

Returns status JSON used by UI and Compose healthchecks.

### `POST /generate`

Input: JD text + settings (max bullets, loop knobs, scoring weights).  
Output: `run_id` + URLs for artifacts (PDF / TEX / report).

Example:

```bash
curl -sS http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "jd_text": "Paste job description here",
    "max_bullets": 16,
    "max_iters": 3
  }'
```

Artifacts are written under `output/` and exposed via:

- `/runs/{run_id}/pdf`
- `/runs/{run_id}/tex`
- `/runs/{run_id}/report`

### Resume CRUD

- `GET /experiences`, `GET /projects`
- `POST /experiences`, `POST /projects`
- `PUT /experiences/{job_id}`, `PUT /projects/{project_id}`
- `DELETE /experiences/{job_id}`, `DELETE /projects/{project_id}`
- `POST /experiences/{job_id}/bullets`, `PUT /experiences/{job_id}/bullets/{local_id}`,
  `DELETE /experiences/{job_id}/bullets/{local_id}`
- `POST /projects/{project_id}/bullets`, `PUT /projects/{project_id}/bullets/{local_id}`,
  `DELETE /projects/{project_id}/bullets/{local_id}`

### Admin operations

- `POST /admin/export` regenerates `data/my_experience.json` from the DB.
- `POST /admin/ingest` exports `data/my_experience.json` and re-ingests Chroma (returns counts + elapsed time).

---

## Work flow

```mermaid
flowchart TD
  subgraph Resume_Data[Resume Data]
    UI[Resume Editor] --> CRUD[FastAPI CRUD]
    CRUD --> DB[(SQL DB)]
    DB -->|export| JSON[data/my_experience.json]
    JSON -->|ingest| INGEST[Re-ingest Chroma]
    INGEST --> CHROMA[(ChromaDB)]
  end

  A[Client submits JD text + settings] --> B[FastAPI /generate]
  DB --> B
  B --> C[Normalize JD text]

  C --> D{JD Parser enabled?}
  D -- No --> Q0[Use manual retrieval queries]
  D -- Yes --> E[Node 1: LLM JD Parser]
  E --> E1[Target Profile v1<br/>atomic keywords + evidence snippets<br/>experience_queries + weights]
  E1 --> Q1[Use experience_queries]

  Q0 --> F[Node 2: Multi-query Retrieve<br/>ChromaDB]
  Q1 --> F

  F --> G[Merge + Dedupe by bullet_id<br/>Rerank multi-hit + weights]
  G --> H[Node 3: Select bullets<br/>Top-K max 16]
  H --> I[Node 4: Keyword Matcher<br/>exact + alias + family tiers]
  I --> J[Node 5: Hybrid Score<br/>alpha*retrieval + 1-alpha*coverage]

  J --> K{Loop enabled<br/>and score less than threshold?}
  K -- No --> L[Node 6: Render .tex<br/>from template + selected bullets]
  K -- Yes --> M[Boost missing must-have keywords<br/>into next retrieval pass]
  M --> F

  L --> N[Node 7: Compile PDF<br/>Tectonic]
  N --> O[Write artifacts to output/<br/>run_id.pdf<br/>run_id.tex<br/>run_id_report.json]
  O --> P[Return response to client<br/>download links / streamed PDF]

  CHROMA --> F
```

---

## Notes and limitations

- JD parser requires `OPENAI_API_KEY`. If it fails or is disabled, the system falls back to local queries and skips keyword coverage scoring.
- The agent never rewrites bullet text; it only selects and arranges existing bullets.
- Retrieval quality depends heavily on query quality. The JD parser is designed to produce dense retrieval queries.
- Re-ingesting deletes and rebuilds the Chroma collection; run it after CRUD changes or after a JSON import.

---

## Troubleshooting

### "Collection does not exist" or 0 records

Ensure you have profile data in the DB, then run ingest:

```bash
python src/ingest.py
```

Or via API (exports from DB first):

```bash
curl -sS -X POST http://localhost:8000/admin/ingest
```

Docker Compose:

```bash
docker compose run --rm api python /app/src/ingest.py
```

### Streamlit cannot connect to API

- Streamlit inside Docker must call the API by service name: `http://api:8000` (Compose sets this).
- Streamlit outside Docker should call `http://localhost:8000`.

### UI loads but `/health` fails in Compose

The Compose healthcheck uses a Python urllib command. If you replace it with `curl`, install curl in the image.
