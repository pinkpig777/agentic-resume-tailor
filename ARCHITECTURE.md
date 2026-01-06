# Technical Documentation

This document covers setup, deployment, architecture, and schemas.

---

## From zero to deployed

### Prerequisites

- Docker (recommended), or Python 3.10+ with `pip`
- Node.js 20.19+ or 22.12+
- Internet access for the initial embedding model download (cached afterward)
- Keep `backend/data/*.json` and `backend/.env` private (gitignored)

### Clone and configure

```bash
git clone https://github.com/pinkpig777/agentic-resume-tailor.git
cd agentic-resume-tailor
```

Create a `backend/.env` (optional, only for secrets):

```env
OPENAI_API_KEY=YOUR_OPENAI_API_KEY
```

Edit app settings (optional):

- `backend/config/user_settings.json` for defaults
- `backend/config/user_settings.local.json` for local runs
- `backend/config/user_settings.docker.json` for Docker Compose

### Build the Docker Image

```bash
docker build -t resume-agent ./backend
```

### Initial Chroma Ingestion

```bash
docker compose run --rm api python -m agentic_resume_tailor.ingest
```

### Deploy with Docker Compose (recommended)

```bash
docker compose up --build
```

Verify:

- API health: `http://localhost:8000/health`
- React SPA: `http://localhost:5173`

The Compose stack runs the FastAPI backend plus a Vite dev server for the frontend. The frontend
service is configured with `VITE_API_URL=http://localhost:8000`.

Then:

1. Open **Resume Editor**, create your profile.
2. Click **Re-ingest ChromaDB**.
3. Open **Generate**, paste a JD, and create a tailored resume.

Stop:

```bash
docker compose down
```

### Local run (uv)

```bash
uv venv
source .venv/bin/activate
cd backend
uv pip install -r requirements.txt

PYTHONPATH=src uv run python -m agentic_resume_tailor.api.server
```

```bash
# in another terminal
cd frontend
npm install
npm run dev
```

---

## System architecture

```mermaid
flowchart LR
  UI[React SPA] -->|REST API| API[FastAPI API]
  API -->|CRUD| DB[(SQL DB)]
  DB -->|export| JSON[backend/data/my_experience.json]
  JSON -->|ingest| CHROMA[(ChromaDB)]
  API -->|query| CHROMA
  API -->|render| OUT[backend/output/*.pdf, *.tex, *_report.json]
```

---

## Workflow diagram

```mermaid
flowchart TD
  A[Resume Editor CRUD] --> B[FastAPI writes SQL DB]
  B --> C[Export DB to backend/data/my_experience.json]
  C --> D[Ingest JSON into ChromaDB]
  D --> E[Build retrieval plan - multi queries]
  E --> F[Multi-query retrieve from ChromaDB]
  F --> G[Select top-K bullets]
  G --> H[Score coverage + retrieval]
  H --> I{Meets threshold?}
  I -- No --> J[Boost missing must-have terms]
  J -->|feedback loop| E
  I -- Yes --> K[Render PDF + report]
  K --> L[Optional UI edits to selection]
  L --> M[Re-render PDF + report]
```

---

## Class diagram (conceptual)

```mermaid
classDiagram
  class ReactSPA {
    +ResumeEditorPage()
    +SettingsPage()
    +GeneratePage()
  }

  class FastAPIService {
    +generate()
    +export_resume()
    +ingest_resume()
    +CRUD endpoints
  }

  class ResumeRepository {
    +export_resume_data()
    +write_resume_json()
    +CRUD helpers
  }

  class RetrievalPipeline {
    +multi_query_retrieve()
    +select_topk()
    +score()
  }

  class Renderer {
    +render_pdf()
  }

  ReactSPA --> FastAPIService
  FastAPIService --> ResumeRepository
  FastAPIService --> RetrievalPipeline
  FastAPIService --> Renderer
```

---

## Database diagram

```mermaid
erDiagram
  PERSONAL_INFO {
    int id PK
    string name
    string phone
    string email
    string linkedin_id
    string github_id
    string linkedin
    string github
  }

  SKILLS {
    int id PK
    string languages_frameworks
    string ai_ml
    string db_tools
  }

  EDUCATION {
    int id PK
    string school
    string dates
    string degree
    string location
    int sort_order
  }

  EDUCATION_BULLETS {
    int id PK
    int education_id FK
    string text_latex
    int sort_order
  }

  EXPERIENCES {
    int id PK
    string job_id
    string company
    string role
    string dates
    string location
    int sort_order
  }

  EXPERIENCE_BULLETS {
    int id PK
    int experience_id FK
    string local_id
    string text_latex
    int sort_order
  }

  PROJECTS {
    int id PK
    string project_id
    string name
    string technologies
    int sort_order
  }

  PROJECT_BULLETS {
    int id PK
    int project_id FK
    string local_id
    string text_latex
    int sort_order
  }

  EDUCATION ||--o{ EDUCATION_BULLETS : has
  EXPERIENCES ||--o{ EXPERIENCE_BULLETS : has
  PROJECTS ||--o{ PROJECT_BULLETS : has
```

---

## Data workflow (DB-first)

- The SQL database is the source of truth (created on first launch).
- The Resume Editor writes directly to the DB via CRUD endpoints.
- Re-ingest exports the DB to `backend/data/my_experience.json`, then ingests Chroma.
- `backend/data/my_experience.json` is an exported artifact for inspection/backups, not the primary store.

### `bullet_id` convention

- Experience bullets: `exp:<job_id>:<bullet_local_id>`
- Project bullets: `proj:<project_id>:<bullet_local_id>`

---

## Settings and environment

Create a `.env` in the repo root (optional, for secrets only):

```env
OPENAI_API_KEY=YOUR_OPENAI_API_KEY
```

All other app settings live in `backend/config/user_settings.json`, and the app auto-creates a
runtime override file on first start. Local runs write
`backend/config/user_settings.local.json`, while Docker/Compose writes
`backend/config/user_settings.docker.json`.
Quantitative bullet bonus tuning lives in `quant_bonus_per_hit` and `quant_bonus_cap`.
`experience_weight` biases experience bullets above projects during retrieval ranking.
`output_pdf_name` controls the download filename for PDFs (run-id artifacts remain on disk).
Set `USER_SETTINGS_FILE` to point at a custom settings file path.

---

## API reference (summary)

- `GET /health`
- `GET/PUT /settings`
- `POST /generate`
- `POST /runs/{run_id}/render`
- `GET /runs/{run_id}/pdf`
- `GET /runs/{run_id}/tex`
- `GET /runs/{run_id}/report`
- `GET/PUT /personal_info`
- `GET/PUT /skills`
- `GET/POST/PUT/DELETE /education`
- `GET/POST/PUT/DELETE /experiences`
- `GET/POST/PUT/DELETE /experiences/{job_id}/bullets`
- `GET/POST/PUT/DELETE /projects`
- `GET/POST/PUT/DELETE /projects/{project_id}/bullets`
- `POST /admin/export`
- `POST /admin/ingest`

---

## Repo layout

- `frontend/`
  - `src/` - React SPA (Vite, Tailwind, shadcn/ui)
- `backend/`
  - `src/agentic_resume_tailor/` - FastAPI backend (API-only, writes artifacts + report)
  - `tests/` - characterization + unit tests
  - `config/` - app settings + taxonomy configs
  - `data/` - exported JSON and local DB artifacts
  - `output/` - generated artifacts (`<run_id>.pdf`, `<run_id>.tex`, `<run_id>_report.json`)
  - `templates/` - LaTeX templates
  - `script/` - debug + utility scripts

---

## Development

Format + lint:

```bash
cd backend
ruff format .
ruff check --fix .
```

Tests:

```bash
# characterization (black-box) test
RUN_CHARACTERIZATION=1 pytest -m characterization

# update expected output if intentional behavior changes
python tests/characterization/run_generate_characterization.py --update

# unit tests
pytest
```
