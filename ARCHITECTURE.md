# Technical Documentation (ART)

This document describes ART's developer-facing architecture. For run commands and quickstart, see [README.md](README.md).

---

## System overview

```
┌─────────────────────────────────────────────────────────┐
│  Browser                                                │
│  React SPA (Vite + TypeScript)                         │
│  EditorPage · GeneratePage · SettingsPage              │
└────────────────────┬────────────────────────────────────┘
                     │ REST / SSE
┌────────────────────▼────────────────────────────────────┐
│  FastAPI backend  (app factory + lifespan)              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  CRUD APIs   │  │  /generate   │  │  /runs/*/    │  │
│  │ (profile DB) │  │  (agent loop)│  │  render/pdf  │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         │                 │                 │           │
│  ┌──────▼───────────────────────────────────▼───────┐   │
│  │          Unified Artifact Pipeline               │   │
│  │  build_tailored_snapshot → render_pdf →          │   │
│  │  trim_to_single_page → write_report              │   │
│  └──────────────────────────────────────────────────┘   │
└───┬────────────────┬────────────────────────────────────┘
    │                │
┌───▼───┐       ┌────▼────┐       ┌──────────────┐
│SQLite │       │ChromaDB │       │ OpenAI API   │
│(src)  │──────▶│(index)  │       │(agents,opt.) │
└───────┘ ingest└─────────┘       └──────────────┘
```

**Components**

- **React UI (Vite + TypeScript)**: Resume Editor, Settings, and Generate pages. Communicates with the backend over REST + Server-Sent Events.
- **FastAPI backend**: Uses the App Factory pattern with lifespan resource management (Chroma client, embeddings, run locks held in `app.state`). Handles CRUD APIs, the agent generation loop, PDF rendering, and artifact output.
- **SQLite (SQLAlchemy)**: Source of truth for all profile data.
- **ChromaDB**: Persistent vector store for multi-query semantic retrieval.
- **OpenAI (optional)**: Powers the Query, Rewrite, and Scoring agents. Requires `OPENAI_API_KEY`. Calls use the **Responses API** (`client.responses.parse`) for structured output.
- **Tectonic**: Renders LaTeX → PDF (optional; `skip_pdf = true` writes TeX only).

---

## Runtime & Settings semantics

### App Factory & Lifespan

The backend uses a FastAPI `lifespan` context manager to load heavy resources once at startup and store them in `app.state`:

- ChromaDB `PersistentClient`
- SentenceTransformers embedding function
- Per-run progress queues and ingest lock

This avoids module-level globals and makes resources injectable for testing.

### Per-request Settings snapshots

`load_settings()` constructs a fresh `Settings` instance on every call (no cache). Settings are layered:

1. `init_settings` (constructor kwargs)
2. JSON user config file (`config/user_settings.json`)
3. Limited env vars (only `OPENAI_API_KEY` and `PORT` are read from environment)
4. File secrets

### Live fields vs. restart-required fields

`GET /settings` returns two metadata arrays:

| Array | Meaning |
|-------|---------|
| `live_fields` | Take effect immediately on the next request |
| `restart_required_fields` | Require API server restart (e.g., `embed_model`, `db_path`, `collection_name`, `cors_origins`, `port`) |

---

## DB-first contract

- **SQLite** is the single authoritative source for resume data.
- The React UI performs all creates/updates/deletes via the CRUD API endpoints.
- `backend/data/my_experience.json` is an **exported artifact** — it is used for Chroma ingest and human-readable backup, but is never a primary source.
- Rewrites produced by the agent loop are ephemeral (stored in the run report only) and are never written back to SQLite or Chroma.

---

## Ingest pipeline invariants

```
SQLite DB  →  export JSON  →  Chroma ingest (delete + recreate collection)
```

- Exported JSON is the **only** source ingested into Chroma.
- Ingest never mutates SQLite; it only rebuilds the retrieval index.
- Triggers: explicit `POST /admin/ingest`, CLI `python -m agentic_resume_tailor.ingest`, or `auto_reingest_on_save = true`.

---

## Agent loop

### Orchestration (`loop_controller.py`)

```
JD text
  │
  ▼
[1] Query Agent  ─────────────────────────── builds target profile + retrieval plan
  │
  ▼
[2] Multi-query retrieve (ChromaDB)  ──────── fetch candidates per query
  │
  ▼
[3] Select Top-K  ─────────────────────────── merge, score, dedupe candidates
  │
  ▼
[4] Rewrite Agent  ────────────────────────── rephrase selected bullets (optional)
  │
  ▼
[5] Scoring Agent  ────────────────────────── deterministic score + semantic feedback
  │
  ├─── score < threshold && iters < max_iters ──▶ boost terms → back to [2]
  │
  └─── done
          │
          ▼
  [6] Unified Artifact Pipeline
       build_tailored_snapshot → render_pdf → trim_to_single_page → write_report
```

The loop keeps the **best-scoring** iteration across all passes and uses its artifacts for the final report.

### A) Query Agent

- **Input**: JD text.
- **Output**: structured `target_profile` (must-have/nice-to-have keywords, role summary) + `retrieval_plan` (3–7 semantically dense queries with weights and boost keywords).
- **LLM prompt version**: `query_v2`
- **Fallback**: heuristic queries derived from JD text when `use_jd_parser = false` or LLM call fails.

### B) Rewrite Agent

- **Input**: selected bullets + per-bullet allowlist + rewrite context (target profile summary, query plan summary, JD excerpt for tone).
- **LLM prompt version**: `rewrite_v2`
- **Rewrite style** (controlled via `rewrite_style` setting or per-request override):
  - `conservative` (default): light clarity edits; sentence structure kept close to original.
  - `creative`: stronger verbs, accomplishment-first framing, clause reordering — but no new facts.
- **Hard constraints** (both modes):
  - No new numbers, metrics, tools, companies, or claims.
  - LaTeX-ready output; original formatting preserved.
  - Length target: `rewrite_min_chars`–`rewrite_max_chars` (default 100–200 chars).
  - Invalid rewrites (similarity below threshold, new numbers/tools detected) are silently reverted to the original.

### C) Scoring Agent (Deterministic + Semantic)

- **Input**: JD, target profile, skills text, original + rewritten bullets, pre-computed retrieval signals.
- **LLM role**: produces **only** semantic gap analysis — missing keyword commentary, candidate boost terms, and a human-readable summary.
- **LLM prompt version**: `scoring_v2`
- **Final numeric score**: computed entirely by local deterministic code (`scoring_agent.py`), using:

| Signal | Description |
|--------|-------------|
| Retrieval score | How well selected bullets match the candidate pool |
| Coverage (bullets-only) | Must-have and nice-to-have keyword coverage in bullet text |
| Coverage (all) | Coverage including skills section |
| Length score | Fraction of bullets within the char target band |
| Redundancy penalty | Near-duplicate bullet pairs penalized |
| Quality score | Quantified signal bonus (metrics/numbers presence) |

The LLM does **not** author numeric scores.

---

## Unified Artifact Pipeline (`core/artifacts.py`)

Both the main generation loop and the manual re-render endpoint (`POST /runs/{run_id}/render`) call the same shared function:

```python
process_and_render_artifacts(
    settings, run_id, static_data, selected_ids, selected_candidates,
    *, rewritten_bullets=None, temp_overrides=None, base_report=None
) -> (pdf_path, tex_path, report_path, final_selected_ids, final_candidates)
```

Steps inside:
1. `build_tailored_snapshot` — filters and orders bullets from the static export.
2. `render_pdf` — renders Jinja2 LaTeX template → `.tex` → Tectonic → `.pdf`.
3. `trim_to_single_page` — iteratively drops lowest-scoring bullets until the PDF fits one page.
4. Write / update `_report.json` with final selections, artifacts, and rewrite metadata.

---

## Prompt versioning

| Prompt | Version | File |
|--------|---------|------|
| Query Agent | `query_v2` | `core/prompts/query.py` |
| Rewrite Agent | `rewrite_v2` | `core/prompts/rewrite.py` |
| Scoring Agent | `scoring_v2` | `core/prompts/scoring.py` |

Prompt versions are recorded in every `_report.json` under `prompt_versions` so you can trace which prompt stack produced a given run.

---

## Keyword canonicalization

`config/canonicalization.json` and `config/families.json` power deterministic keyword matching used in coverage scoring and keyword deduplication.

- **Canonicalization**: maps raw terms (e.g., `"k8s"`) to a canonical form (`"kubernetes"`).
- **Families**: groups related skills so that `"pytorch"` can satisfy a `"deep learning"` requirement.

These configs were significantly expanded in commit `7570eb3`.

---

## Report JSON schema

Per-run artifacts are written to `backend/output/`:

| File | Content |
|------|---------|
| `<run_id>.pdf` | Final tailored resume |
| `<run_id>.tex` | LaTeX source |
| `<run_id>_report.json` | Full explainability report |

`report.json` fields:

```jsonc
{
  "run_id": "...",
  "created_at": "...",
  "profile_used": "...",            // which target profile was used
  "target_profile_summary": "...",
  "prompt_versions": {              // query_v2 / rewrite_v2 / scoring_v2
    "query": "query_v2",
    "rewrite": "rewrite_v2",
    "scoring": "scoring_v2"
  },
  "rewrite_style": "conservative",  // or "creative"
  "agents": { "query": { "model": "...", "used": true, "fallback": false } },
  "best_iteration_index": 0,
  "selected_ids": ["exp:...", "proj:..."],
  "best_score": {
    "final_score": 85,
    "retrieval_score": 0.9,
    "coverage_bullets_only": 0.8,
    "coverage_all": 0.85,
    "length_score": 0.95,
    "redundancy_penalty": 0.0,
    "quality_score": 0.6,
    "must_missing_bullets_only": ["..."],
    "nice_missing_bullets_only": ["..."]
  },
  "scoring_semantic_feedback": {
    "summary": "...",
    "notes": ["..."],
    "candidate_boost_terms": ["..."]
  },
  "iterations": [{ "queries": [], "selected_ids": [], "score": {}, "rewrite_conditioning": {} }],
  "rewritten_bullets": [
    { "bullet_id": "...", "original_text": "...", "rewritten_text": "...",
      "changed": true, "fallback_used": false, "violations": [], "new_numbers": [], "new_tools": [] }
  ],
  "artifacts": { "pdf": "<run_id>.pdf", "tex": "<run_id>.tex" }
}
```

---

## Workflow diagram

```mermaid
flowchart LR
  UI[React UI] -->|REST| API[FastAPI]
  API -->|CRUD| DB[(SQLite)]
  DB -->|export| JSON[data/my_experience.json]
  JSON -->|ingest| CHROMA[(ChromaDB)]

  JD[Job description] --> QA[Query Agent\nquery_v2]
  QA --> PLAN[Target profile\n+ retrieval plan]
  PLAN --> RETRIEVE[Multi-query retrieve]
  CHROMA --> RETRIEVE
  RETRIEVE --> SELECT[Select Top-K]
  SELECT --> REWRITE[Rewrite Agent\nrewrite_v2\nconservative | creative]
  JD --> EXCERPT[JD excerpt - tone only]
  EXCERPT -..->|tone ref| REWRITE
  REWRITE --> SCORE[Scoring Agent\nscoring_v2\ndeterministic score\n+ LLM gap analysis]
  SCORE -->|boost terms| QA
  SCORE -->|best iteration| ARTIFACT[Unified Artifact Pipeline\nbuild_tailored_snapshot\nrender_pdf · trim · write_report]
  ARTIFACT --> OUT[PDF + TeX + report.json]
```

---

## Deployment notes

- **Local dev**: API at `http://localhost:8000`, UI at `http://localhost:5173`. Set `VITE_API_URL` if ports differ.
- **Docker Compose**: frontend container reaches backend at `http://api:8000` (internal Docker network); browser accesses `localhost:5173`. Set `VITE_API_URL=http://api:8000` in the frontend service environment.
- **Tectonic** is installed in the Docker API image. For local dev without Tectonic, set `skip_pdf = true`.

---

## Migration notes

| From | To | Notes |
|------|----|-------|
| Streamlit UI | React/Vite SPA | Removed in `635d953` |
| v3 naming | No suffix | Dropped in `128ad40`/`b97af07` |
| `gpt-4.1-nano` | `gpt-5.4-nano` | Default model updated in `f886350` |
| OpenAI Chat Completions | Responses API (`client.responses.parse`) | Migrated in `f886350` for structured output |
| Duplicate artifact logic | Unified `process_and_render_artifacts` | Both `/generate` and `/runs/*/render` use the same pipeline |
| Legacy `src/` root | `backend/src/` | Restructured in the react-refactor PR |
