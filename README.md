# 📄 AI Resume Agent

A Local, Privacy-First AI Agent that generates tailored resumes for specific job descriptions using **Docker**, **Python (RAG)**, and **LaTeX (Tectonic)**.

## 🚀 Project Status
- [x] **Phase 1: The Engine** - Dockerized LaTeX rendering environment (Tectonic).
- [x] **Phase 2: The Brain (In Progress)** - Parsing experience data into Vector DB (ChromaDB).
- [ ] **Phase 3: The Agent** - LLM integration to select bullets based on Job Description.

---

## 🛠️ Prerequisites
- **Docker Desktop** (Running)
- **Git**
- **VS Code** (Recommended)

---

## 📂 Project Structure

```text
resume-agent/
├── data/
│   ├── my_experience.json    # Your REAL data (Gitignored!)
│   └── processed/            # Where Vector DB stores data
├── output/                   # Generated PDFs go here
├── src/
│   ├── main.py               # Entry point (Coming soon)
│   ├── test_render.py        # Test script to generate PDF
│   └── ingest.py             # Script to load JSON into ChromaDB
├── templates/
│   └── resume.tex            # Jinja2-ready LaTeX template
├── .env                      # API Keys (Gitignored)
├── .gitignore                # Security rules
├── Dockerfile                # Multi-arch build instructions
├── requirements.txt          # Python dependencies
└── README.md                 # This file

⚡ Quick Start
1. Setup Data
Create a folder data/ and a file data/my_experience.json. Do not commit this file to GitHub!

Example format:
```json
{
  "personal_info": { "name": "Alice Bob", ... },
  "education": [ ... ],
  "experiences": [ ... ],
  "projects": [ ... ]
}
```


2. Build the Docker Image
Run this whenever you change requirements.txt or Dockerfile.

```bash

docker build -t resume-agent .

```

🖥️ Usage Commands
We use Docker Volumes to map your local folders into the container. This allows you to edit code locally and run it instantly without rebuilding.

1. Generate a Test Resume (The Engine)
This uses dummy data (or your hardcoded test data) to prove the LaTeX engine works.
```bash
docker run --rm \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/src:/app/src \
  -v $(pwd)/templates:/app/templates \
  resume-agent python src/test_render.py
```

2. Ingest Data into "The Brain" (RAG)
This reads your data/my_experience.json and stores it in the local Vector Database (ChromaDB).

```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/src:/app/src \
  resume-agent python src/ingest.py
```
Output: You should see Successfully stored X resume bullet points.

⚠️ Troubleshooting
1. "Unable to locate package libicu..."

Fix: Ensure your Dockerfile uses generic package names (libicu-dev) rather than specific versions. Rebuild the image.

2. "Undefined control sequence \titrule"

Fix: It's a typo in resume.tex. Change it to \titlerule.

3. "Forbidden control sequence... \check@nocorr@"

Fix: You are using \\ or \textbf{} incorrectly inside a list. Remove manual newlines inside itemize.

4. "Missing end of comment tag"

Fix: Jinja2 is clashing with LaTeX. Ensure src/test_render.py sets custom delimiters (<< >>, ((% %))).

🔒 Security Note
Never commit data/my_experience.json.

Never commit .env.

The .gitignore file is configured to prevent this, but always double-check before pushing.
