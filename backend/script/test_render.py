from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any, Dict

from jinja2 import Environment, FileSystemLoader, TemplateNotFound

from agentic_resume_tailor.settings import get_settings
from agentic_resume_tailor.utils.logging import configure_logging

configure_logging()
logger = logging.getLogger(__name__)
settings = get_settings()


def _build_env(template_dir: Path) -> Environment:
    """Build env.

    Args:
        template_dir: Directory path for template.

    Returns:
        Result value.
    """
    # Custom delimiters avoid clashing with LaTeX syntax.
    return Environment(
        loader=FileSystemLoader(str(template_dir)),
        block_start_string="((%",
        block_end_string="%))",
        variable_start_string="<<",
        variable_end_string=">>",
        comment_start_string="((#",
        comment_end_string="#))",
    )


def render_resume(data: Dict[str, Any], template_dir: Path, output_dir: Path) -> Path:
    """Render resume.

    Args:
        data: The data value.
        template_dir: Directory path for template.
        output_dir: Output directory path.

    Returns:
        Result value.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    env = _build_env(template_dir)
    try:
        template = env.get_template("resume.tex")
    except TemplateNotFound as exc:
        raise SystemExit(f"Template not found: {template_dir / 'resume.tex'}") from exc

    rendered_tex = template.render(data)
    tex_path = output_dir / "temp_resume.tex"
    tex_path.write_text(rendered_tex, encoding="utf-8")
    return tex_path


def compile_pdf(tex_path: Path, output_dir: Path) -> Path:
    """Compile PDF.

    Args:
        tex_path: Filesystem path for TeX.
        output_dir: Output directory path.

    Returns:
        Result value.
    """
    # --interaction=nonstopmode helps Tectonic/LaTeX finish even if there are small warnings
    result = subprocess.run(
        ["tectonic", str(tex_path), "--outdir", str(output_dir)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error("Tectonic compilation failed: %s", result.stderr.strip())
        raise SystemExit(1)
    return output_dir / "temp_resume.pdf"


# Mock Data (This is what your RAG Agent will eventually generate)
sample_data = {
    "personal_info": {
        "name": "Hsiang-Chen (Charlie) Chiu",
        "phone": "(346)-531-2146",
        "email": "charly729.chiu@gmail.com",
        "linkedin": "https://linkedin.com/in/charliechiu0729",
        "linkedin_id": "charliechiu0729",
        "github": "https://github.com/pinkpig777",
        "github_id": "pinkpig777",
    },
    "skills": {
        "languages_frameworks": "Python, C++, TypeScript, React, FastAPI",
        "ai_ml": "PyTorch, LangChain, LangGraph",
        "db_tools": "PostgreSQL, Docker, Redis",
    },
    "education": [
        {
            "school": "Texas A\&M University",
            "dates": "Aug 2025 -- Dec 2026",
            "degree": "Master of Computer Science",
            "location": "TX",
        }
    ],
    "experiences": [
        {
            "company": "SaturnAI",
            "dates": "Jun 2025 -- Aug 2025",
            "role": "AI Software Engineer",
            "location": "Taiwan",
            "bullets": [
                "Engineered real-time monitoring systems.",
                "Scaled ingestion to 30 concurrent streams.",
            ],
        }
    ],
    "projects": [
        {
            "name": "Agentic Resume Tailor",
            "technologies": "LangGraph, Python, Docker",
            "bullets": ["Built a multi-agent system for automated resume optimization."],
        }
    ],
}

if __name__ == "__main__":
    template_dir = Path(settings.template_dir)
    output_dir = Path(settings.output_dir)
    tex_path = render_resume(sample_data, template_dir, output_dir)
    pdf_path = compile_pdf(tex_path, output_dir)
    logger.info("Resume generated: %s", pdf_path)
