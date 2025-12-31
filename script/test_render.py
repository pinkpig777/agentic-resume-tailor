import os
import subprocess

from jinja2 import Environment, FileSystemLoader


def render_resume(data):
    # Setup Jinja2 with custom delimiters to avoid clashing with LaTeX
    env = Environment(
        loader=FileSystemLoader("templates"),
        block_start_string="((%",
        block_end_string="%))",
        variable_start_string="<<",
        variable_end_string=">>",
        comment_start_string="((#",
        comment_end_string="#))",
    )

    template = env.get_template("resume.tex")
    rendered_tex = template.render(data)

    with open("output/temp_resume.tex", "w") as f:
        f.write(rendered_tex)

    print("Compiling PDF...")
    # --interaction=nonstopmode helps Tectonic/LaTeX finish even if there are small warnings
    result = subprocess.run(["tectonic", "output/temp_resume.tex"], capture_output=True, text=True)

    if result.returncode == 0:
        print("Success! Resume generated: output/temp_resume.pdf")
    else:
        print("Error during compilation:")
        print(result.stderr)


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
    if not os.path.exists("output"):
        os.makedirs("output")
    render_resume(sample_data)
