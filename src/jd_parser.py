import os
import json
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List

# Define the structure we WANT the AI to give us


class JobRequirements(BaseModel):
    hard_skills: List[str] = Field(
        description="List of technical skills, languages, and tools mentioned.")
    soft_skills: List[str] = Field(
        description="List of soft skills like leadership, communication, etc.")
    experience_queries: List[str] = Field(
        description="3-5 specific search queries to find relevant experience in a resume DB.")


def parse_job_description(jd_text):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    print("🤖 Analyzing Job Description...")

    completion = client.beta.chat.completions.parse(
        model="gpt-5-nano-2025-08-07",  # Or gpt-4o-mini
        messages=[
            {"role": "system", "content": "You are an expert technical recruiter in software industry. Extract key requirements from job descriptions to help tailor a resume."},
            {"role": "user", "content": f"Analyze this job description and extract the requirements:\n\n{jd_text}"}
        ],
        response_format=JobRequirements,
    )

    result = completion.choices[0].message.parsed
    return result


if __name__ == "__main__":
    # Test with a fake JD
    sample_jd = """
    We are looking for a Python Engineer with experience in Computer Vision.
    Must know PyTorch, OpenCV, and have experience deploying models to edge devices.
    Bonus if you have worked with Docker and CI/CD pipelines.
    """

    data = parse_job_description(sample_jd)
    print(json.dumps(data.model_dump(), indent=2))
