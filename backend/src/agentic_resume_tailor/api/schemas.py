from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field


class TempAddition(BaseModel):
    parent_type: Literal["experience", "project"]
    parent_id: str = Field(min_length=1)
    text_latex: str = Field(min_length=1)
    temp_id: str | None = None


class TempOverrides(BaseModel):
    edits: Dict[str, str] = Field(default_factory=dict)
    removals: List[str] = Field(default_factory=list)
    additions: List[TempAddition] = Field(default_factory=list)


class GenerateRequest(BaseModel):
    jd_text: str = Field(min_length=1)
    run_id: str | None = Field(default=None, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")

    max_bullets: int | None = Field(default=None, ge=4, le=32)
    per_query_k: int | None = Field(default=None, ge=1, le=50)
    final_k: int | None = Field(default=None, ge=5, le=200)

    max_iters: int | None = Field(default=None, ge=1, le=6)
    threshold: int | None = Field(default=None, ge=0, le=100)

    alpha: float | None = Field(default=None, ge=0.0, le=1.0)
    must_weight: float | None = Field(default=None, ge=0.0, le=1.0)

    boost_weight: float | None = Field(default=None, ge=0.1, le=3.0)
    boost_top_n_missing: int | None = Field(default=None, ge=1, le=20)

    enable_bullet_rewrite: bool | None = None
    rewrite_style: Literal["conservative", "creative"] | None = None


class GenerateResponse(BaseModel):
    run_id: str
    profile_used: bool
    best_iteration_index: int
    pdf_url: str
    tex_url: str
    report_url: str


class GenerateV3Request(GenerateRequest):
    """Deprecated. Use GenerateRequest."""


class GenerateV3Response(GenerateResponse):
    """Deprecated. Use GenerateResponse."""


class RenderSelectionRequest(BaseModel):
    selected_ids: List[str] = Field(default_factory=list)
    temp_overrides: TempOverrides | None = None
    rewritten_bullets: Dict[str, str] | None = None


class PersonalInfoUpdate(BaseModel):
    name: str | None = None
    phone: str | None = None
    email: str | None = None
    linkedin_id: str | None = None
    github_id: str | None = None
    linkedin: str | None = None
    github: str | None = None


class SkillsUpdate(BaseModel):
    languages_frameworks: str | None = None
    ai_ml: str | None = None
    db_tools: str | None = None


class ExperienceCreate(BaseModel):
    company: str = Field(min_length=1)
    role: str = Field(min_length=1)
    dates: str = ""
    location: str = ""
    sort_order: int | None = None
    bullets: List[str] = Field(default_factory=list)


class ExperienceUpdate(BaseModel):
    company: str | None = None
    role: str | None = None
    dates: str | None = None
    location: str | None = None
    sort_order: int | None = None


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1)
    technologies: str = ""
    sort_order: int | None = None
    bullets: List[str] = Field(default_factory=list)


class ProjectUpdate(BaseModel):
    name: str | None = None
    technologies: str | None = None
    sort_order: int | None = None


class BulletCreate(BaseModel):
    text_latex: str = Field(min_length=1)
    sort_order: int | None = None


class BulletUpdate(BaseModel):
    text_latex: str | None = None
    sort_order: int | None = None


class EducationCreate(BaseModel):
    school: str = Field(min_length=1)
    dates: str = ""
    degree: str = ""
    location: str = ""
    bullets: List[str] = Field(default_factory=list)
    sort_order: int | None = None


class EducationUpdate(BaseModel):
    school: str | None = None
    dates: str | None = None
    degree: str | None = None
    location: str | None = None
    bullets: List[str] | None = None
    sort_order: int | None = None


class SettingsResponse(BaseModel):
    config_path: str
    live_fields: List[str]
    restart_required_fields: List[str]
    values: Dict[str, Any]
