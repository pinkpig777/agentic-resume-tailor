from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="ART_", extra="ignore")

    db_path: str = "/app/data/processed/chroma_db"
    data_file: str = "/app/data/my_experience.json"
    template_dir: str = "/app/templates"
    output_dir: str = "/app/output"

    collection_name: str = "resume_experience"
    embed_model: str = "BAAI/bge-small-en-v1.5"

    use_jd_parser: bool = True

    max_bullets: int = 16
    per_query_k: int = 10
    final_k: int = 30

    max_iters: int = 3
    threshold: int = 80
    alpha: float = 0.7
    must_weight: float = 0.8

    boost_weight: float = 1.6
    boost_top_n_missing: int = 6

    cors_origins: str = "*"

    skip_pdf: bool = False
    run_id: str | None = None
    jd_model: str = "gpt-4.1-nano-2025-04-14"

    canon_config: str = "config/canonicalization.json"
    family_config: str = "config/families.json"

    api_url: str = "http://localhost:8000"

    log_level: str = "INFO"
    log_json: bool = False

    port: int = Field(default=8000, validation_alias="PORT")
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")


@lru_cache
def get_settings() -> Settings:
    return Settings()
