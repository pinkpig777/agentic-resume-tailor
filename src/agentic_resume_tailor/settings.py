from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict

from dotenv import dotenv_values
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from agentic_resume_tailor.user_config import get_user_config_path, load_user_config


def _json_settings_source() -> Dict[str, Any]:
    """JSON settings source.

    Returns:
        Dictionary result.
    """
    return load_user_config(get_user_config_path())


def _limited_env_settings_source() -> Dict[str, Any]:
    """Limited env settings source.

    Returns:
        Dictionary result.
    """
    data: Dict[str, Any] = {}
    env: Dict[str, Any] = {}
    env.update(dotenv_values(".env"))
    env.update(os.environ)

    if "OPENAI_API_KEY" in env:
        data["openai_api_key"] = env["OPENAI_API_KEY"]
    if "PORT" in env:
        data["port"] = env["PORT"]
    return data


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    db_path: str = "data/processed/chroma_db"
    sql_db_url: str = "sqlite:///data/processed/resume.db"
    export_file: str = "data/my_experience.json"
    auto_reingest_on_save: bool = False
    template_dir: str = "templates"
    output_dir: str = "output"

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
    quant_bonus_per_hit: float = 0.05
    quant_bonus_cap: float = 0.20

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

    port: int = 8000
    openai_api_key: str | None = None

    @classmethod
    def settings_customise_sources(
        cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings
    ):
        """Settings customise sources.

        Args:
            settings_cls: The settings cls value.
            init_settings: The init settings value.
            env_settings: The env settings value.
            dotenv_settings: The dotenv settings value.
            file_secret_settings: The file secret settings value.
        """
        return (
            init_settings,
            _json_settings_source,
            _limited_env_settings_source,
            file_secret_settings,
        )


@lru_cache
def get_settings() -> Settings:
    """Get settings.

    Returns:
        Result value.
    """
    return Settings()
