from agentic_resume_tailor.db.utils import (
    ensure_unique_slug,
    make_job_id,
    make_project_id,
    next_bullet_id,
    primary_role,
)


def test_make_job_id_uses_primary_role() -> None:
    """Test job id uses primary role segment."""
    job_id = make_job_id("Acme Inc", "Engineer$|$Platform")
    assert job_id == "acme_inc__engineer"


def test_make_project_id_slugifies() -> None:
    """Test project id slugify."""
    project_id = make_project_id("My Project")
    assert project_id == "my_project"


def test_primary_role_default() -> None:
    """Test primary role default."""
    assert primary_role(None) == "unknown"


def test_next_bullet_id_skips_renumber() -> None:
    """Test next bullet id does not renumber existing ids."""
    next_id = next_bullet_id(["b01", "b03"])
    assert next_id == "b04"


def test_next_bullet_id_empty() -> None:
    """Test next bullet id for empty list."""
    assert next_bullet_id([]) == "b01"


def test_ensure_unique_slug_suffix() -> None:
    """Test ensure unique slug appends suffix."""
    unique = ensure_unique_slug("proj", ["proj", "proj__2"])
    assert unique == "proj__3"
