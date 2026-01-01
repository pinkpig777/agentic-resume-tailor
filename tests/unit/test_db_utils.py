import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from agentic_resume_tailor.db.utils import (
    ensure_unique_slug,
    make_job_id,
    make_project_id,
    next_bullet_id,
    primary_role,
)


class TestDbUtils(unittest.TestCase):
    def test_make_job_id_uses_primary_role(self) -> None:
        """Test job id uses primary role segment."""
        job_id = make_job_id("Acme Inc", "Engineer$|$Platform")
        self.assertEqual(job_id, "acme_inc__engineer")

    def test_make_project_id_slugifies(self) -> None:
        """Test project id slugify."""
        project_id = make_project_id("My Project")
        self.assertEqual(project_id, "my_project")

    def test_primary_role_default(self) -> None:
        """Test primary role default."""
        self.assertEqual(primary_role(None), "unknown")

    def test_next_bullet_id_skips_renumber(self) -> None:
        """Test next bullet id does not renumber existing ids."""
        next_id = next_bullet_id(["b01", "b03"])
        self.assertEqual(next_id, "b04")

    def test_next_bullet_id_empty(self) -> None:
        """Test next bullet id for empty list."""
        self.assertEqual(next_bullet_id([]), "b01")

    def test_ensure_unique_slug_suffix(self) -> None:
        """Test ensure unique slug appends suffix."""
        unique = ensure_unique_slug("proj", ["proj", "proj__2"])
        self.assertEqual(unique, "proj__3")


if __name__ == "__main__":
    unittest.main()
