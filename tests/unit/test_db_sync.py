import sys
import unittest
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from agentic_resume_tailor.db.base import Base
from agentic_resume_tailor.db.models import (
    Education,
    EducationBullet,
    Experience,
    ExperienceBullet,
    PersonalInfo,
    Project,
    ProjectBullet,
    Skills,
)
from agentic_resume_tailor.db.sync import export_resume_data


class TestDbSync(unittest.TestCase):
    def setUp(self) -> None:
        """Create an in-memory database session."""
        engine = create_engine("sqlite:///:memory:", future=True)
        Base.metadata.create_all(bind=engine)
        SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
        self.session = SessionLocal()

    def tearDown(self) -> None:
        """Close the database session."""
        self.session.close()

    def test_export_resume_data_orders_and_ids(self) -> None:
        """Test export respects sort order and stable ids."""
        self.session.add(PersonalInfo(name="Jane"))
        self.session.add(Skills(languages_frameworks="Python"))

        edu = Education(school="School", sort_order=2)
        edu_b1 = EducationBullet(text_latex="Top 5%", sort_order=2)
        edu_b2 = EducationBullet(text_latex="Dean's list", sort_order=1)
        edu.bullets.extend([edu_b1, edu_b2])

        exp1 = Experience(job_id="job_a", company="A", role="Eng", sort_order=2)
        exp2 = Experience(job_id="job_b", company="B", role="Eng", sort_order=1)
        exp1.bullets.append(ExperienceBullet(local_id="b01", text_latex="Did X", sort_order=2))
        exp1.bullets.append(ExperienceBullet(local_id="b02", text_latex="Did Y", sort_order=1))
        exp2.bullets.append(ExperienceBullet(local_id="b01", text_latex="Did Z", sort_order=1))

        proj = Project(project_id="proj_a", name="Proj", sort_order=1)
        proj.bullets.append(ProjectBullet(local_id="b01", text_latex="Built P", sort_order=1))

        self.session.add_all([edu, exp1, exp2, proj])
        self.session.commit()

        data = export_resume_data(self.session)
        self.assertEqual(data["schema_version"], "my_experience_v2")

        self.assertEqual(data["education"][0]["bullets"], ["Dean's list", "Top 5%"])
        self.assertEqual(data["experiences"][0]["job_id"], "job_b")
        self.assertEqual(data["experiences"][1]["job_id"], "job_a")
        self.assertEqual(
            data["experiences"][1]["bullets"],
            [
                {"id": "b02", "text_latex": "Did Y"},
                {"id": "b01", "text_latex": "Did X"},
            ],
        )
        self.assertEqual(data["projects"][0]["project_id"], "proj_a")


if __name__ == "__main__":
    unittest.main()
