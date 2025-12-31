from sqlalchemy import Column, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import relationship

from agentic_resume_tailor.db.base import Base


class PersonalInfo(Base):
    __tablename__ = "personal_info"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False, default="")
    phone = Column(String(64), nullable=False, default="")
    email = Column(String(255), nullable=False, default="")
    linkedin_id = Column(String(255), nullable=False, default="")
    github_id = Column(String(255), nullable=False, default="")
    linkedin = Column(String(255), nullable=False, default="")
    github = Column(String(255), nullable=False, default="")


class Skills(Base):
    __tablename__ = "skills"

    id = Column(Integer, primary_key=True)
    languages_frameworks = Column(Text, nullable=False, default="")
    ai_ml = Column(Text, nullable=False, default="")
    db_tools = Column(Text, nullable=False, default="")


class Education(Base):
    __tablename__ = "education"

    id = Column(Integer, primary_key=True)
    school = Column(String(255), nullable=False, default="")
    dates = Column(String(255), nullable=False, default="")
    degree = Column(String(255), nullable=False, default="")
    location = Column(String(255), nullable=False, default="")
    sort_order = Column(Integer, nullable=False, default=0)

    bullets = relationship(
        "EducationBullet",
        back_populates="education",
        cascade="all, delete-orphan",
        order_by="EducationBullet.sort_order",
    )


class EducationBullet(Base):
    __tablename__ = "education_bullets"

    id = Column(Integer, primary_key=True)
    education_id = Column(Integer, ForeignKey("education.id"), nullable=False)
    text_latex = Column(Text, nullable=False, default="")
    sort_order = Column(Integer, nullable=False, default=0)

    education = relationship("Education", back_populates="bullets")


class Experience(Base):
    __tablename__ = "experiences"

    id = Column(Integer, primary_key=True)
    job_id = Column(String(255), nullable=False, unique=True)
    company = Column(String(255), nullable=False, default="")
    role = Column(String(255), nullable=False, default="")
    dates = Column(String(255), nullable=False, default="")
    location = Column(String(255), nullable=False, default="")
    sort_order = Column(Integer, nullable=False, default=0)

    bullets = relationship(
        "ExperienceBullet",
        back_populates="experience",
        cascade="all, delete-orphan",
        order_by="ExperienceBullet.sort_order",
    )


class ExperienceBullet(Base):
    __tablename__ = "experience_bullets"
    __table_args__ = (UniqueConstraint("experience_id", "local_id", name="uq_exp_bullet"),)

    id = Column(Integer, primary_key=True)
    experience_id = Column(Integer, ForeignKey("experiences.id"), nullable=False)
    local_id = Column(String(16), nullable=False)
    text_latex = Column(Text, nullable=False, default="")
    sort_order = Column(Integer, nullable=False, default=0)

    experience = relationship("Experience", back_populates="bullets")


class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True)
    project_id = Column(String(255), nullable=False, unique=True)
    name = Column(String(255), nullable=False, default="")
    technologies = Column(Text, nullable=False, default="")
    sort_order = Column(Integer, nullable=False, default=0)

    bullets = relationship(
        "ProjectBullet",
        back_populates="project",
        cascade="all, delete-orphan",
        order_by="ProjectBullet.sort_order",
    )


class ProjectBullet(Base):
    __tablename__ = "project_bullets"
    __table_args__ = (UniqueConstraint("project_id", "local_id", name="uq_proj_bullet"),)

    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    local_id = Column(String(16), nullable=False)
    text_latex = Column(Text, nullable=False, default="")
    sort_order = Column(Integer, nullable=False, default=0)

    project = relationship("Project", back_populates="bullets")
