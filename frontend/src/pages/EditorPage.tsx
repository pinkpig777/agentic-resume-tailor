import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Download, FileText, Loader2, Plus, RefreshCcw } from "lucide-react";

import { EducationCard } from "@/components/EducationCard";
import { ExperienceCard } from "@/components/ExperienceCard";
import { ProjectCard } from "@/components/ProjectCard";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import {
  createEducation,
  createExperience,
  createExperienceBullet,
  createProject,
  createProjectBullet,
  deleteEducation,
  deleteExperience,
  deleteExperienceBullet,
  deleteProject,
  deleteProjectBullet,
  exportResume,
  fetchData,
  triggerIngest,
  updateEducation,
  updateExperience,
  updateExperienceBullet,
  updatePersonalInfo,
  updateProject,
  updateProjectBullet,
  updateSkills,
} from "@/lib/api";
import type {
  Bullet,
  Education,
  EducationCreatePayload,
  EducationUpdatePayload,
  Experience,
  ExperienceCreatePayload,
  ExperienceUpdatePayload,
  PersonalInfo,
  Project,
  ProjectCreatePayload,
  ProjectUpdatePayload,
  ResumeData,
  Skills,
} from "@/types/schema";

const emptyPersonalInfo: PersonalInfo = {
  name: "",
  phone: "",
  email: "",
  linkedin_id: "",
  github_id: "",
  linkedin: "",
  github: "",
};

const emptySkills: Skills = {
  languages_frameworks: "",
  ai_ml: "",
  db_tools: "",
};

const parseBullets = (value: string) =>
  value
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);

const sortBullets = (bullets: Bullet[]) =>
  [...bullets].sort(
    (a, b) => a.sort_order - b.sort_order || a.id.localeCompare(b.id),
  );

const sortEducation = (items: Education[]) =>
  [...items].sort(
    (a, b) => a.sort_order - b.sort_order || a.id - b.id,
  );

const sortExperiences = (items: Experience[]) =>
  [...items].sort(
    (a, b) => a.sort_order - b.sort_order || a.job_id.localeCompare(b.job_id),
  );

const sortProjects = (items: Project[]) =>
  [...items].sort(
    (a, b) =>
      a.sort_order - b.sort_order || a.project_id.localeCompare(b.project_id),
  );

type StatusTone = "success" | "error";

type StatusMessage = {
  tone: StatusTone;
  message: string;
};

type EducationDraft = Omit<EducationCreatePayload, "bullets"> & {
  bulletsText: string;
};

type ExperienceDraft = Omit<ExperienceCreatePayload, "bullets"> & {
  bulletsText: string;
};

type ProjectDraft = Omit<ProjectCreatePayload, "bullets"> & {
  bulletsText: string;
};

const emptyEducationDraft: EducationDraft = {
  school: "",
  degree: "",
  dates: "",
  location: "",
  bulletsText: "",
};

const emptyExperienceDraft: ExperienceDraft = {
  company: "",
  role: "",
  dates: "",
  location: "",
  bulletsText: "",
};

const emptyProjectDraft: ProjectDraft = {
  name: "",
  technologies: "",
  bulletsText: "",
};

export default function EditorPage() {
  const queryClient = useQueryClient();
  const [personalDraft, setPersonalDraft] = useState<PersonalInfo>(
    emptyPersonalInfo,
  );
  const [skillsDraft, setSkillsDraft] = useState<Skills>(emptySkills);
  const [newEducation, setNewEducation] = useState<EducationDraft>(
    emptyEducationDraft,
  );
  const [newExperience, setNewExperience] = useState<ExperienceDraft>(
    emptyExperienceDraft,
  );
  const [newProject, setNewProject] = useState<ProjectDraft>(
    emptyProjectDraft,
  );
  const [status, setStatus] = useState<StatusMessage | null>(null);

  const { data, isLoading, isError } = useQuery({
    queryKey: ["resumeData"],
    queryFn: fetchData,
  });

  useEffect(() => {
    if (!data) {
      return;
    }
    setPersonalDraft(data.personal_info);
    setSkillsDraft(data.skills);
  }, [data]);

  const updateResumeData = (updater: (current: ResumeData) => ResumeData) => {
    queryClient.setQueryData<ResumeData>(["resumeData"], (current) =>
      current ? updater(current) : current,
    );
  };

  const setError = (message: string) => {
    setStatus({ tone: "error", message });
  };

  const setSuccess = (message: string) => {
    setStatus({ tone: "success", message });
  };

  const updatePersonalInfoMutation = useMutation({
    mutationFn: updatePersonalInfo,
    onSuccess: (updated) => {
      updateResumeData((current) => ({
        ...current,
        personal_info: updated,
      }));
    },
    onError: () => setError("Failed to update personal info."),
  });

  const updateSkillsMutation = useMutation({
    mutationFn: updateSkills,
    onSuccess: (updated) => {
      updateResumeData((current) => ({
        ...current,
        skills: updated,
      }));
    },
    onError: () => setError("Failed to update skills."),
  });

  const createEducationMutation = useMutation({
    mutationFn: createEducation,
    onSuccess: (created) => {
      updateResumeData((current) => ({
        ...current,
        education: sortEducation([...current.education, created]),
      }));
      setNewEducation(emptyEducationDraft);
    },
    onError: () => setError("Failed to add education."),
  });

  const updateEducationMutation = useMutation({
    mutationFn: ({ id, payload }: { id: number; payload: EducationUpdatePayload }) =>
      updateEducation(id, payload),
    onSuccess: (updated) => {
      updateResumeData((current) => ({
        ...current,
        education: sortEducation(
          current.education.map((entry) =>
            entry.id === updated.id ? updated : entry,
          ),
        ),
      }));
    },
    onError: () => setError("Failed to update education."),
  });

  const deleteEducationMutation = useMutation({
    mutationFn: deleteEducation,
    onSuccess: (_result, educationId) => {
      updateResumeData((current) => ({
        ...current,
        education: current.education.filter((entry) => entry.id !== educationId),
      }));
    },
    onError: () => setError("Failed to delete education."),
  });

  const createExperienceMutation = useMutation({
    mutationFn: createExperience,
    onSuccess: (created) => {
      updateResumeData((current) => ({
        ...current,
        experiences: sortExperiences([...current.experiences, created]),
      }));
      setNewExperience(emptyExperienceDraft);
    },
    onError: () => setError("Failed to add experience."),
  });

  const updateExperienceMutation = useMutation({
    mutationFn: ({ jobId, payload }: { jobId: string; payload: ExperienceUpdatePayload }) =>
      updateExperience(jobId, payload),
    onSuccess: (updated, variables) => {
      updateResumeData((current) => ({
        ...current,
        experiences: sortExperiences(
          current.experiences.map((entry) =>
            entry.job_id === variables.jobId ? updated : entry,
          ),
        ),
      }));
    },
    onError: () => setError("Failed to update experience."),
  });

  const deleteExperienceMutation = useMutation({
    mutationFn: deleteExperience,
    onSuccess: (_result, jobId) => {
      updateResumeData((current) => ({
        ...current,
        experiences: current.experiences.filter((entry) => entry.job_id !== jobId),
      }));
    },
    onError: () => setError("Failed to delete experience."),
  });

  const createExperienceBulletMutation = useMutation({
    mutationFn: ({ jobId, text }: { jobId: string; text: string }) =>
      createExperienceBullet(jobId, { text_latex: text }),
    onSuccess: (created, variables) => {
      updateResumeData((current) => ({
        ...current,
        experiences: current.experiences.map((entry) =>
          entry.job_id === variables.jobId
            ? {
                ...entry,
                bullets: sortBullets([...entry.bullets, created]),
              }
            : entry,
        ),
      }));
    },
    onError: () => setError("Failed to add experience bullet."),
  });

  const updateExperienceBulletMutation = useMutation({
    mutationFn: ({ jobId, bullet }: { jobId: string; bullet: Bullet }) =>
      updateExperienceBullet(jobId, bullet),
    onSuccess: (updated, variables) => {
      updateResumeData((current) => ({
        ...current,
        experiences: current.experiences.map((entry) =>
          entry.job_id === variables.jobId
            ? {
                ...entry,
                bullets: sortBullets(
                  entry.bullets.map((bullet) =>
                    bullet.id === updated.id ? updated : bullet,
                  ),
                ),
              }
            : entry,
        ),
      }));
    },
    onError: () => setError("Failed to update experience bullet."),
  });

  const deleteExperienceBulletMutation = useMutation({
    mutationFn: ({ jobId, bulletId }: { jobId: string; bulletId: string }) =>
      deleteExperienceBullet(jobId, bulletId),
    onSuccess: (_result, variables) => {
      updateResumeData((current) => ({
        ...current,
        experiences: current.experiences.map((entry) =>
          entry.job_id === variables.jobId
            ? {
                ...entry,
                bullets: entry.bullets.filter(
                  (bullet) => bullet.id !== variables.bulletId,
                ),
              }
            : entry,
        ),
      }));
    },
    onError: () => setError("Failed to delete experience bullet."),
  });

  const createProjectMutation = useMutation({
    mutationFn: createProject,
    onSuccess: (created) => {
      updateResumeData((current) => ({
        ...current,
        projects: sortProjects([...current.projects, created]),
      }));
      setNewProject(emptyProjectDraft);
    },
    onError: () => setError("Failed to add project."),
  });

  const updateProjectMutation = useMutation({
    mutationFn: ({ projectId, payload }: { projectId: string; payload: ProjectUpdatePayload }) =>
      updateProject(projectId, payload),
    onSuccess: (updated, variables) => {
      updateResumeData((current) => ({
        ...current,
        projects: sortProjects(
          current.projects.map((entry) =>
            entry.project_id === variables.projectId ? updated : entry,
          ),
        ),
      }));
    },
    onError: () => setError("Failed to update project."),
  });

  const deleteProjectMutation = useMutation({
    mutationFn: deleteProject,
    onSuccess: (_result, projectId) => {
      updateResumeData((current) => ({
        ...current,
        projects: current.projects.filter(
          (entry) => entry.project_id !== projectId,
        ),
      }));
    },
    onError: () => setError("Failed to delete project."),
  });

  const createProjectBulletMutation = useMutation({
    mutationFn: ({ projectId, text }: { projectId: string; text: string }) =>
      createProjectBullet(projectId, { text_latex: text }),
    onSuccess: (created, variables) => {
      updateResumeData((current) => ({
        ...current,
        projects: current.projects.map((entry) =>
          entry.project_id === variables.projectId
            ? {
                ...entry,
                bullets: sortBullets([...entry.bullets, created]),
              }
            : entry,
        ),
      }));
    },
    onError: () => setError("Failed to add project bullet."),
  });

  const updateProjectBulletMutation = useMutation({
    mutationFn: ({ projectId, bullet }: { projectId: string; bullet: Bullet }) =>
      updateProjectBullet(projectId, bullet),
    onSuccess: (updated, variables) => {
      updateResumeData((current) => ({
        ...current,
        projects: current.projects.map((entry) =>
          entry.project_id === variables.projectId
            ? {
                ...entry,
                bullets: sortBullets(
                  entry.bullets.map((bullet) =>
                    bullet.id === updated.id ? updated : bullet,
                  ),
                ),
              }
            : entry,
        ),
      }));
    },
    onError: () => setError("Failed to update project bullet."),
  });

  const deleteProjectBulletMutation = useMutation({
    mutationFn: ({ projectId, bulletId }: { projectId: string; bulletId: string }) =>
      deleteProjectBullet(projectId, bulletId),
    onSuccess: (_result, variables) => {
      updateResumeData((current) => ({
        ...current,
        projects: current.projects.map((entry) =>
          entry.project_id === variables.projectId
            ? {
                ...entry,
                bullets: entry.bullets.filter(
                  (bullet) => bullet.id !== variables.bulletId,
                ),
              }
            : entry,
        ),
      }));
    },
    onError: () => setError("Failed to delete project bullet."),
  });

  const exportResumeMutation = useMutation({
    mutationFn: () => exportResume(false),
    onSuccess: (result) => {
      setSuccess(`Exported JSON to ${result.path}.`);
    },
    onError: () => setError("Failed to export JSON."),
  });

  const ingestMutation = useMutation({
    mutationFn: triggerIngest,
    onSuccess: (result) => {
      if (result.status === "ok") {
        setSuccess(
          `Re-ingested ${result.count} bullets in ${result.elapsed_s}s.`,
        );
      } else {
        setError(result.error ?? "Re-ingest failed.");
      }
    },
    onError: () => setError("Failed to re-ingest."),
  });

  const canAddEducation = useMemo(
    () => newEducation.school.trim().length > 0,
    [newEducation.school],
  );
  const canAddExperience = useMemo(
    () =>
      newExperience.company.trim().length > 0 &&
      newExperience.role.trim().length > 0,
    [newExperience.company, newExperience.role],
  );
  const canAddProject = useMemo(
    () => newProject.name.trim().length > 0,
    [newProject.name],
  );

  const handlePersonalBlur = (field: keyof PersonalInfo) => {
    if (!data) {
      return;
    }
    const next = personalDraft[field].trim();
    const current = data.personal_info[field];
    if (next === current) {
      return;
    }
    updatePersonalInfoMutation.mutate({ [field]: next });
  };

  const handleSkillsBlur = (field: keyof Skills) => {
    if (!data) {
      return;
    }
    const next = skillsDraft[field];
    const current = data.skills[field];
    if (next === current) {
      return;
    }
    updateSkillsMutation.mutate({ [field]: next });
  };

  const handleAddEducation = () => {
    if (!canAddEducation) {
      setError("School is required for education entries.");
      return;
    }
    const payload: EducationCreatePayload = {
      school: newEducation.school.trim(),
      degree: newEducation.degree.trim(),
      dates: newEducation.dates.trim(),
      location: newEducation.location.trim(),
      bullets: parseBullets(newEducation.bulletsText),
    };
    createEducationMutation.mutate(payload);
  };

  const handleAddExperience = () => {
    if (!canAddExperience) {
      setError("Company and role are required for experiences.");
      return;
    }
    const payload: ExperienceCreatePayload = {
      company: newExperience.company.trim(),
      role: newExperience.role.trim(),
      dates: newExperience.dates.trim(),
      location: newExperience.location.trim(),
      bullets: parseBullets(newExperience.bulletsText),
    };
    createExperienceMutation.mutate(payload);
  };

  const handleAddProject = () => {
    if (!canAddProject) {
      setError("Project name is required.");
      return;
    }
    const payload: ProjectCreatePayload = {
      name: newProject.name.trim(),
      technologies: newProject.technologies.trim(),
      bullets: parseBullets(newProject.bulletsText),
    };
    createProjectMutation.mutate(payload);
  };

  const handleExperienceReorder = async (jobId: string, bullets: Bullet[]) => {
    updateResumeData((current) => ({
      ...current,
      experiences: current.experiences.map((entry) =>
        entry.job_id === jobId
          ? { ...entry, bullets: sortBullets(bullets) }
          : entry,
      ),
    }));
    try {
      await Promise.all(
        bullets.map((bullet) =>
          updateExperienceBulletMutation.mutateAsync({ jobId, bullet }),
        ),
      );
    } catch {
      setError("Failed to reorder experience bullets.");
    }
  };

  const handleProjectReorder = async (projectId: string, bullets: Bullet[]) => {
    updateResumeData((current) => ({
      ...current,
      projects: current.projects.map((entry) =>
        entry.project_id === projectId
          ? { ...entry, bullets: sortBullets(bullets) }
          : entry,
      ),
    }));
    try {
      await Promise.all(
        bullets.map((bullet) =>
          updateProjectBulletMutation.mutateAsync({ projectId, bullet }),
        ),
      );
    } catch {
      setError("Failed to reorder project bullets.");
    }
  };

  if (isLoading) {
    return (
      <div className="space-y-6">
        <header className="space-y-2">
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.2em] text-muted-foreground">
            <FileText className="h-4 w-4" />
            Resume Editor
          </div>
          <h1 className="text-3xl font-semibold md:text-4xl">
            Keep your profile sharp.
          </h1>
          <p className="max-w-2xl text-sm text-muted-foreground">
            Loading your profile data from the FastAPI service.
          </p>
        </header>
        <Card>
          <CardContent className="py-10 text-center text-sm text-muted-foreground">
            Loading editor data...
          </CardContent>
        </Card>
      </div>
    );
  }

  if (isError || !data) {
    return (
      <div className="space-y-6">
        <header className="space-y-2">
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.2em] text-muted-foreground">
            <FileText className="h-4 w-4" />
            Resume Editor
          </div>
          <h1 className="text-3xl font-semibold md:text-4xl">
            Keep your profile sharp.
          </h1>
          <p className="max-w-2xl text-sm text-muted-foreground">
            The editor could not load resume data. Make sure the API is running.
          </p>
        </header>
        <Card>
          <CardContent className="py-10 text-center text-sm text-destructive">
            Failed to load profile data.
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <header className="space-y-3">
        <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.2em] text-muted-foreground">
          <FileText className="h-4 w-4" />
          Resume Editor
        </div>
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="space-y-2">
            <h1 className="text-3xl font-semibold md:text-4xl">
              Keep your profile sharp.
            </h1>
            <p className="max-w-2xl text-sm text-muted-foreground">
              Changes auto-save to the database on blur. Export JSON or re-ingest
              to refresh Chroma when you are ready.
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <Button
              variant="secondary"
              onClick={() => exportResumeMutation.mutate()}
              disabled={exportResumeMutation.isPending}
            >
              {exportResumeMutation.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Exporting
                </>
              ) : (
                <>
                  <Download className="h-4 w-4" />
                  Export JSON
                </>
              )}
            </Button>
            <Button
              variant="secondary"
              onClick={() => ingestMutation.mutate()}
              disabled={ingestMutation.isPending}
            >
              {ingestMutation.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Re-ingesting
                </>
              ) : (
                <>
                  <RefreshCcw className="h-4 w-4" />
                  Re-ingest
                </>
              )}
            </Button>
          </div>
        </div>
        {status ? (
          <div
            className={cn(
              "rounded-lg border px-3 py-2 text-sm",
              status.tone === "success"
                ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                : "border-destructive/30 bg-destructive/10 text-destructive",
            )}
          >
            {status.message}
          </div>
        ) : null}
      </header>

      <section className="space-y-4">
        <Card className="animate-rise">
          <CardHeader>
            <CardTitle>Personal info</CardTitle>
            <CardDescription>
              Keep contact details synced for every run.
            </CardDescription>
          </CardHeader>
          <CardContent className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="personal-name">Name</Label>
              <Input
                id="personal-name"
                value={personalDraft.name}
                onChange={(event) =>
                  setPersonalDraft((prev) => ({
                    ...prev,
                    name: event.target.value,
                  }))
                }
                onBlur={() => handlePersonalBlur("name")}
                placeholder="Full name"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="personal-email">Email</Label>
              <Input
                id="personal-email"
                value={personalDraft.email}
                onChange={(event) =>
                  setPersonalDraft((prev) => ({
                    ...prev,
                    email: event.target.value,
                  }))
                }
                onBlur={() => handlePersonalBlur("email")}
                placeholder="you@example.com"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="personal-phone">Phone</Label>
              <Input
                id="personal-phone"
                value={personalDraft.phone}
                onChange={(event) =>
                  setPersonalDraft((prev) => ({
                    ...prev,
                    phone: event.target.value,
                  }))
                }
                onBlur={() => handlePersonalBlur("phone")}
                placeholder="(555) 123-4567"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="personal-linkedin-id">LinkedIn handle</Label>
              <Input
                id="personal-linkedin-id"
                value={personalDraft.linkedin_id}
                onChange={(event) =>
                  setPersonalDraft((prev) => ({
                    ...prev,
                    linkedin_id: event.target.value,
                  }))
                }
                onBlur={() => handlePersonalBlur("linkedin_id")}
                placeholder="linkedin.com/in/username"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="personal-github-id">GitHub handle</Label>
              <Input
                id="personal-github-id"
                value={personalDraft.github_id}
                onChange={(event) =>
                  setPersonalDraft((prev) => ({
                    ...prev,
                    github_id: event.target.value,
                  }))
                }
                onBlur={() => handlePersonalBlur("github_id")}
                placeholder="github.com/username"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="personal-linkedin">LinkedIn URL</Label>
              <Input
                id="personal-linkedin"
                value={personalDraft.linkedin}
                onChange={(event) =>
                  setPersonalDraft((prev) => ({
                    ...prev,
                    linkedin: event.target.value,
                  }))
                }
                onBlur={() => handlePersonalBlur("linkedin")}
                placeholder="https://linkedin.com/in/username"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="personal-github">GitHub URL</Label>
              <Input
                id="personal-github"
                value={personalDraft.github}
                onChange={(event) =>
                  setPersonalDraft((prev) => ({
                    ...prev,
                    github: event.target.value,
                  }))
                }
                onBlur={() => handlePersonalBlur("github")}
                placeholder="https://github.com/username"
              />
            </div>
          </CardContent>
        </Card>

        <Card className="animate-rise animate-rise-delay-1">
          <CardHeader>
            <CardTitle>Skills</CardTitle>
            <CardDescription>
              Edit skills once and reuse across generated runs.
            </CardDescription>
          </CardHeader>
          <CardContent className="grid gap-4 md:grid-cols-3">
            <div className="space-y-2">
              <Label htmlFor="skills-languages">Languages & frameworks</Label>
              <Textarea
                id="skills-languages"
                value={skillsDraft.languages_frameworks}
                onChange={(event) =>
                  setSkillsDraft((prev) => ({
                    ...prev,
                    languages_frameworks: event.target.value,
                  }))
                }
                onBlur={() => handleSkillsBlur("languages_frameworks")}
                placeholder="Python, TypeScript, React"
                className="min-h-[120px]"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="skills-ml">AI / ML</Label>
              <Textarea
                id="skills-ml"
                value={skillsDraft.ai_ml}
                onChange={(event) =>
                  setSkillsDraft((prev) => ({
                    ...prev,
                    ai_ml: event.target.value,
                  }))
                }
                onBlur={() => handleSkillsBlur("ai_ml")}
                placeholder="Prompting, LLM ops, embeddings"
                className="min-h-[120px]"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="skills-db">DB / Tools</Label>
              <Textarea
                id="skills-db"
                value={skillsDraft.db_tools}
                onChange={(event) =>
                  setSkillsDraft((prev) => ({
                    ...prev,
                    db_tools: event.target.value,
                  }))
                }
                onBlur={() => handleSkillsBlur("db_tools")}
                placeholder="Postgres, Chroma, Docker"
                className="min-h-[120px]"
              />
            </div>
          </CardContent>
        </Card>
      </section>

      <section className="space-y-4">
        <div className="space-y-1">
          <h2 className="text-xl font-semibold">Education</h2>
          <p className="text-sm text-muted-foreground">
            Add or edit education history and highlights.
          </p>
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Add education</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="new-edu-school">School</Label>
                <Input
                  id="new-edu-school"
                  value={newEducation.school}
                  onChange={(event) =>
                    setNewEducation((prev) => ({
                      ...prev,
                      school: event.target.value,
                    }))
                  }
                  placeholder="University"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="new-edu-degree">Degree</Label>
                <Input
                  id="new-edu-degree"
                  value={newEducation.degree}
                  onChange={(event) =>
                    setNewEducation((prev) => ({
                      ...prev,
                      degree: event.target.value,
                    }))
                  }
                  placeholder="B.S. Computer Science"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="new-edu-dates">Dates</Label>
                <Input
                  id="new-edu-dates"
                  value={newEducation.dates}
                  onChange={(event) =>
                    setNewEducation((prev) => ({
                      ...prev,
                      dates: event.target.value,
                    }))
                  }
                  placeholder="2016 - 2020"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="new-edu-location">Location</Label>
                <Input
                  id="new-edu-location"
                  value={newEducation.location}
                  onChange={(event) =>
                    setNewEducation((prev) => ({
                      ...prev,
                      location: event.target.value,
                    }))
                  }
                  placeholder="City, Country"
                />
              </div>
            </div>
            <div className="space-y-2">
              <Label htmlFor="new-edu-bullets">Highlights</Label>
              <Textarea
                id="new-edu-bullets"
                value={newEducation.bulletsText}
                onChange={(event) =>
                  setNewEducation((prev) => ({
                    ...prev,
                    bulletsText: event.target.value,
                  }))
                }
                placeholder="One highlight per line"
              />
            </div>
            <div className="flex justify-end">
              <Button
                type="button"
                onClick={handleAddEducation}
                disabled={!canAddEducation || createEducationMutation.isPending}
              >
                {createEducationMutation.isPending ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Saving
                  </>
                ) : (
                  <>
                    <Plus className="h-4 w-4" />
                    Add education
                  </>
                )}
              </Button>
            </div>
          </CardContent>
        </Card>

        <div className="space-y-4">
          {data.education.length ? (
            data.education.map((entry) => (
              <EducationCard
                key={entry.id}
                education={entry}
                onUpdate={(educationId, payload) =>
                  updateEducationMutation.mutate({ id: educationId, payload })
                }
                onDelete={(educationId) =>
                  deleteEducationMutation.mutate(educationId)
                }
              />
            ))
          ) : (
            <Card className="border-dashed">
              <CardContent className="py-6 text-center text-sm text-muted-foreground">
                No education entries yet.
              </CardContent>
            </Card>
          )}
        </div>
      </section>

      <section className="space-y-4">
        <div className="space-y-1">
          <h2 className="text-xl font-semibold">Experience</h2>
          <p className="text-sm text-muted-foreground">
            Drag bullets to reorder and auto-save on blur.
          </p>
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Add experience</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="new-exp-company">Company</Label>
                <Input
                  id="new-exp-company"
                  value={newExperience.company}
                  onChange={(event) =>
                    setNewExperience((prev) => ({
                      ...prev,
                      company: event.target.value,
                    }))
                  }
                  placeholder="Company"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="new-exp-role">Role</Label>
                <Input
                  id="new-exp-role"
                  value={newExperience.role}
                  onChange={(event) =>
                    setNewExperience((prev) => ({
                      ...prev,
                      role: event.target.value,
                    }))
                  }
                  placeholder="Role title"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="new-exp-dates">Dates</Label>
                <Input
                  id="new-exp-dates"
                  value={newExperience.dates}
                  onChange={(event) =>
                    setNewExperience((prev) => ({
                      ...prev,
                      dates: event.target.value,
                    }))
                  }
                  placeholder="2021 - Present"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="new-exp-location">Location</Label>
                <Input
                  id="new-exp-location"
                  value={newExperience.location}
                  onChange={(event) =>
                    setNewExperience((prev) => ({
                      ...prev,
                      location: event.target.value,
                    }))
                  }
                  placeholder="City, Country"
                />
              </div>
            </div>
            <div className="space-y-2">
              <Label htmlFor="new-exp-bullets">Bullets</Label>
              <Textarea
                id="new-exp-bullets"
                value={newExperience.bulletsText}
                onChange={(event) =>
                  setNewExperience((prev) => ({
                    ...prev,
                    bulletsText: event.target.value,
                  }))
                }
                placeholder="One bullet per line"
              />
            </div>
            <div className="flex justify-end">
              <Button
                type="button"
                onClick={handleAddExperience}
                disabled={!canAddExperience || createExperienceMutation.isPending}
              >
                {createExperienceMutation.isPending ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Saving
                  </>
                ) : (
                  <>
                    <Plus className="h-4 w-4" />
                    Add experience
                  </>
                )}
              </Button>
            </div>
          </CardContent>
        </Card>

        <div className="space-y-4">
          {data.experiences.length ? (
            data.experiences.map((entry) => (
              <ExperienceCard
                key={entry.job_id}
                experience={entry}
                onExperienceUpdate={(jobId, payload) =>
                  updateExperienceMutation.mutate({ jobId, payload })
                }
                onExperienceDelete={(jobId) =>
                  deleteExperienceMutation.mutate(jobId)
                }
                onBulletCreate={(jobId, text) =>
                  createExperienceBulletMutation.mutate({ jobId, text })
                }
                onBulletUpdate={(jobId, bullet) =>
                  updateExperienceBulletMutation.mutate({ jobId, bullet })
                }
                onBulletDelete={(jobId, bulletId) =>
                  deleteExperienceBulletMutation.mutate({ jobId, bulletId })
                }
                onBulletsReorder={handleExperienceReorder}
              />
            ))
          ) : (
            <Card className="border-dashed">
              <CardContent className="py-6 text-center text-sm text-muted-foreground">
                No experience entries yet.
              </CardContent>
            </Card>
          )}
        </div>
      </section>

      <section className="space-y-4">
        <div className="space-y-1">
          <h2 className="text-xl font-semibold">Projects</h2>
          <p className="text-sm text-muted-foreground">
            Keep flagship projects ready to slot into the resume.
          </p>
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Add project</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="new-project-name">Name</Label>
                <Input
                  id="new-project-name"
                  value={newProject.name}
                  onChange={(event) =>
                    setNewProject((prev) => ({
                      ...prev,
                      name: event.target.value,
                    }))
                  }
                  placeholder="Project name"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="new-project-tech">Technologies</Label>
                <Input
                  id="new-project-tech"
                  value={newProject.technologies}
                  onChange={(event) =>
                    setNewProject((prev) => ({
                      ...prev,
                      technologies: event.target.value,
                    }))
                  }
                  placeholder="React, FastAPI, Postgres"
                />
              </div>
            </div>
            <div className="space-y-2">
              <Label htmlFor="new-project-bullets">Bullets</Label>
              <Textarea
                id="new-project-bullets"
                value={newProject.bulletsText}
                onChange={(event) =>
                  setNewProject((prev) => ({
                    ...prev,
                    bulletsText: event.target.value,
                  }))
                }
                placeholder="One bullet per line"
              />
            </div>
            <div className="flex justify-end">
              <Button
                type="button"
                onClick={handleAddProject}
                disabled={!canAddProject || createProjectMutation.isPending}
              >
                {createProjectMutation.isPending ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Saving
                  </>
                ) : (
                  <>
                    <Plus className="h-4 w-4" />
                    Add project
                  </>
                )}
              </Button>
            </div>
          </CardContent>
        </Card>

        <div className="space-y-4">
          {data.projects.length ? (
            data.projects.map((entry) => (
              <ProjectCard
                key={entry.project_id}
                project={entry}
                onProjectUpdate={(projectId, payload) =>
                  updateProjectMutation.mutate({ projectId, payload })
                }
                onProjectDelete={(projectId) =>
                  deleteProjectMutation.mutate(projectId)
                }
                onBulletCreate={(projectId, text) =>
                  createProjectBulletMutation.mutate({ projectId, text })
                }
                onBulletUpdate={(projectId, bullet) =>
                  updateProjectBulletMutation.mutate({ projectId, bullet })
                }
                onBulletDelete={(projectId, bulletId) =>
                  deleteProjectBulletMutation.mutate({ projectId, bulletId })
                }
                onBulletsReorder={handleProjectReorder}
              />
            ))
          ) : (
            <Card className="border-dashed">
              <CardContent className="py-6 text-center text-sm text-muted-foreground">
                No project entries yet.
              </CardContent>
            </Card>
          )}
        </div>
      </section>
    </div>
  );
}
