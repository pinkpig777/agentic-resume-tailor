import axios from "axios";

import type {
  Bullet,
  BulletCreatePayload,
  Education,
  EducationCreatePayload,
  EducationUpdatePayload,
  Experience,
  ExperienceCreatePayload,
  ExperienceUpdatePayload,
  GenerateResponse,
  PersonalInfo,
  PersonalInfoUpdatePayload,
  Project,
  ProjectCreatePayload,
  ProjectUpdatePayload,
  ResumeData,
  Skills,
  SkillsUpdatePayload,
} from "../types/schema";

export const API_BASE_URL =
  import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export const api = axios.create({
  baseURL: API_BASE_URL,
});

export async function fetchData(): Promise<ResumeData> {
  const [personalInfo, skills, education, experiences, projects] =
    await Promise.all([
      api.get("/personal_info"),
      api.get("/skills"),
      api.get("/education"),
      api.get("/experiences"),
      api.get("/projects"),
    ]);

  return {
    personal_info: personalInfo.data,
    skills: skills.data,
    education: education.data,
    experiences: experiences.data,
    projects: projects.data,
  };
}

export async function updatePersonalInfo(
  payload: PersonalInfoUpdatePayload,
): Promise<PersonalInfo> {
  const { data } = await api.put("/personal_info", payload);
  return data;
}

export async function updateSkills(
  payload: SkillsUpdatePayload,
): Promise<Skills> {
  const { data } = await api.put("/skills", payload);
  return data;
}

export async function createEducation(
  payload: EducationCreatePayload,
): Promise<Education> {
  const { data } = await api.post("/education", payload);
  return data;
}

export async function updateEducation(
  educationId: number,
  payload: EducationUpdatePayload,
): Promise<Education> {
  const { data } = await api.put(`/education/${educationId}`, payload);
  return data;
}

export async function deleteEducation(
  educationId: number,
): Promise<{ status: string; id: number }> {
  const { data } = await api.delete(`/education/${educationId}`);
  return data;
}

export async function createExperience(
  payload: ExperienceCreatePayload,
): Promise<Experience> {
  const { data } = await api.post("/experiences", payload);
  return data;
}

export async function updateExperience(
  jobId: string,
  payload: ExperienceUpdatePayload,
): Promise<Experience> {
  const { data } = await api.put(`/experiences/${jobId}`, payload);
  return data;
}

export async function deleteExperience(
  jobId: string,
): Promise<{ status: string; job_id: string }> {
  const { data } = await api.delete(`/experiences/${jobId}`);
  return data;
}

export async function createExperienceBullet(
  jobId: string,
  payload: BulletCreatePayload,
): Promise<Bullet> {
  const { data } = await api.post(`/experiences/${jobId}/bullets`, payload);
  return data;
}

export async function updateExperienceBullet(
  jobId: string,
  bullet: Bullet,
): Promise<Bullet> {
  const { data } = await api.put(
    `/experiences/${jobId}/bullets/${bullet.id}`,
    {
      text_latex: bullet.text_latex,
      sort_order: bullet.sort_order,
    },
  );
  return data;
}

export async function deleteExperienceBullet(
  jobId: string,
  localId: string,
): Promise<{ status: string; id: string }> {
  const { data } = await api.delete(
    `/experiences/${jobId}/bullets/${localId}`,
  );
  return data;
}

export async function createProject(
  payload: ProjectCreatePayload,
): Promise<Project> {
  const { data } = await api.post("/projects", payload);
  return data;
}

export async function updateProject(
  projectId: string,
  payload: ProjectUpdatePayload,
): Promise<Project> {
  const { data } = await api.put(`/projects/${projectId}`, payload);
  return data;
}

export async function deleteProject(
  projectId: string,
): Promise<{ status: string; project_id: string }> {
  const { data } = await api.delete(`/projects/${projectId}`);
  return data;
}

export async function createProjectBullet(
  projectId: string,
  payload: BulletCreatePayload,
): Promise<Bullet> {
  const { data } = await api.post(`/projects/${projectId}/bullets`, payload);
  return data;
}

export async function updateProjectBullet(
  projectId: string,
  bullet: Bullet,
): Promise<Bullet> {
  const { data } = await api.put(
    `/projects/${projectId}/bullets/${bullet.id}`,
    {
      text_latex: bullet.text_latex,
      sort_order: bullet.sort_order,
    },
  );
  return data;
}

export async function deleteProjectBullet(
  projectId: string,
  localId: string,
): Promise<{ status: string; id: string }> {
  const { data } = await api.delete(
    `/projects/${projectId}/bullets/${localId}`,
  );
  return data;
}

export async function exportResume(reingest = false): Promise<{
  status: string;
  path: string;
  reingested: boolean;
}> {
  const { data } = await api.post("/admin/export", null, {
    params: { reingest },
  });
  return data;
}

export async function triggerIngest(): Promise<{
  status: string;
  count: number;
  elapsed_s: number;
  error?: string;
}> {
  const { data } = await api.post("/admin/ingest");
  return data;
}

export async function generateResume(jdText: string): Promise<GenerateResponse> {
  const { data } = await api.post("/generate", { jd_text: jdText });
  return data;
}
