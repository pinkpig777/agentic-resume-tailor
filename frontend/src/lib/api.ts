import axios from "axios";

import type {
  Bullet,
  Experience,
  GenerateResponse,
  ResumeData,
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

export async function saveExperience(
  experience: Experience,
): Promise<Experience> {
  const { job_id, company, role, dates, location, sort_order } = experience;
  const { data } = await api.put(`/experiences/${job_id}`, {
    company,
    role,
    dates,
    location,
    sort_order,
  });
  return data;
}

export async function updateBullet(
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

export async function triggerIngest() {
  const { data } = await api.post("/admin/ingest");
  return data;
}

export async function generateResume(jdText: string): Promise<GenerateResponse> {
  const { data } = await api.post("/generate", { jd_text: jdText });
  return data;
}
