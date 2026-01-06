export interface Bullet {
  id: string;
  text_latex: string;
  sort_order: number;
}

export interface Experience {
  job_id: string;
  company: string;
  role: string;
  dates: string;
  location: string;
  sort_order: number;
  bullets: Bullet[];
}

export interface Project {
  project_id: string;
  name: string;
  technologies: string;
  sort_order: number;
  bullets: Bullet[];
}

export interface PersonalInfo {
  name: string;
  phone: string;
  email: string;
  linkedin_id: string;
  github_id: string;
  linkedin: string;
  github: string;
}

export interface Skills {
  languages_frameworks: string;
  ai_ml: string;
  db_tools: string;
}

export interface Education {
  id: number;
  school: string;
  dates: string;
  degree: string;
  location: string;
  sort_order: number;
  bullets: string[];
}

export interface ResumeData {
  personal_info: PersonalInfo;
  skills: Skills;
  education: Education[];
  experiences: Experience[];
  projects: Project[];
}

export interface GenerateResponse {
  run_id: string;
  profile_used: boolean;
  best_iteration_index: number;
  pdf_url: string;
  tex_url: string;
  report_url: string;
}
