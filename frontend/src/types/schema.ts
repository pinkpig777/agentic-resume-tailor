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

export interface SettingsData {
  db_path: string;
  sql_db_url: string;
  export_file: string;
  output_pdf_name: string | null;
  auto_reingest_on_save: boolean;
  template_dir: string;
  output_dir: string;
  collection_name: string;
  embed_model: string;
  use_jd_parser: boolean;
  max_bullets: number;
  per_query_k: number;
  final_k: number;
  max_iters: number;
  threshold: number;
  alpha: number;
  must_weight: number;
  quant_bonus_per_hit: number;
  quant_bonus_cap: number;
  boost_weight: number;
  boost_top_n_missing: number;
  cors_origins: string;
  skip_pdf: boolean;
  run_id: string | null;
  jd_model: string;
  canon_config: string;
  family_config: string;
  api_url: string;
  log_level: string;
  log_json: boolean;
  port: number;
  config_path: string;
}

export interface ExperienceCreatePayload {
  company: string;
  role: string;
  dates: string;
  location: string;
  sort_order?: number;
  bullets: string[];
}

export interface ProjectCreatePayload {
  name: string;
  technologies: string;
  sort_order?: number;
  bullets: string[];
}

export interface EducationCreatePayload {
  school: string;
  dates: string;
  degree: string;
  location: string;
  sort_order?: number;
  bullets: string[];
}

export type PersonalInfoUpdatePayload = Partial<PersonalInfo>;
export type SkillsUpdatePayload = Partial<Skills>;
export type ExperienceUpdatePayload = Partial<
  Pick<Experience, "company" | "role" | "dates" | "location" | "sort_order">
>;
export type ProjectUpdatePayload = Partial<
  Pick<Project, "name" | "technologies" | "sort_order">
>;
export type EducationUpdatePayload = Partial<
  Pick<Education, "school" | "dates" | "degree" | "location" | "sort_order" | "bullets">
>;
export type BulletCreatePayload = {
  text_latex: string;
  sort_order?: number;
};
export type BulletUpdatePayload = Partial<BulletCreatePayload>;

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

export interface TempAddition {
  temp_id?: string;
  parent_type: "experience" | "project";
  parent_id: string;
  text_latex: string;
  bullet_id?: string;
}

export interface TempOverrides {
  additions?: TempAddition[];
  edits?: Record<string, string>;
  removals?: string[];
}

export interface RunReport {
  run_id: string;
  selected_ids: string[];
  temp_additions?: TempAddition[];
  temp_edits?: Record<string, string>;
  temp_removals?: string[];
  artifacts?: {
    pdf?: string;
    tex?: string;
  };
}

export interface RenderSelectionResponse {
  status: string;
  run_id: string;
  pdf_url: string;
  tex_url: string;
  report_url: string;
}
