import time
from typing import Any, Dict, Tuple

import pandas as pd
import requests
import streamlit as st

from agentic_resume_tailor.settings import get_settings


# ----------------------------
# Health check
# ----------------------------
def check_server_health(api_base: str, timeout_s: float = 1.5) -> Tuple[bool, Any]:
    """
    Returns (ok: bool, payload_or_error: dict|str)
    """
    url = api_base.rstrip("/") + "/health"
    try:
        r = requests.get(url, timeout=timeout_s)
        if r.status_code != 200:
            return False, f"HTTP {r.status_code}: {r.text[:200]}"
        return True, r.json()
    except Exception as e:
        return False, str(e)


def get_health_cached(api_base: str, ttl_s: float = 2.0) -> Tuple[bool, Any]:
    now = time.time()
    last = st.session_state.get("_health_last_checked", 0.0)
    force = st.session_state.get("_health_force_refresh", 0.0)

    if (now - last) > ttl_s or force > last:
        ok, info = check_server_health(api_base)
        st.session_state["_health_ok"] = ok
        st.session_state["_health_info"] = info
        st.session_state["_health_last_checked"] = now

    return st.session_state.get("_health_ok", False), st.session_state.get(
        "_health_info", "not checked"
    )


def api_request(
    method: str, api_base: str, path: str, timeout_s: float = 10.0, **kwargs: Any
) -> Tuple[bool, Any, str]:
    url = api_base.rstrip("/") + path
    try:
        resp = requests.request(method, url, timeout=timeout_s, **kwargs)
    except Exception as exc:
        return False, None, str(exc)

    if resp.status_code >= 400:
        return False, None, f"HTTP {resp.status_code}: {resp.text.strip()}"

    if resp.text:
        try:
            return True, resp.json(), ""
        except Exception:
            return True, resp.text, ""
    return True, None, ""


def _fetch_app_settings(api_url: str) -> Tuple[bool, Dict[str, Any], str]:
    ok, data, err = api_request("GET", api_url, "/settings", timeout_s=10)
    if ok and isinstance(data, dict):
        return True, data, ""
    return False, {}, err or "Failed to load settings."


def _set_editor_message(level: str, text: str) -> None:
    st.session_state["editor_message"] = {"level": level, "text": text}


def _show_editor_message() -> None:
    msg = st.session_state.pop("editor_message", None)
    if not msg:
        return
    level = msg.get("level", "info")
    text = msg.get("text", "")
    if level == "success":
        st.success(text)
    elif level == "error":
        st.error(text)
    else:
        st.info(text)


def _render_bullet_controls(
    api_url: str, section: str, parent_id: str, bullet: dict
) -> None:
    bullet_id = bullet.get("id", "")
    text = bullet.get("text_latex", "")
    edit_key = f"edit_{section}_{parent_id}_{bullet_id}"
    text_key = f"text_{section}_{parent_id}_{bullet_id}"

    col_text, col_edit, col_del = st.columns([8, 1, 1])
    col_text.write(f"`{bullet_id}` {text}")
    if col_edit.button("Edit", key=f"edit_btn_{edit_key}"):
        st.session_state[edit_key] = not st.session_state.get(edit_key, False)
    if col_del.button("Delete", key=f"del_btn_{edit_key}"):
        ok, _, err = api_request(
            "DELETE", api_url, f"/{section}/{parent_id}/bullets/{bullet_id}"
        )
        if ok:
            _set_editor_message("success", f"Deleted {section} bullet {bullet_id}.")
            st.rerun()
        else:
            st.error(err)

    if st.session_state.get(edit_key, False):
        new_text = st.text_area("Bullet text", value=text, key=text_key, height=90)
        if st.button("Save", key=f"save_{edit_key}"):
            if not new_text.strip():
                st.error("Bullet text cannot be empty.")
            else:
                ok, _, err = api_request(
                    "PUT",
                    api_url,
                    f"/{section}/{parent_id}/bullets/{bullet_id}",
                    json={"text_latex": new_text},
                )
                if ok:
                    st.session_state[edit_key] = False
                    _set_editor_message("success", f"Updated {section} bullet {bullet_id}.")
                    st.rerun()
                else:
                    st.error(err)


def render_health_sidebar(api_url: str) -> Tuple[bool, Any]:
    ok, info = get_health_cached(api_url)
    icon = "🟢" if ok else "🔴"
    st.sidebar.markdown(f"**Server** {icon}")
    return ok, info


def render_settings_page(api_url: str) -> None:
    st.header("Settings")

    ok, app_settings, err = _fetch_app_settings(api_url)
    if not ok:
        st.error(f"Failed to load settings: {err}")
        return

    with st.form("app_settings_form"):
        st.subheader("Basics")
        col1, col2 = st.columns(2)
        with col1:
            auto_reingest = st.checkbox(
                "Auto re-ingest on save",
                value=bool(app_settings.get("auto_reingest_on_save", False)),
                help="Automatically export and re-ingest Chroma after any profile edit.",
            )
            export_file = st.text_input(
                "Export file path",
                value=app_settings.get("export_file", ""),
                help="Path for the exported resume JSON (written on saves and ingest).",
            )
        with col2:
            use_jd_parser = st.checkbox(
                "Enable JD parser",
                value=bool(app_settings.get("use_jd_parser", True)),
                help="Use the LLM-based JD parser to build retrieval queries.",
            )
            jd_model = st.text_input(
                "JD model",
                value=app_settings.get("jd_model", ""),
                help="OpenAI model name for JD parsing.",
            )
            skip_pdf = st.checkbox(
                "Skip PDF render (dev/testing)",
                value=bool(app_settings.get("skip_pdf", False)),
                help="When enabled, generation skips the LaTeX/PDF step.",
            )

        st.subheader("Generation defaults")
        colA, colB = st.columns(2)
        with colA:
            max_iters = st.number_input(
                "Max loop iterations",
                min_value=1,
                max_value=10,
                value=int(app_settings.get("max_iters", 3) or 3),
                step=1,
                help="Maximum agent iterations before stopping.",
            )
            per_query_k = st.number_input(
                "per_query_k",
                min_value=1,
                max_value=50,
                value=int(app_settings.get("per_query_k", 10) or 10),
                step=1,
                help="Top-K results retrieved per query before merging.",
            )
        with colB:
            max_bullets = st.number_input(
                "Max bullets on page",
                min_value=4,
                max_value=32,
                value=int(app_settings.get("max_bullets", 16) or 16),
                step=1,
                help="Maximum bullets selected for the final resume.",
            )
            final_k = st.number_input(
                "final_k",
                min_value=5,
                max_value=200,
                value=int(app_settings.get("final_k", 30) or 30),
                step=1,
                help="Final candidate pool size before selection.",
            )

        with st.expander("Advanced tuning", expanded=False):
            threshold = st.number_input(
                "Stop threshold (final score)",
                min_value=0,
                max_value=100,
                value=int(app_settings.get("threshold", 80) or 80),
                step=1,
                help="Stop early if the hybrid score reaches this value.",
            )
            alpha = st.number_input(
                "Alpha (retrieval weight)",
                min_value=0.0,
                max_value=1.0,
                value=float(app_settings.get("alpha", 0.7) or 0.7),
                step=0.05,
                help="Blend weight between retrieval and keyword coverage.",
            )
            must_weight = st.number_input(
                "Must-have weight",
                min_value=0.0,
                max_value=1.0,
                value=float(app_settings.get("must_weight", 0.8) or 0.8),
                step=0.05,
                help="Weight applied to must-have keyword coverage.",
            )
            boost_weight = st.number_input(
                "Boost query weight",
                min_value=0.1,
                max_value=3.0,
                value=float(app_settings.get("boost_weight", 1.6) or 1.6),
                step=0.1,
                help="Strength of boosted queries for missing must-have terms.",
            )
            boost_top_n_missing = st.number_input(
                "Boost top-N missing must-have",
                min_value=1,
                max_value=20,
                value=int(app_settings.get("boost_top_n_missing", 6) or 6),
                step=1,
                help="How many missing must-have terms to boost per iteration.",
            )

        submitted = st.form_submit_button("Save settings")

    if submitted:
        ok, _, err = api_request(
            "PUT",
            api_url,
            "/settings",
            json={
                "auto_reingest_on_save": auto_reingest,
                "export_file": export_file,
                "use_jd_parser": use_jd_parser,
                "jd_model": jd_model,
                "skip_pdf": skip_pdf,
                "max_iters": max_iters,
                "max_bullets": max_bullets,
                "per_query_k": per_query_k,
                "final_k": final_k,
                "threshold": threshold,
                "alpha": alpha,
                "must_weight": must_weight,
                "boost_weight": boost_weight,
                "boost_top_n_missing": boost_top_n_missing,
            },
        )
        if ok:
            st.success("Saved app settings.")
            st.rerun()
        else:
            st.error(err)

    st.caption(f"Settings file: {app_settings.get('config_path', '')}")


def render_resume_editor(api_url: str) -> None:
    st.header("Resume Editor")

    _show_editor_message()

    ingest_running = st.session_state.get("ingest_running", False)
    st.warning("Re-ingesting may take ~10–60s the first time.")
    if st.button(
        "Re-ingest ChromaDB",
        type="primary",
        use_container_width=True,
        disabled=ingest_running,
    ):
        st.session_state["ingest_running"] = True
        status = st.empty()
        with st.spinner("Re-ingesting ChromaDB..."):
            ok, data, err = api_request("POST", api_url, "/admin/ingest", timeout_s=1200)
            if not ok and "HTTP 404" in err:
                ok, data, err = api_request(
                    "POST", api_url, "/admin/export?reingest=1", timeout_s=1200
                )
        st.session_state["ingest_running"] = False

        if ok and isinstance(data, dict):
            if "count" in data:
                status.success(
                    f"Ingested {data.get('count', 0)} bullets in {data.get('elapsed_s', 0)}s."
                )
            else:
                status.success("Re-ingest completed.")
        else:
            status.error(err or "Ingest failed.")

    st.divider()

    st.subheader("Profile")
    ok, personal_info, err = api_request("GET", api_url, "/personal_info", timeout_s=10)
    if not ok:
        st.error(f"Failed to load personal info: {err}")
        return

    with st.form("personal_info_form"):
        name = st.text_input("Name", value=personal_info.get("name", ""))
        phone = st.text_input("Phone", value=personal_info.get("phone", ""))
        email = st.text_input("Email", value=personal_info.get("email", ""))
        linkedin_id = st.text_input("LinkedIn ID", value=personal_info.get("linkedin_id", ""))
        github_id = st.text_input("GitHub ID", value=personal_info.get("github_id", ""))
        linkedin = st.text_input("LinkedIn URL", value=personal_info.get("linkedin", ""))
        github = st.text_input("GitHub URL", value=personal_info.get("github", ""))
        submitted = st.form_submit_button("Save personal info")

    if submitted:
        ok, _, err = api_request(
            "PUT",
            api_url,
            "/personal_info",
            json={
                "name": name,
                "phone": phone,
                "email": email,
                "linkedin_id": linkedin_id,
                "github_id": github_id,
                "linkedin": linkedin,
                "github": github,
            },
        )
        if ok:
            _set_editor_message("success", "Saved personal info.")
            st.rerun()
        else:
            st.error(err)

    st.subheader("Skills")
    ok, skills, err = api_request("GET", api_url, "/skills", timeout_s=10)
    if not ok:
        st.error(f"Failed to load skills: {err}")
        return

    with st.form("skills_form"):
        languages_frameworks = st.text_area(
            "Languages & Frameworks", value=skills.get("languages_frameworks", ""), height=80
        )
        ai_ml = st.text_area("AI/ML", value=skills.get("ai_ml", ""), height=80)
        db_tools = st.text_area("Database & Tools", value=skills.get("db_tools", ""), height=80)
        submitted = st.form_submit_button("Save skills")

    if submitted:
        ok, _, err = api_request(
            "PUT",
            api_url,
            "/skills",
            json={
                "languages_frameworks": languages_frameworks,
                "ai_ml": ai_ml,
                "db_tools": db_tools,
            },
        )
        if ok:
            _set_editor_message("success", "Saved skills.")
            st.rerun()
        else:
            st.error(err)

    st.divider()

    st.subheader("Education")
    ok, education, err = api_request("GET", api_url, "/education", timeout_s=20)
    if not ok:
        st.error(f"Failed to load education: {err}")
        return

    with st.form("add_education_form", clear_on_submit=True):
        edu_school = st.text_input("School", key="new_edu_school")
        edu_degree = st.text_input("Degree", key="new_edu_degree")
        edu_dates = st.text_input("Dates", key="new_edu_dates")
        edu_location = st.text_input("Location", key="new_edu_location")
        edu_bullets = st.text_area(
            "Bullets (one per line)", key="new_edu_bullets", height=120
        )
        submitted_edu = st.form_submit_button("Create Education")

    if submitted_edu:
        bullets = [line.strip() for line in edu_bullets.splitlines() if line.strip()]
        ok, _, err = api_request(
            "POST",
            api_url,
            "/education",
            json={
                "school": edu_school,
                "degree": edu_degree,
                "dates": edu_dates,
                "location": edu_location,
                "bullets": bullets,
            },
        )
        if ok:
            _set_editor_message("success", f"Created education entry for {edu_school}.")
            st.rerun()
        else:
            st.error(err)

    for edu in education or []:
        edu_id = edu.get("id")
        title = f"{edu.get('school', '')} — {edu.get('degree', '')}"
        meta = f"{edu.get('dates', '')} · {edu.get('location', '')}"
        with st.expander(title, expanded=False):
            st.caption(meta)
            col_edit, col_delete = st.columns([1, 1])
            edit_key = f"edit_edu_{edu_id}"
            if col_edit.button("Edit education", key=f"toggle_{edit_key}"):
                st.session_state[edit_key] = not st.session_state.get(edit_key, False)
            if col_delete.button("Delete education", key=f"delete_edu_{edu_id}"):
                ok, _, err = api_request("DELETE", api_url, f"/education/{edu_id}")
                if ok:
                    _set_editor_message("success", f"Deleted education entry {edu_id}.")
                    st.rerun()
                else:
                    st.error(err)

            bullets = edu.get("bullets", []) or []
            if bullets:
                st.caption("Bullets")
                for bullet in bullets:
                    st.write(f"- {bullet}")
            else:
                st.caption("No bullets yet.")

            if st.session_state.get(edit_key, False):
                with st.form(f"edit_edu_form_{edu_id}"):
                    school = st.text_input(
                        "School", value=edu.get("school", ""), key=f"{edit_key}_school"
                    )
                    degree = st.text_input(
                        "Degree", value=edu.get("degree", ""), key=f"{edit_key}_degree"
                    )
                    dates = st.text_input(
                        "Dates", value=edu.get("dates", ""), key=f"{edit_key}_dates"
                    )
                    location = st.text_input(
                        "Location", value=edu.get("location", ""), key=f"{edit_key}_location"
                    )
                    bullets_text = st.text_area(
                        "Bullets (one per line)",
                        value="\n".join(bullets),
                        key=f"{edit_key}_bullets",
                        height=120,
                    )
                    submitted = st.form_submit_button("Save education")

                if submitted:
                    if not school.strip():
                        st.error("School is required.")
                    else:
                        new_bullets = [
                            line.strip() for line in bullets_text.splitlines() if line.strip()
                        ]
                        ok, _, err = api_request(
                            "PUT",
                            api_url,
                            f"/education/{edu_id}",
                            json={
                                "school": school,
                                "degree": degree,
                                "dates": dates,
                                "location": location,
                                "bullets": new_bullets,
                            },
                        )
                        if ok:
                            st.session_state[edit_key] = False
                            _set_editor_message("success", f"Updated education {edu_id}.")
                            st.rerun()
                        else:
                            st.error(err)

    st.divider()

    st.subheader("Add Experience")
    with st.form("add_experience_form", clear_on_submit=True):
        exp_company = st.text_input("Company", key="new_exp_company")
        exp_role = st.text_input("Role", key="new_exp_role")
        exp_dates = st.text_input("Dates", key="new_exp_dates")
        exp_location = st.text_input("Location", key="new_exp_location")
        exp_bullets = st.text_area(
            "Bullets (one per line)", key="new_exp_bullets", height=120
        )
        submitted = st.form_submit_button("Create Experience")

    if submitted:
        bullets = [line.strip() for line in exp_bullets.splitlines() if line.strip()]
        ok, _, err = api_request(
            "POST",
            api_url,
            "/experiences",
            json={
                "company": exp_company,
                "role": exp_role,
                "dates": exp_dates,
                "location": exp_location,
                "bullets": bullets,
            },
        )
        if ok:
            _set_editor_message("success", f"Created experience for {exp_company}.")
            st.rerun()
        else:
            st.error(err)

    st.divider()

    ok, experiences, err = api_request("GET", api_url, "/experiences", timeout_s=20)
    if not ok:
        st.error(f"Failed to load experiences: {err}")
        return

    ok, projects, err = api_request("GET", api_url, "/projects", timeout_s=20)
    if not ok:
        st.error(f"Failed to load projects: {err}")
        return

    profile_empty = (
        not any(value.strip() for value in personal_info.values() if isinstance(value, str))
        and not any(value.strip() for value in skills.values() if isinstance(value, str))
        and not (education or [])
        and not (experiences or [])
        and not (projects or [])
    )
    if profile_empty:
        st.info(
            "Create your profile below. Start with personal info and skills, then add education, "
            "experiences, projects, and bullets."
        )

    st.subheader("Work Experience")
    for exp in experiences or []:
        job_id = exp.get("job_id", "")
        title = f"{exp.get('company', '')} — {exp.get('role', '')}"
        meta = f"{exp.get('dates', '')} · {exp.get('location', '')}"
        with st.expander(title, expanded=True):
            st.caption(meta)
            st.caption(f"job_id: {job_id}")
            col_edit, col_delete = st.columns([1, 1])
            edit_key = f"edit_exp_{job_id}"
            if col_edit.button("Edit experience", key=f"toggle_{edit_key}"):
                st.session_state[edit_key] = not st.session_state.get(edit_key, False)
            if col_delete.button("Delete experience", key=f"delete_exp_{job_id}"):
                ok, _, err = api_request("DELETE", api_url, f"/experiences/{job_id}")
                if ok:
                    _set_editor_message("success", f"Deleted experience {job_id}.")
                    st.rerun()
                else:
                    st.error(err)

            if st.session_state.get(edit_key, False):
                with st.form(f"edit_exp_form_{job_id}"):
                    company = st.text_input(
                        "Company", value=exp.get("company", ""), key=f"{edit_key}_company"
                    )
                    role = st.text_input(
                        "Role", value=exp.get("role", ""), key=f"{edit_key}_role"
                    )
                    dates = st.text_input(
                        "Dates", value=exp.get("dates", ""), key=f"{edit_key}_dates"
                    )
                    location = st.text_input(
                        "Location", value=exp.get("location", ""), key=f"{edit_key}_location"
                    )
                    submitted = st.form_submit_button("Save experience")

                if submitted:
                    if not company.strip() or not role.strip():
                        st.error("Company and role are required.")
                    else:
                        ok, _, err = api_request(
                            "PUT",
                            api_url,
                            f"/experiences/{job_id}",
                            json={
                                "company": company,
                                "role": role,
                                "dates": dates,
                                "location": location,
                            },
                        )
                        if ok:
                            st.session_state[edit_key] = False
                            _set_editor_message("success", f"Updated experience {job_id}.")
                            st.rerun()
                        else:
                            st.error(err)

            bullets = exp.get("bullets", []) or []
            if not bullets:
                st.info("No bullets yet.")
            for bullet in bullets:
                _render_bullet_controls(api_url, "experiences", job_id, bullet)

            new_key = f"new_exp_{job_id}"
            new_text = st.text_area("New bullet", key=new_key, height=90)
            if st.button("Add bullet", key=f"add_exp_{job_id}"):
                if not new_text.strip():
                    st.error("Bullet text cannot be empty.")
                else:
                    ok, _, err = api_request(
                        "POST",
                        api_url,
                        f"/experiences/{job_id}/bullets",
                        json={"text_latex": new_text},
                    )
                    if ok:
                        st.session_state[new_key] = ""
                        _set_editor_message("success", f"Added bullet to {job_id}.")
                        st.rerun()
                    else:
                        st.error(err)

    st.subheader("Add Project")
    with st.form("add_project_form", clear_on_submit=True):
        proj_name = st.text_input("Project name", key="new_proj_name")
        proj_tech = st.text_input("Technologies", key="new_proj_tech")
        proj_bullets = st.text_area(
            "Bullets (one per line)", key="new_proj_bullets", height=120
        )
        submitted_proj = st.form_submit_button("Create Project")

    if submitted_proj:
        bullets = [line.strip() for line in proj_bullets.splitlines() if line.strip()]
        ok, _, err = api_request(
            "POST",
            api_url,
            "/projects",
            json={"name": proj_name, "technologies": proj_tech, "bullets": bullets},
        )
        if ok:
            _set_editor_message("success", f"Created project {proj_name}.")
            st.rerun()
        else:
            st.error(err)

    st.subheader("Projects")
    for proj in projects or []:
        project_id = proj.get("project_id", "")
        title = f"{proj.get('name', '')} — {proj.get('technologies', '')}"
        with st.expander(title, expanded=True):
            st.caption(f"project_id: {project_id}")
            col_edit, col_delete = st.columns([1, 1])
            edit_key = f"edit_proj_{project_id}"
            if col_edit.button("Edit project", key=f"toggle_{edit_key}"):
                st.session_state[edit_key] = not st.session_state.get(edit_key, False)
            if col_delete.button("Delete project", key=f"delete_proj_{project_id}"):
                ok, _, err = api_request("DELETE", api_url, f"/projects/{project_id}")
                if ok:
                    _set_editor_message("success", f"Deleted project {project_id}.")
                    st.rerun()
                else:
                    st.error(err)

            if st.session_state.get(edit_key, False):
                with st.form(f"edit_proj_form_{project_id}"):
                    name = st.text_input(
                        "Project name", value=proj.get("name", ""), key=f"{edit_key}_name"
                    )
                    technologies = st.text_input(
                        "Technologies",
                        value=proj.get("technologies", ""),
                        key=f"{edit_key}_tech",
                    )
                    submitted = st.form_submit_button("Save project")

                if submitted:
                    if not name.strip():
                        st.error("Project name is required.")
                    else:
                        ok, _, err = api_request(
                            "PUT",
                            api_url,
                            f"/projects/{project_id}",
                            json={"name": name, "technologies": technologies},
                        )
                        if ok:
                            st.session_state[edit_key] = False
                            _set_editor_message("success", f"Updated project {project_id}.")
                            st.rerun()
                        else:
                            st.error(err)

            bullets = proj.get("bullets", []) or []
            if not bullets:
                st.info("No bullets yet.")
            for bullet in bullets:
                _render_bullet_controls(api_url, "projects", project_id, bullet)

            new_key = f"new_proj_{project_id}"
            new_text = st.text_area("New bullet", key=new_key, height=90)
            if st.button("Add bullet", key=f"add_proj_{project_id}"):
                if not new_text.strip():
                    st.error("Bullet text cannot be empty.")
                else:
                    ok, _, err = api_request(
                        "POST",
                        api_url,
                        f"/projects/{project_id}/bullets",
                        json={"text_latex": new_text},
                    )
                    if ok:
                        st.session_state[new_key] = ""
                        _set_editor_message("success", f"Added bullet to {project_id}.")
                        st.rerun()
                    else:
                        st.error(err)


def render_generate_page(api_url: str) -> None:
    st.header("Generate")

    st.subheader("Job Description")
    jd_text = st.text_area(
        "Paste the JD here", height=260, placeholder="Paste a job description..."
    )

    colA, colB = st.columns([1, 2], gap="large")

    with colA:
        if st.button("Generate", type="primary", use_container_width=True):
            if not jd_text.strip():
                st.error("JD is empty.")
            else:
                payload = {
                    "jd_text": jd_text.strip(),
                }
                with st.spinner("Running agent..."):
                    try:
                        resp = requests.post(f"{api_url}/generate", json=payload, timeout=600)
                        resp.raise_for_status()
                        out = resp.json()
                        st.session_state["last_run"] = out
                    except Exception as e:
                        st.exception(e)

    with colB:
        st.subheader("Run Output")
        run = st.session_state.get("last_run")

        if not run:
            st.info("No run yet. Click Generate.")
        else:
            run_id = run.get("run_id")
            st.write(
                {
                    "run_id": run_id,
                    "profile_used": run.get("profile_used"),
                    "best_iteration_index": run.get("best_iteration_index"),
                }
            )

            report = None
            try:
                r = requests.get(f"{api_url}{run['report_url']}", timeout=60)
                r.raise_for_status()
                report = r.json()
            except Exception as e:
                st.warning("Failed to load report.json from API")
                st.exception(e)

            if report:
                best = report.get("best_score")
                if best:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Final score", best.get("final_score"))
                    c2.metric("Retrieval", round(float(best.get("retrieval_score", 0.0)), 3))
                    c3.metric(
                        "Coverage (bullets)",
                        round(float(best.get("coverage_bullets_only", 0.0)), 3),
                    )
                    c4.metric(
                        "Coverage (all+skills)",
                        round(float(best.get("coverage_all", 0.0)), 3),
                    )

                iters = report.get("iterations", [])
                if iters:
                    rows = []
                    for it in iters:
                        scores = it.get("scores") or {}
                        missing = it.get("missing") or {}
                        rows.append(
                            {
                                "iter": it.get("iteration"),
                                "final": scores.get("final"),
                                "retrieval": scores.get("retrieval"),
                                "cov_bullets": scores.get("coverage_bullets_only"),
                                "cov_all": scores.get("coverage_all"),
                                "missing_must_bullets": len(
                                    missing.get("must_bullets_only") or []
                                ),
                                "missing_nice_bullets": len(
                                    missing.get("nice_bullets_only") or []
                                ),
                            }
                        )

                    df = pd.DataFrame(rows)
                    st.subheader("Iterations")
                    st.dataframe(df, use_container_width=True, hide_index=True)

                    st.subheader("Iteration details")
                    idxs = [int(x.get("iteration", 0)) for x in iters]
                    default_idx = min(
                        int(report.get("best_iteration_index", 0)), max(len(idxs) - 1, 0)
                    )
                    pick = st.selectbox("Pick iteration", idxs, index=default_idx)

                    chosen = next(
                        (x for x in iters if int(x.get("iteration", -1)) == int(pick)), None
                    )
                    if chosen:
                        st.markdown("**Queries used**")
                        st.code("\n".join(chosen.get("queries_used") or []))

                        missing = chosen.get("missing") or {}
                        st.markdown("**Missing must-have (bullets only)**")
                        st.write(missing.get("must_bullets_only") or [])
                        st.markdown("**Missing nice-to-have (bullets only)**")
                        st.write(missing.get("nice_bullets_only") or [])

                        st.markdown("**Selected IDs**")
                        st.code("\n".join(chosen.get("selected_ids") or []))

                with st.expander("Raw report.json"):
                    st.json(report)

            st.subheader("Download")
            try:
                pdf = requests.get(f"{api_url}{run['pdf_url']}", timeout=120).content
                st.download_button(
                    "Download tailored_resume.pdf",
                    data=pdf,
                    file_name="tailored_resume.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception:
                st.warning("PDF not ready yet or download failed.")

            try:
                rep_bytes = requests.get(f"{api_url}{run['report_url']}", timeout=60).content
                st.download_button(
                    "Download resume_report.json",
                    data=rep_bytes,
                    file_name="resume_report.json",
                    mime="application/json",
                    use_container_width=True,
                )
            except Exception:
                st.warning("report.json not ready yet or download failed.")

    st.markdown("---")
    st.caption(
        "Tip: update app defaults in the Settings page or edit config/user_settings.json if needed."
    )


def main() -> None:
    settings = get_settings()
    api_url = settings.api_url.rstrip("/")

    st.set_page_config(page_title="AI Resume Agent", layout="wide")
    st.title("AI Resume Agent")

    ok, info = render_health_sidebar(api_url)
    if not ok:
        st.error(
            f"API server is DOWN: {api_url}\n\n"
            f"Health check failed: {info}\n\n"
            "Start FastAPI (server.py) and click Re-check in the sidebar.",
            icon="🚨",
        )
        st.stop()

    st.sidebar.divider()
    st.sidebar.subheader("Navigation")
    page = st.sidebar.radio("Page", ["Generate", "Resume Editor", "Settings"])

    st.sidebar.divider()

    if page == "Resume Editor":
        render_resume_editor(api_url)
        return
    if page == "Settings":
        render_settings_page(api_url)
        return

    render_generate_page(api_url)


if __name__ == "__main__":
    main()
