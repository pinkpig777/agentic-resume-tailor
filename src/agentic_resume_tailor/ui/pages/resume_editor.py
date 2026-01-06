from typing import Any, Dict

import streamlit as st

from agentic_resume_tailor.ui.common import api_request, get_health_cached


def _set_editor_message(level: str, text: str) -> None:
    """Store a flash message for the editor.

    Args:
        level: The level value.
        text: The text value.
    """
    st.session_state["editor_message"] = {"level": level, "text": text}


def _show_editor_message() -> None:
    """Render and clear the editor flash message."""
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


def _set_open_expander(expander_key: str) -> None:
    """Persist the currently active expander for the resume editor."""
    st.session_state["resume_editor_open_expander"] = expander_key


def _is_expander_open(expander_key: str) -> bool:
    """Check whether an expander should render open on rerun."""
    return st.session_state.get("resume_editor_open_expander") == expander_key


def _render_bullet_controls(
    api_url: str,
    section: str,
    parent_id: str,
    bullet: dict,
    expander_key: str,
    add_spacer: bool = True,
) -> None:
    """Render edit/delete controls for a bullet.

    Args:
        api_url: Base URL for the API.
        section: The section value.
        parent_id: Parent identifier.
        bullet: The bullet value.
        expander_key: The expander identifier.
        add_spacer: Whether to add a spacer after the bullet row.
    """
    bullet_id = bullet.get("id", "")
    text = bullet.get("text_latex", "")
    edit_key = f"edit_{section}_{parent_id}_{bullet_id}"
    text_key = f"text_{section}_{parent_id}_{bullet_id}"

    col_text, col_edit, col_del = st.columns([8, 1, 1])
    col_text.markdown(f"<div class='bullet-card'>{text}</div>", unsafe_allow_html=True)
    if col_edit.button("Edit", key=f"edit_btn_{edit_key}"):
        _set_open_expander(expander_key)
        st.session_state[edit_key] = not st.session_state.get(edit_key, False)
    if col_del.button("Delete", key=f"del_btn_{edit_key}"):
        _set_open_expander(expander_key)
        ok, _, err = api_request("DELETE", api_url, f"/{section}/{parent_id}/bullets/{bullet_id}")
        if ok:
            _set_editor_message("success", f"Deleted {section} bullet.")
            st.rerun()
        else:
            st.error(err)

    if st.session_state.get(edit_key, False):
        new_text = st.text_area("Bullet text", value=text, key=text_key, height=90)
        if st.button("Save", key=f"save_{edit_key}"):
            _set_open_expander(expander_key)
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
                    _set_editor_message("success", f"Updated {section} bullet.")
                    st.rerun()
                else:
                    st.error(err)
    if add_spacer:
        st.markdown("<div class='bullet-row-spacer'></div>", unsafe_allow_html=True)


def render_resume_editor(api_url: str) -> None:
    """Render the Resume Editor page UI.

    Args:
        api_url: Base URL for the API.
    """
    st.header("Resume Editor")
    ok, _ = get_health_cached(api_url)
    if not ok:
        st.error("API server is down. Start the backend to edit your profile.", icon="🚨")
        return

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

    with st.expander("Add Education", expanded=False):
        with st.form("add_education_form", clear_on_submit=True):
            edu_school = st.text_input("School", key="new_edu_school")
            edu_degree = st.text_input("Degree", key="new_edu_degree")
            edu_dates = st.text_input("Dates", key="new_edu_dates")
            edu_location = st.text_input("Location", key="new_edu_location")
            edu_bullets = st.text_area("Bullets (one per line)", key="new_edu_bullets", height=120)
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
        expander_key = f"edu_{edu_id}"
        with st.expander(title, expanded=_is_expander_open(expander_key)):
            st.caption(meta)
            col_edit, col_delete = st.columns([1, 1])
            edit_key = f"edit_edu_{edu_id}"
            if col_edit.button("Edit education", key=f"toggle_{edit_key}"):
                _set_open_expander(expander_key)
                st.session_state[edit_key] = not st.session_state.get(edit_key, False)
            if col_delete.button("Delete education", key=f"delete_edu_{edu_id}"):
                ok, _, err = api_request("DELETE", api_url, f"/education/{edu_id}")
                if ok:
                    _set_editor_message("success", "Deleted education entry.")
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
                    _set_open_expander(expander_key)
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
                            _set_editor_message("success", "Updated education entry.")
                            st.rerun()
                        else:
                            st.error(err)

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
    with st.expander("Add Experience", expanded=False):
        with st.form("add_experience_form", clear_on_submit=True):
            exp_company = st.text_input("Company", key="new_exp_company")
            exp_role = st.text_input("Role", key="new_exp_role")
            exp_dates = st.text_input("Dates", key="new_exp_dates")
            exp_location = st.text_input("Location", key="new_exp_location")
            exp_bullets = st.text_area("Bullets (one per line)", key="new_exp_bullets", height=140)
            submitted_exp = st.form_submit_button("Create Experience")

    if submitted_exp:
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

    for exp in experiences or []:
        job_id = exp.get("job_id")
        header = f"{exp.get('company', '')} — {exp.get('role', '')}"
        meta = f"{exp.get('dates', '')} · {exp.get('location', '')}"
        expander_key = f"exp_{job_id}"
        with st.expander(header, expanded=_is_expander_open(expander_key)):
            st.caption(meta)
            cols = st.columns([1, 1])
            edit_key = f"edit_exp_{job_id}"
            if cols[0].button("Edit", key=f"toggle_{edit_key}"):
                _set_open_expander(expander_key)
                st.session_state[edit_key] = not st.session_state.get(edit_key, False)
            if cols[1].button("Delete", key=f"delete_{edit_key}"):
                ok, _, err = api_request("DELETE", api_url, f"/experiences/{job_id}")
                if ok:
                    _set_editor_message("success", "Deleted experience.")
                    st.rerun()
                else:
                    st.error(err)

            if st.session_state.get(edit_key, False):
                with st.form(f"edit_exp_form_{job_id}"):
                    company = st.text_input(
                        "Company", value=exp.get("company", ""), key=f"{edit_key}_company"
                    )
                    role = st.text_input("Role", value=exp.get("role", ""), key=f"{edit_key}_role")
                    dates = st.text_input(
                        "Dates", value=exp.get("dates", ""), key=f"{edit_key}_dates"
                    )
                    location = st.text_input(
                        "Location", value=exp.get("location", ""), key=f"{edit_key}_location"
                    )
                    submitted = st.form_submit_button("Save experience")

                if submitted:
                    _set_open_expander(expander_key)
                    if not company.strip():
                        st.error("Company is required.")
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
                            _set_editor_message("success", "Updated experience.")
                            st.rerun()
                        else:
                            st.error(err)

            bullets = exp.get("bullets", []) or []
            if not bullets:
                st.info("No bullets yet.")
            for index, bullet in enumerate(bullets):
                _render_bullet_controls(
                    api_url,
                    "experiences",
                    job_id,
                    bullet,
                    expander_key,
                    add_spacer=index < len(bullets) - 1,
                )

            new_key = f"new_exp_{job_id}"
            with st.form(f"add_exp_bullet_form_{job_id}", clear_on_submit=True):
                new_text = st.text_area("New bullet", key=new_key, height=90)
                submitted_bullet = st.form_submit_button("Add bullet")
            if submitted_bullet:
                _set_open_expander(expander_key)
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
                        _set_editor_message("success", "Added bullet to experience.")
                        st.rerun()
                    else:
                        st.error(err)

    st.subheader("Projects")
    with st.expander("Add Project", expanded=False):
        with st.form("add_project_form", clear_on_submit=True):
            proj_name = st.text_input("Project name", key="new_proj_name")
            proj_tech = st.text_input("Technologies", key="new_proj_tech")
            submitted_proj = st.form_submit_button("Create Project")

    if submitted_proj:
        ok, _, err = api_request(
            "POST",
            api_url,
            "/projects",
            json={"name": proj_name, "technologies": proj_tech},
        )
        if ok:
            _set_editor_message("success", f"Created project {proj_name}.")
            st.rerun()
        else:
            st.error(err)

    for proj in projects or []:
        project_id = proj.get("project_id")
        title = proj.get("name", "")
        meta = proj.get("technologies", "")
        expander_key = f"proj_{project_id}"
        with st.expander(title, expanded=_is_expander_open(expander_key)):
            st.caption(meta)
            cols = st.columns([1, 1])
            edit_key = f"edit_proj_{project_id}"
            if cols[0].button("Edit", key=f"toggle_{edit_key}"):
                _set_open_expander(expander_key)
                st.session_state[edit_key] = not st.session_state.get(edit_key, False)
            if cols[1].button("Delete", key=f"delete_proj_{project_id}"):
                ok, _, err = api_request("DELETE", api_url, f"/projects/{project_id}")
                if ok:
                    _set_editor_message("success", "Deleted project.")
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
                    _set_open_expander(expander_key)
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
                            _set_editor_message("success", "Updated project.")
                            st.rerun()
                        else:
                            st.error(err)

            bullets = proj.get("bullets", []) or []
            if not bullets:
                st.info("No bullets yet.")
            for index, bullet in enumerate(bullets):
                _render_bullet_controls(
                    api_url,
                    "projects",
                    project_id,
                    bullet,
                    expander_key,
                    add_spacer=index < len(bullets) - 1,
                )

            new_key = f"new_proj_{project_id}"
            with st.form(f"add_proj_bullet_form_{project_id}", clear_on_submit=True):
                new_text = st.text_area("New bullet", key=new_key, height=90)
                submitted_bullet = st.form_submit_button("Add bullet")
            if submitted_bullet:
                _set_open_expander(expander_key)
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
                        _set_editor_message("success", "Added bullet to project.")
                        st.rerun()
                    else:
                        st.error(err)
