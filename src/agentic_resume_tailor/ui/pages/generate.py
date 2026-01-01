from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

import requests
import streamlit as st
import streamlit.components.v1 as components

from agentic_resume_tailor.ui.common import api_request, check_server_health, get_health_cached

GEN_EXECUTOR = ThreadPoolExecutor(max_workers=1)


def _autorefresh(interval_ms: int) -> None:
    """Trigger a client-driven rerun without blocking the server thread."""
    components.html(
        f"""
        <script>
        const sendMessage = (value) => {{
          window.parent.postMessage(
            {{ isStreamlitMessage: true, type: "streamlit:setComponentValue", value: value }},
            "*"
          );
        }};
        setTimeout(() => sendMessage(Date.now()), {interval_ms});
        </script>
        """,
        height=0,
    )


def _run_generate(api_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run the generate request and return the response payload.

    Args:
        api_url: Base URL for the API.
        payload: Request payload.

    Returns:
        Response JSON payload.
    """
    resp = requests.post(f"{api_url}/generate", json=payload, timeout=(3, 600))
    resp.raise_for_status()
    return resp.json()


def _check_generate_status() -> None:
    """Update session state if a background generate task completed."""
    future = st.session_state.get("generate_future")
    if not isinstance(future, Future):
        return
    if not future.done():
        return
    st.session_state["generate_future"] = None
    try:
        out = future.result()
    except Exception as exc:
        st.session_state["generate_error"] = str(exc)
        return
    st.session_state["generate_error"] = ""
    st.session_state["last_run"] = out
    st.session_state["selection_run_id"] = out.get("run_id")
    for key in list(st.session_state.keys()):
        if key.startswith("keep_"):
            del st.session_state[key]


def _fetch_resume_sections(
    api_url: str,
) -> Tuple[bool, List[Dict[str, Any]], List[Dict[str, Any]], str]:
    """Fetch experiences and projects for bullet mapping.

    Args:
        api_url: Base URL for the API.

    Returns:
        Tuple of ok flag, experiences, projects, and error message.
    """
    ok_exp, experiences, err = api_request("GET", api_url, "/experiences", timeout_s=10)
    ok_proj, projects, err_proj = api_request("GET", api_url, "/projects", timeout_s=10)
    if not ok_exp:
        return False, [], [], err
    if not ok_proj:
        return False, [], [], err_proj
    return True, experiences or [], projects or [], ""


def _build_bullet_lookup(
    experiences: List[Dict[str, Any]],
    projects: List[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
    """Build lookup maps for bullet metadata.

    Args:
        experiences: Experience entries from the API.
        projects: Project entries from the API.

    Returns:
        Tuple of bullet lookup, experience metadata, and project metadata.
    """
    lookup: Dict[str, Dict[str, Any]] = {}
    exp_meta: Dict[str, Dict[str, str]] = {}
    proj_meta: Dict[str, Dict[str, str]] = {}

    for exp in experiences:
        job_id = exp.get("job_id") or ""
        if not job_id:
            continue
        exp_meta[job_id] = {
            "company": exp.get("company", ""),
            "role": exp.get("role", ""),
            "dates": exp.get("dates", ""),
            "location": exp.get("location", ""),
        }
        for bullet in exp.get("bullets", []) or []:
            local_id = bullet.get("id", "")
            if not local_id:
                continue
            bullet_id = f"exp:{job_id}:{local_id}"
            lookup[bullet_id] = {
                "section": "experience",
                "group_id": job_id,
                "text": bullet.get("text_latex", ""),
                "is_temp": False,
            }

    for proj in projects:
        project_id = proj.get("project_id") or ""
        if not project_id:
            continue
        proj_meta[project_id] = {
            "name": proj.get("name", ""),
            "technologies": proj.get("technologies", ""),
        }
        for bullet in proj.get("bullets", []) or []:
            local_id = bullet.get("id", "")
            if not local_id:
                continue
            bullet_id = f"proj:{project_id}:{local_id}"
            lookup[bullet_id] = {
                "section": "project",
                "group_id": project_id,
                "text": bullet.get("text_latex", ""),
                "is_temp": False,
            }

    return lookup, exp_meta, proj_meta


def _safe_key(value: str) -> str:
    """Normalize a string to a safe Streamlit key."""
    return value.replace(":", "_").replace("/", "_")


def _temp_bullet_id(addition: Dict[str, Any]) -> str:
    """Build a bullet id for a temp addition."""
    parent_type = addition.get("parent_type")
    parent_id = addition.get("parent_id")
    temp_id = addition.get("temp_id")
    if not parent_type or not parent_id or not temp_id:
        return ""
    prefix = "exp" if parent_type == "experience" else "proj"
    return f"{prefix}:{parent_id}:{temp_id}"


def _ensure_temp_state(run_id: str) -> None:
    """Reset temp bullet state when the run changes."""
    if st.session_state.get("temp_run_id") != run_id:
        st.session_state["temp_run_id"] = run_id
        st.session_state["temp_additions"] = []
        st.session_state["temp_edits"] = {}
        st.session_state["temp_bullet_counter"] = 0


def _seed_temp_state_from_report(report: Dict[str, Any]) -> None:
    """Seed temp state from report if local state is empty."""
    if not st.session_state.get("temp_additions") and report.get("temp_additions"):
        additions: List[Dict[str, Any]] = []
        for addition in report.get("temp_additions", []):
            if not isinstance(addition, dict):
                continue
            temp_id = addition.get("temp_id")
            bullet_id = addition.get("bullet_id", "")
            if not temp_id and isinstance(bullet_id, str):
                parts = bullet_id.split(":")
                if len(parts) >= 3:
                    temp_id = parts[2]
            parent_type = addition.get("parent_type")
            parent_id = addition.get("parent_id")
            if not temp_id or not parent_type or not parent_id:
                continue
            additions.append(
                {
                    "temp_id": temp_id,
                    "parent_type": parent_type,
                    "parent_id": parent_id,
                    "text_latex": addition.get("text_latex", ""),
                }
            )
        st.session_state["temp_additions"] = additions

    if not st.session_state.get("temp_edits") and report.get("temp_edits"):
        st.session_state["temp_edits"] = report.get("temp_edits", {})


def _next_temp_id() -> str:
    """Return the next temp bullet id for this run."""
    counter = int(st.session_state.get("temp_bullet_counter", 0) or 0) + 1
    st.session_state["temp_bullet_counter"] = counter
    return f"tmp{counter:03d}"


def _add_temp_bullets_to_lookup(
    lookup: Dict[str, Dict[str, Any]],
    temp_additions: List[Dict[str, Any]],
) -> None:
    """Add temp bullets to the lookup table."""
    for addition in temp_additions:
        bullet_id = _temp_bullet_id(addition)
        if not bullet_id:
            continue
        parent_type = addition.get("parent_type")
        if parent_type == "experience":
            section = "experience"
        elif parent_type == "project":
            section = "project"
        else:
            continue
        lookup[bullet_id] = {
            "section": section,
            "group_id": addition.get("parent_id", ""),
            "text": addition.get("text_latex", ""),
            "is_temp": True,
        }


def _group_selected_bullets(
    selected_ids: List[str],
    lookup: Dict[str, Dict[str, Any]],
    exp_meta: Dict[str, Dict[str, str]],
    proj_meta: Dict[str, Dict[str, str]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Group selected bullets by experience and project.

    Args:
        selected_ids: Selected bullet identifiers.
        lookup: Bullet lookup map.
        exp_meta: Experience metadata map.
        proj_meta: Project metadata map.

    Returns:
        Tuple of experience groups and project groups.
    """
    exp_groups: Dict[str, Dict[str, Any]] = {}
    proj_groups: Dict[str, Dict[str, Any]] = {}
    for bullet_id in selected_ids:
        info = lookup.get(bullet_id)
        if not info:
            continue
        group_id = info.get("group_id", "")
        if info.get("section") == "experience":
            group = exp_groups.setdefault(
                group_id, {"meta": exp_meta.get(group_id, {}), "bullets": []}
            )
            group["bullets"].append(
                {
                    "id": bullet_id,
                    "text": info.get("text", ""),
                    "is_temp": info.get("is_temp", False),
                }
            )
        elif info.get("section") == "project":
            group = proj_groups.setdefault(
                group_id, {"meta": proj_meta.get(group_id, {}), "bullets": []}
            )
            group["bullets"].append(
                {
                    "id": bullet_id,
                    "text": info.get("text", ""),
                    "is_temp": info.get("is_temp", False),
                }
            )
    return exp_groups, proj_groups


def _render_badges(items: List[str], kind: str) -> None:
    """Render keyword badges.

    Args:
        items: List of keywords.
        kind: Badge style kind.
    """
    if not items:
        st.markdown("<span class='art-subtle'>None</span>", unsafe_allow_html=True)
        return
    badge_class = "art-badge--must" if kind == "must" else "art-badge--nice"
    html = "".join(f"<span class='art-badge {badge_class}'>{item}</span>" for item in items)
    st.markdown(html, unsafe_allow_html=True)


def _matched_keywords(
    profile_keywords: Dict[str, Any],
    missing: Dict[str, Any],
    key: str,
    missing_key: str,
) -> List[str]:
    """Compute matched keywords for a category."""
    missing_set = {
        str(item).strip().lower()
        for item in (missing.get(missing_key, []) or [])
        if str(item).strip()
    }
    items = profile_keywords.get(key, []) or []
    matched: List[str] = []
    seen: set[str] = set()
    for item in items:
        if isinstance(item, dict):
            raw = str(item.get("raw") or "").strip()
            canonical = str(item.get("canonical") or raw).strip().lower()
            label = raw or canonical
        else:
            label = str(item).strip()
            canonical = label.lower()
        if not label or not canonical:
            continue
        if canonical in missing_set or canonical in seen:
            continue
        seen.add(canonical)
        matched.append(label)
    return matched


def render_generate_page(api_url: str) -> None:
    """Render the Generate page UI.

    Args:
        api_url: Base URL for the API.
    """
    st.header("Generate")
    ok, info = get_health_cached(api_url)
    if not ok:
        st.error(
            "API server is down. Start the backend and click Re-check in the sidebar.",
            icon="🚨",
        )

    _check_generate_status()
    st.markdown("### Job Description")
    st.session_state.setdefault("jd_text_input", "")
    jd_text = st.text_area(
        "Paste the JD here",
        height=320,
        placeholder="Paste a job description...",
        label_visibility="collapsed",
        key="jd_text_input",
    )
    disabled = not ok
    if st.button(
        "Generate",
        type="primary",
        use_container_width=True,
        disabled=disabled,
    ):
        live_ok, info = check_server_health(api_url, timeout_s=2.0)
        if not live_ok:
            st.error(
                f"Cannot reach API server at {api_url}. "
                "Start the backend and click Re-check in the sidebar."
            )
            return
        if not jd_text.strip():
            st.error("Please paste a job description first.")
        else:
            payload = {"jd_text": jd_text.strip()}
            future = st.session_state.get("generate_future")
            if isinstance(future, Future) and not future.done():
                st.warning("Generation already running.")
            else:
                st.session_state["generate_error"] = ""
                st.session_state["generate_future"] = GEN_EXECUTOR.submit(
                    _run_generate, api_url, payload
                )

    future = st.session_state.get("generate_future")
    if isinstance(future, Future) and not future.done():
        st.info("Generation running. Auto-refreshing until results are ready.")
        _autorefresh(750)
    if st.session_state.get("generate_error"):
        st.error(st.session_state.get("generate_error"))
    run = st.session_state.get("last_run")
    if not run:
        st.info("Generate a run to see summary metrics and bullets.")
        return

    report = None
    try:
        r = requests.get(f"{api_url}{run['report_url']}", timeout=60)
        r.raise_for_status()
        report = r.json()
    except Exception as e:
        st.warning("Failed to load report.json from API")
        st.exception(e)

    if not report:
        st.info("No report data available yet.")
        return

    run_id = run.get("run_id", "")
    if run_id:
        _ensure_temp_state(run_id)
        _seed_temp_state_from_report(report)

    temp_additions: List[Dict[str, Any]] = st.session_state.get("temp_additions", [])
    temp_edits: Dict[str, str] = st.session_state.get("temp_edits", {})
    if st.session_state.pop("rerender_success", False):
        st.success("Updated artifacts with your selection.")

    best = report.get("best_score") or {}
    st.subheader("Results summary")
    c1, c2 = st.columns(2)
    c1.metric("Final score", best.get("final_score", "—"))
    c2.metric("Retrieval", round(float(best.get("retrieval_score", 0.0)), 3))
    c3, c4 = st.columns(2)
    c3.metric(
        "Coverage (bullets)",
        round(float(best.get("coverage_bullets_only", 0.0)), 3),
    )
    c4.metric(
        "Coverage (all)",
        round(float(best.get("coverage_all", 0.0)), 3),
    )

    iters = report.get("iterations", []) or []
    best_idx = int(report.get("best_iteration_index", 0) or 0)
    best_iter = next((x for x in iters if int(x.get("iteration", -1)) == best_idx), None)
    missing = (best_iter or {}).get("missing") or {}

    st.subheader("Matched keywords")
    profile_keywords = report.get("profile_keywords") or {}
    if profile_keywords:
        matched_must = _matched_keywords(profile_keywords, missing, "must_have", "must_all")
        matched_nice = _matched_keywords(profile_keywords, missing, "nice_to_have", "nice_all")
        st.markdown("<div class='art-subtle'>Must-have</div>", unsafe_allow_html=True)
        _render_badges(matched_must, "must")
        st.markdown("<div class='art-subtle'>Nice-to-have</div>", unsafe_allow_html=True)
        _render_badges(matched_nice, "nice")
    else:
        st.markdown(
            "<span class='art-subtle'>Not available (JD parser disabled)</span>",
            unsafe_allow_html=True,
        )

    st.subheader("Missing keywords")
    st.markdown("<div class='art-subtle'>Must-have</div>", unsafe_allow_html=True)
    _render_badges(missing.get("must_bullets_only") or [], "must")
    st.markdown("<div class='art-subtle'>Nice-to-have</div>", unsafe_allow_html=True)
    _render_badges(missing.get("nice_bullets_only") or [], "nice")

    ok_sections, experiences, projects, err = _fetch_resume_sections(api_url)
    lookup, exp_meta, proj_meta = _build_bullet_lookup(
        experiences if ok_sections else [], projects if ok_sections else []
    )
    _add_temp_bullets_to_lookup(lookup, temp_additions)
    selected_ids = report.get("selected_ids") or []
    temp_ids = [_temp_bullet_id(addition) for addition in temp_additions]
    display_ids = list(dict.fromkeys(selected_ids + [bid for bid in temp_ids if bid]))
    exp_groups, proj_groups = _group_selected_bullets(display_ids, lookup, exp_meta, proj_meta)

    st.subheader("Selected bullets")
    st.caption("Uncheck bullets to exclude them from re-render.")
    kept_ids: List[str] = []
    selected_set = set(selected_ids)
    pending_edits: Dict[str, str] = {}
    pending_temp_edits: Dict[str, str] = {}
    temp_additions_by_id = {
        _temp_bullet_id(addition): addition
        for addition in temp_additions
        if _temp_bullet_id(addition)
    }

    with st.expander("Add temporary bullet", expanded=False):
        if not exp_meta and not proj_meta:
            st.info("Add experiences or projects in Resume Editor to attach temporary bullets.")
        else:
            target_type = st.radio(
                "Attach to",
                options=["Experience", "Project"],
                horizontal=True,
                key="temp_target_type",
            )
            parent_id = ""
            if target_type == "Experience":
                exp_ids = list(exp_meta.keys())
                if exp_ids:
                    exp_labels = {
                        exp_id: f"{exp_meta[exp_id].get('company', '')} · "
                        f"{exp_meta[exp_id].get('role', '')}"
                        for exp_id in exp_ids
                    }
                    parent_id = st.selectbox(
                        "Choose experience",
                        options=exp_ids,
                        format_func=lambda x: exp_labels.get(x, x),
                        key="temp_parent_exp",
                    )
                else:
                    st.info("No experiences available.")
            else:
                proj_ids = list(proj_meta.keys())
                if proj_ids:
                    proj_labels = {
                        proj_id: f"{proj_meta[proj_id].get('name', '')} · "
                        f"{proj_meta[proj_id].get('technologies', '')}"
                        for proj_id in proj_ids
                    }
                    parent_id = st.selectbox(
                        "Choose project",
                        options=proj_ids,
                        format_func=lambda x: proj_labels.get(x, x),
                        key="temp_parent_proj",
                    )
                else:
                    st.info("No projects available.")
            text_latex = st.text_area(
                "Bullet text (LaTeX-ready)",
                height=100,
                placeholder="Write a LaTeX-ready bullet...",
                key="temp_text_latex",
            )
            if st.button("Add temporary bullet", key="temp_add_submit"):
                if not parent_id:
                    st.error("Select a target experience or project.")
                elif not text_latex.strip():
                    st.error("Bullet text cannot be empty.")
                else:
                    temp_additions.append(
                        {
                            "temp_id": _next_temp_id(),
                            "parent_type": "experience"
                            if target_type == "Experience"
                            else "project",
                            "parent_id": parent_id,
                            "text_latex": text_latex,
                        }
                    )
                    st.session_state["temp_additions"] = temp_additions
                    st.session_state["temp_text_latex"] = ""
                    st.rerun()

    if exp_groups:
        st.markdown("**Experience**")
        for job_id, group in exp_groups.items():
            meta = group.get("meta", {})
            header = f"{meta.get('company', '')} · {meta.get('role', '')}"
            st.markdown(f"<div class='bullet-meta'>{header}</div>", unsafe_allow_html=True)
            if meta.get("dates") or meta.get("location"):
                st.markdown(
                    f"<div class='art-subtle'>{meta.get('dates', '')} | {meta.get('location', '')}</div>",
                    unsafe_allow_html=True,
                )
            for bullet in group.get("bullets", []):
                bullet_id = bullet.get("id", "")
                is_temp = bullet.get("is_temp", False)
                safe_id = _safe_key(bullet_id)
                default_keep = bullet_id in selected_set or is_temp
                col_keep, col_body = st.columns([0.2, 0.8])
                keep = col_keep.checkbox("Keep", value=default_keep, key=f"keep_{safe_id}")
                display_text = bullet.get("text", "")
                if is_temp:
                    temp_add = temp_additions_by_id.get(bullet_id)
                    if temp_add:
                        display_text = temp_add.get("text_latex", display_text)
                else:
                    display_text = temp_edits.get(bullet_id, display_text)
                badge = " <span class='art-badge art-badge--tag'>TEMP</span>" if is_temp else ""
                col_body.markdown(
                    f"<div class='bullet-card'>{display_text}{badge}</div>",
                    unsafe_allow_html=True,
                )
                if not is_temp and bullet_id in temp_edits:
                    col_body.markdown(
                        "<div class='art-subtle'>Edited for this render</div>",
                        unsafe_allow_html=True,
                    )

                if is_temp:
                    edit_toggle = col_body.checkbox(
                        "Edit temporary bullet",
                        key=f"temp_edit_toggle_{safe_id}",
                    )
                    if edit_toggle:
                        new_text = col_body.text_area(
                            "Temporary bullet text",
                            value=display_text,
                            key=f"temp_edit_text_{safe_id}",
                            height=90,
                        )
                        pending_temp_edits[bullet_id] = new_text
                        btn_cols = col_body.columns([1, 1])
                        if btn_cols[0].button("Save temp edit", key=f"save_temp_{safe_id}"):
                            if not new_text.strip():
                                st.error("Bullet text cannot be empty.")
                            else:
                                temp_add = temp_additions_by_id.get(bullet_id)
                                if temp_add:
                                    temp_add["text_latex"] = new_text
                                    st.session_state["temp_additions"] = temp_additions
                                    st.rerun()
                        if btn_cols[1].button("Delete temp bullet", key=f"del_temp_{safe_id}"):
                            temp_additions = [
                                add for add in temp_additions if _temp_bullet_id(add) != bullet_id
                            ]
                            st.session_state["temp_additions"] = temp_additions
                            st.rerun()
                else:
                    edit_toggle = col_body.checkbox(
                        "Edit for this render",
                        key=f"edit_toggle_{safe_id}",
                    )
                    if edit_toggle:
                        current_text = temp_edits.get(bullet_id, display_text)
                        new_text = col_body.text_area(
                            "Edited text",
                            value=current_text,
                            key=f"edit_text_{safe_id}",
                            height=90,
                        )
                        pending_edits[bullet_id] = new_text
                        btn_cols = col_body.columns([1, 1])
                        if btn_cols[0].button("Save edit", key=f"save_edit_{safe_id}"):
                            if not new_text.strip():
                                st.error("Bullet text cannot be empty.")
                            else:
                                temp_edits[bullet_id] = new_text
                                st.session_state["temp_edits"] = temp_edits
                                st.rerun()
                        if btn_cols[1].button("Clear edit", key=f"clear_edit_{safe_id}"):
                            temp_edits.pop(bullet_id, None)
                            st.session_state["temp_edits"] = temp_edits
                            st.rerun()

                if keep:
                    kept_ids.append(bullet_id)

    if proj_groups:
        st.markdown("**Projects**")
        for project_id, group in proj_groups.items():
            meta = group.get("meta", {})
            header = f"{meta.get('name', '')} · {meta.get('technologies', '')}"
            st.markdown(f"<div class='bullet-meta'>{header}</div>", unsafe_allow_html=True)
            for bullet in group.get("bullets", []):
                bullet_id = bullet.get("id", "")
                is_temp = bullet.get("is_temp", False)
                safe_id = _safe_key(bullet_id)
                default_keep = bullet_id in selected_set or is_temp
                col_keep, col_body = st.columns([0.2, 0.8])
                keep = col_keep.checkbox("Keep", value=default_keep, key=f"keep_{safe_id}")
                display_text = bullet.get("text", "")
                if is_temp:
                    temp_add = temp_additions_by_id.get(bullet_id)
                    if temp_add:
                        display_text = temp_add.get("text_latex", display_text)
                else:
                    display_text = temp_edits.get(bullet_id, display_text)
                badge = " <span class='art-badge art-badge--tag'>TEMP</span>" if is_temp else ""
                col_body.markdown(
                    f"<div class='bullet-card'>{display_text}{badge}</div>",
                    unsafe_allow_html=True,
                )
                if not is_temp and bullet_id in temp_edits:
                    col_body.markdown(
                        "<div class='art-subtle'>Edited for this render</div>",
                        unsafe_allow_html=True,
                    )

                if is_temp:
                    edit_toggle = col_body.checkbox(
                        "Edit temporary bullet",
                        key=f"temp_edit_toggle_{safe_id}",
                    )
                    if edit_toggle:
                        new_text = col_body.text_area(
                            "Temporary bullet text",
                            value=display_text,
                            key=f"temp_edit_text_{safe_id}",
                            height=90,
                        )
                        pending_temp_edits[bullet_id] = new_text
                        btn_cols = col_body.columns([1, 1])
                        if btn_cols[0].button("Save temp edit", key=f"save_temp_{safe_id}"):
                            if not new_text.strip():
                                st.error("Bullet text cannot be empty.")
                            else:
                                temp_add = temp_additions_by_id.get(bullet_id)
                                if temp_add:
                                    temp_add["text_latex"] = new_text
                                    st.session_state["temp_additions"] = temp_additions
                                    st.rerun()
                        if btn_cols[1].button("Delete temp bullet", key=f"del_temp_{safe_id}"):
                            temp_additions = [
                                add for add in temp_additions if _temp_bullet_id(add) != bullet_id
                            ]
                            st.session_state["temp_additions"] = temp_additions
                            st.rerun()
                else:
                    edit_toggle = col_body.checkbox(
                        "Edit for this render",
                        key=f"edit_toggle_{safe_id}",
                    )
                    if edit_toggle:
                        current_text = temp_edits.get(bullet_id, display_text)
                        new_text = col_body.text_area(
                            "Edited text",
                            value=current_text,
                            key=f"edit_text_{safe_id}",
                            height=90,
                        )
                        pending_edits[bullet_id] = new_text
                        btn_cols = col_body.columns([1, 1])
                        if btn_cols[0].button("Save edit", key=f"save_edit_{safe_id}"):
                            if not new_text.strip():
                                st.error("Bullet text cannot be empty.")
                            else:
                                temp_edits[bullet_id] = new_text
                                st.session_state["temp_edits"] = temp_edits
                                st.rerun()
                        if btn_cols[1].button("Clear edit", key=f"clear_edit_{safe_id}"):
                            temp_edits.pop(bullet_id, None)
                            st.session_state["temp_edits"] = temp_edits
                            st.rerun()

                if keep:
                    kept_ids.append(bullet_id)

    if not exp_groups and not proj_groups:
        st.caption("No bullet text available yet. Add bullets in Resume Editor.")

    apply_cols = st.columns([1, 1])
    apply_disabled = not kept_ids
    if apply_cols[0].button(
        "Apply selection and re-render",
        use_container_width=True,
        disabled=apply_disabled,
    ):
        for bullet_id, text in pending_edits.items():
            if isinstance(text, str) and text.strip():
                temp_edits[bullet_id] = text
        if pending_edits:
            st.session_state["temp_edits"] = temp_edits

        for addition in temp_additions:
            bullet_id = _temp_bullet_id(addition)
            if not bullet_id:
                continue
            text = pending_temp_edits.get(bullet_id)
            if isinstance(text, str) and text.strip():
                addition["text_latex"] = text
        if pending_temp_edits:
            st.session_state["temp_additions"] = temp_additions

        removals = [bid for bid in selected_ids if bid not in kept_ids]
        merged_edits = {**temp_edits, **pending_edits}
        edits_payload = {
            bid: text
            for bid, text in merged_edits.items()
            if bid in kept_ids and isinstance(text, str) and text.strip()
        }
        additions_payload = [
            {
                "parent_type": addition.get("parent_type"),
                "parent_id": addition.get("parent_id"),
                "text_latex": addition.get("text_latex", ""),
                "temp_id": addition.get("temp_id"),
            }
            for addition in temp_additions
            if addition.get("parent_type")
            and addition.get("parent_id")
            and addition.get("temp_id")
            and str(addition.get("text_latex", "")).strip()
        ]
        payload: Dict[str, Any] = {"selected_ids": kept_ids}
        if edits_payload or removals or additions_payload:
            payload["temp_overrides"] = {
                "edits": edits_payload,
                "removals": removals,
                "additions": additions_payload,
            }
        ok_apply, _, err_apply = api_request(
            "POST",
            api_url,
            f"/runs/{run.get('run_id')}/render",
            json=payload,
            timeout_s=120,
        )
        if ok_apply:
            st.session_state["rerender_success"] = True
            st.rerun()
        else:
            st.error(err_apply)

    st.subheader("Downloads")
    pdf_url = f"{api_url}{run['pdf_url']}"
    tex_url = f"{api_url}{run['tex_url']}"
    report_url = f"{api_url}{run['report_url']}"

    try:
        pdf_resp = requests.get(pdf_url, timeout=60)
        pdf_resp.raise_for_status()
        st.download_button(
            "Download tailored_resume.pdf",
            data=pdf_resp.content,
            file_name="tailored_resume.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    except Exception:
        st.warning("pdf not ready yet or download failed.")

    try:
        tex_bytes = requests.get(tex_url, timeout=60).content
        st.download_button(
            "Download tailored_resume.tex",
            data=tex_bytes,
            file_name="tailored_resume.tex",
            mime="application/x-tex",
            use_container_width=True,
        )
    except Exception:
        st.warning("tex not ready yet or download failed.")

    try:
        rep_bytes = requests.get(report_url, timeout=60).content
        st.download_button(
            "Download resume_report.json",
            data=rep_bytes,
            file_name="resume_report.json",
            mime="application/json",
            use_container_width=True,
        )
    except Exception:
        st.warning("report.json not ready yet or download failed.")
