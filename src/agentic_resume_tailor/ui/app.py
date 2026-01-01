import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

import requests
import streamlit as st

from agentic_resume_tailor.settings import get_settings

GEN_EXECUTOR = ThreadPoolExecutor(max_workers=1)


# ----------------------------
# Health check
# ----------------------------
def check_server_health(api_base: str, timeout_s: float = 1.5) -> Tuple[bool, Any]:
    """Check the API health endpoint and return status.

    Args:
        api_base: Base URL for the API.
        timeout_s: The timeout s value (optional).

    Returns:
        True if the condition is met, otherwise False.
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
    """Return cached health status with a TTL.

    Args:
        api_base: Base URL for the API.
        ttl_s: The ttl s value (optional).

    Returns:
        Tuple of results.
    """
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
    """Call the API and normalize success/error responses.

    Args:
        method: The method value.
        api_base: Base URL for the API.
        path: Filesystem path.
        timeout_s: The timeout s value (optional).
        **kwargs: Additional keyword arguments.

    Returns:
        Tuple of results.
    """
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
    """Fetch settings from the API.

    Args:
        api_url: Base URL for the API.

    Returns:
        Tuple of results.
    """
    ok, data, err = api_request("GET", api_url, "/settings", timeout_s=10)
    if ok and isinstance(data, dict):
        return True, data, ""
    return False, {}, err or "Failed to load settings."


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


def _render_bullet_controls(api_url: str, section: str, parent_id: str, bullet: dict) -> None:
    """Render edit/delete controls for a bullet.

    Args:
        api_url: Base URL for the API.
        section: The section value.
        parent_id: Parent identifier.
        bullet: The bullet value.
    """
    bullet_id = bullet.get("id", "")
    text = bullet.get("text_latex", "")
    edit_key = f"edit_{section}_{parent_id}_{bullet_id}"
    text_key = f"text_{section}_{parent_id}_{bullet_id}"

    col_text, col_edit, col_del = st.columns([8, 1, 1])
    col_text.markdown(f"<div class='bullet-card'>{text}</div>", unsafe_allow_html=True)
    if col_edit.button("Edit", key=f"edit_btn_{edit_key}"):
        st.session_state[edit_key] = not st.session_state.get(edit_key, False)
    if col_del.button("Delete", key=f"del_btn_{edit_key}"):
        ok, _, err = api_request("DELETE", api_url, f"/{section}/{parent_id}/bullets/{bullet_id}")
        if ok:
            _set_editor_message("success", f"Deleted {section} bullet.")
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
                    _set_editor_message("success", f"Updated {section} bullet.")
                    st.rerun()
                else:
                    st.error(err)


def _inject_app_styles() -> None:
    """Inject shared CSS for a unified UI theme."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&display=swap');
        html, body, [class*="css"] {
          font-family: "Space Grotesk", sans-serif;
        }
        .stApp {
          background: #f4f5f7;
        }
        .main .block-container {
          padding-top: 1.75rem;
        }
        .art-card {
          background: #ffffff;
          border: 1px solid #e5e7eb;
          border-radius: 16px;
          padding: 16px 18px;
          box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        }
        .art-card + .art-card {
          margin-top: 12px;
        }
        .art-title {
          font-size: 1.1rem;
          font-weight: 600;
          color: #111827;
          margin: 0 0 8px 0;
        }
        .art-subtle {
          color: #6b7280;
          font-size: 0.9rem;
        }
        .art-badge {
          display: inline-block;
          padding: 4px 10px;
          border-radius: 999px;
          font-size: 0.78rem;
          font-weight: 500;
          margin: 2px 6px 2px 0;
        }
        .art-badge--must {
          background: #fee2e2;
          color: #991b1b;
          border: 1px solid #fecaca;
        }
        .art-badge--nice {
          background: #e0f2fe;
          color: #075985;
          border: 1px solid #bae6fd;
        }
        .art-badge--tag {
          background: #ecfdf3;
          color: #027a48;
          border: 1px solid #abefc6;
        }
        .nav-link {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 8px 10px;
          border-radius: 10px;
          text-decoration: none;
          color: #111827;
          font-weight: 500;
        }
        .nav-link:hover {
          background: #e5e7eb;
        }
        .nav-link.active {
          background: #111827;
          color: #ffffff;
        }
        .sidebar-card {
          background: #ffffff;
          border: 1px solid #e5e7eb;
          border-radius: 14px;
          padding: 12px;
          margin-bottom: 12px;
        }
        .bullet-card {
          background: #ffffff;
          border: 1px solid #e5e7eb;
          border-radius: 12px;
          padding: 10px 12px;
        }
        .bullet-meta {
          color: #6b7280;
          font-size: 0.8rem;
          margin-bottom: 6px;
        }
        section[data-testid="stForm"] {
          background: #ffffff;
          border: 1px solid #e5e7eb;
          border-radius: 16px;
          padding: 16px;
        }
        div[data-testid="stExpander"] {
          border-radius: 14px;
          border: 1px solid #e5e7eb;
          background: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar(api_url: str) -> Tuple[bool, Any, str]:
    """Render sidebar health + navigation.

    Args:
        api_url: Base URL for the API.

    Returns:
        Tuple of results including health and active page.
    """
    ok, info = get_health_cached(api_url)
    icon = "🟢" if ok else "🔴"
    status = "Healthy" if ok else "Down"
    st.sidebar.markdown(
        f"""
        <div class="sidebar-card">
          <div style="font-weight:600; margin-bottom:6px;">Server Health</div>
          <div style="display:flex; align-items:center; gap:6px;">
            <span>{icon}</span>
            <span>{status}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.sidebar.button("Re-check", use_container_width=True):
        st.session_state["_health_force_refresh"] = time.time()
        ok, info = get_health_cached(api_url)

    params = st.query_params
    page_val = params.get("page", "Generate")
    page = page_val[0] if isinstance(page_val, list) else page_val
    nav_items = [
        ("Generate", "✨"),
        ("Resume Editor", "🧩"),
        ("Settings", "⚙️"),
    ]
    st.sidebar.markdown(
        f"""
        <div class="sidebar-card">
          <div style="font-weight:600; margin-bottom:6px;">Navigation</div>
          {
            "".join(
                f"<a class='nav-link {'active' if name == page else ''}' href='?page={name}'>{icon} {name}</a>"
                for name, icon in nav_items
            )
        }
        </div>
        """,
        unsafe_allow_html=True,
    )
    return ok, info, page


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


def render_settings_page(api_url: str) -> None:
    """Render the Settings page UI.

    Args:
        api_url: Base URL for the API.
    """
    st.header("Settings")
    ok, _ = get_health_cached(api_url)
    if not ok:
        st.error("API server is down. Start the backend to edit settings.", icon="🚨")
        return

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
            jd_model_current = app_settings.get("jd_model", "") or ""
            jd_model_options = [
                "gpt-5.2-2025-12-11",
                "gpt-5-mini-2025-08-07",
                "gpt-5-nano-2025-08-07",
                "gpt-4.1",
                "gpt-4.1-2025-04-14",
                "gpt-4.1-mini",
                "gpt-4.1-mini-2025-04-14",
                "gpt-4.1-nano",
                "gpt-4.1-nano-2025-04-14",
                "gpt-4o",
                "gpt-4o-2024-08-06",
                "gpt-4o-mini",
                "gpt-4o-mini-2024-07-18",
            ]
            if jd_model_current and jd_model_current not in jd_model_options:
                jd_model_options = [jd_model_current] + jd_model_options
            jd_model = st.selectbox(
                "JD model",
                options=jd_model_options,
                index=jd_model_options.index(jd_model_current)
                if jd_model_current in jd_model_options
                else 0,
                help="OpenAI model used for JD parsing.",
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
            quant_bonus_per_hit = st.number_input(
                "Quant bonus per hit",
                min_value=0.0,
                max_value=0.5,
                value=float(app_settings.get("quant_bonus_per_hit", 0.05) or 0.05),
                step=0.01,
                help="Bonus added per quantitative pattern match in a bullet.",
            )
            quant_bonus_cap = st.number_input(
                "Quant bonus cap",
                min_value=0.0,
                max_value=1.0,
                value=float(app_settings.get("quant_bonus_cap", 0.2) or 0.2),
                step=0.05,
                help="Maximum total bonus allowed per bullet.",
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
                "quant_bonus_per_hit": quant_bonus_per_hit,
                "quant_bonus_cap": quant_bonus_cap,
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
        with st.expander(title, expanded=False):
            st.caption(meta)
            col_edit, col_delete = st.columns([1, 1])
            edit_key = f"edit_edu_{edu_id}"
            if col_edit.button("Edit education", key=f"toggle_{edit_key}"):
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
            exp_bullets = st.text_area("Bullets (one per line)", key="new_exp_bullets", height=120)
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

    for exp in experiences or []:
        job_id = exp.get("job_id", "")
        title = f"{exp.get('company', '')} — {exp.get('role', '')}"
        meta = f"{exp.get('dates', '')} · {exp.get('location', '')}"
        with st.expander(title, expanded=True):
            st.caption(meta)
            col_edit, col_delete = st.columns([1, 1])
            edit_key = f"edit_exp_{job_id}"
            if col_edit.button("Edit experience", key=f"toggle_{edit_key}"):
                st.session_state[edit_key] = not st.session_state.get(edit_key, False)
            if col_delete.button("Delete experience", key=f"delete_exp_{job_id}"):
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
                            _set_editor_message("success", "Updated experience.")
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
                        _set_editor_message("success", "Added bullet to experience.")
                        st.rerun()
                    else:
                        st.error(err)

    st.subheader("Projects")
    with st.expander("Add Project", expanded=False):
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

    for proj in projects or []:
        project_id = proj.get("project_id", "")
        title = f"{proj.get('name', '')} — {proj.get('technologies', '')}"
        with st.expander(title, expanded=True):
            col_edit, col_delete = st.columns([1, 1])
            edit_key = f"edit_proj_{project_id}"
            if col_edit.button("Edit project", key=f"toggle_{edit_key}"):
                st.session_state[edit_key] = not st.session_state.get(edit_key, False)
            if col_delete.button("Delete project", key=f"delete_proj_{project_id}"):
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
                        _set_editor_message("success", "Added bullet to project.")
                        st.rerun()
                    else:
                        st.error(err)


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
        st.info("Generation running. You can navigate to other pages while this runs.")
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


def main() -> None:
    """Run the Streamlit app."""
    settings = get_settings()
    api_url = settings.api_url.rstrip("/")

    st.set_page_config(page_title="AI Resume Agent", layout="wide")
    _inject_app_styles()
    st.title("Agentic Resume Tailor")

    ok, info, page = _render_sidebar(api_url)

    if page == "Resume Editor":
        render_resume_editor(api_url)
        return
    if page == "Settings":
        render_settings_page(api_url)
        return

    render_generate_page(api_url)


if __name__ == "__main__":
    main()
