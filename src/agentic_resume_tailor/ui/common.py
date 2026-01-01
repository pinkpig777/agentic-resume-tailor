import time
from typing import Any, Dict, Tuple

import requests
import streamlit as st


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
        textarea, input, select {
          background-color: #ffffff !important;
          color: #111827 !important;
          border: 1px solid #cbd5e1 !important;
          box-shadow: inset 0 0 0 1px #cbd5e1;
        }
        textarea::placeholder, input::placeholder {
          color: #6b7280 !important;
        }
        textarea:focus, input:focus, select:focus {
          border-color: #2563eb !important;
          box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.15) !important;
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
