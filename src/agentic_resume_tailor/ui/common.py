import time
from typing import Any, Dict, Tuple

import requests
import streamlit as st
from concurrent.futures import Future


def check_server_health(api_base: str, timeout_s: float = 10.0) -> Tuple[bool, Any]:
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


def get_health_cached(api_base: str, ttl_s: float = 6.0) -> Tuple[bool, Any]:
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

    future = st.session_state.get("generate_future")
    if isinstance(future, Future) and not future.done():
        return st.session_state.get("_health_ok", False), st.session_state.get(
            "_health_info", "not checked"
        )

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
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
        :root {
          --art-bg: #ffffff;
          --art-bg-2: #f9fafb;
          --art-surface: #ffffff;
          --art-surface-glass: #ffffff;
          --art-text: #000000;
          --art-muted: #4b5563;
          --art-border: #e5e7eb;
          --art-accent: #000000;
          --art-accent-2: #000000;
          --art-accent-soft: #f3f4f6;
          --art-shadow: none;
          --art-shadow-soft: none;
          --art-radius-lg: 8px;
          --art-radius-md: 6px;
          --art-radius-sm: 4px;
        }
        html, body, [class*="css"] {
          font-family: "Manrope", "Space Grotesk", sans-serif;
          color: var(--art-text);
        }
        h1, h2, h3, h4, h5 {
          font-family: "Space Grotesk", sans-serif;
          letter-spacing: -0.02em;
          color: #000000;
          font-weight: 700;
        }
        body {
          background: var(--art-bg);
        }
        .stApp {
          background: #ffffff;
        }
        div[data-testid="stDecoration"],
        #MainMenu,
        footer {
          display: none;
        }
        header[data-testid="stHeader"] {
          background: transparent;
          height: 0;
          border: 0;
        }
        div[data-testid="stToolbar"] {
          background: transparent;
          height: 0;
          padding: 0;
          margin: 0;
        }
        div[data-testid="stToolbar"] > div {
          height: 0;
          padding: 0;
          margin: 0;
        }
        div[data-testid="collapsedControl"] {
          position: fixed;
          top: 12px;
          left: 12px;
          width: 36px;
          height: 36px;
          border-radius: 999px;
          background: #ffffff;
          border: 2px solid #000000;
          display: flex;
          align-items: center;
          justify-content: center;
          box-shadow: none;
          z-index: 1000;
        }
        div[data-testid="collapsedControl"] button {
          color: #000000;
        }
        div[data-testid="collapsedControl"] svg {
          width: 18px;
          height: 18px;
        }
        div[data-testid="stSidebarNav"],
        nav[data-testid="stSidebarNav"] {
          display: none;
        }
        section[data-testid="stSidebar"] > div {
          background: #ffffff;
          border-right: 2px solid #e5e7eb;
        }
        .main .block-container {
          padding-top: 0.2rem;
          padding-bottom: 2.5rem;
          max-width: 1200px;
          animation: rise-in 0.45s ease-out;
        }
        .main h1 {
          margin-top: 0.15rem;
        }
        .app-topbar {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 0.1rem 0 0.7rem;
          margin-bottom: 1.1rem;
          border-bottom: 2px solid #e5e7eb;
        }
        @keyframes rise-in {
          from {
            opacity: 0;
            transform: translateY(8px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        .topbar-brand {
          font-size: 2rem;
          font-weight: 700;
          letter-spacing: -0.02em;
          color: #000000;
          padding: 0;
          line-height: 1.05;
          display: flex;
          align-items: center;
          height: auto;
        }
        .topbar-status {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          gap: 0;
          height: 44px;
          padding: 0;
          border-radius: 999px;
          background: transparent;
          border: 0;
          font-size: 0.85rem;
          font-weight: 600;
          white-space: nowrap;
          width: 100%;
        }
        .art-card {
          background: #ffffff;
          border: 2px solid #e5e7eb;
          border-radius: var(--art-radius-lg);
          padding: 16px 18px;
        }
        .art-card + .art-card {
          margin-top: 12px;
        }
        .art-title {
          font-size: 1.1rem;
          font-weight: 700;
          color: #000000;
          margin: 0 0 8px 0;
        }
        .art-subtle {
          color: var(--art-muted);
          font-size: 0.9rem;
        }
        .art-badge {
          display: inline-block;
          padding: 4px 10px;
          border-radius: 999px;
          font-size: 0.78rem;
          font-weight: 700;
          margin: 2px 6px 2px 0;
          border: 2px solid transparent;
        }
        .art-badge--must {
          background: #ffffff;
          color: #b91c1c;
          border: 2px solid #b91c1c;
        }
        .art-badge--nice {
          background: #ffffff;
          color: #0369a1;
          border: 2px solid #0369a1;
        }
        .art-badge--tag {
          background: #ffffff;
          color: #15803d;
          border: 2px solid #15803d;
        }
        .nav-link {
          position: relative;
          display: flex;
          align-items: center;
          gap: 10px;
          padding: 10px 12px;
          border-radius: var(--art-radius-sm);
          text-decoration: none;
          color: #000000;
          font-weight: 600;
          border: 2px solid transparent;
          transition: all 0.2s ease;
        }
        .nav-link:hover {
          background: #f3f4f6;
          border-color: #000000;
        }
        .nav-link.active {
          background: #000000;
          color: #ffffff;
          border-color: #000000;
        }
        .sidebar-title {
          font-size: 0.95rem;
          font-weight: 600;
          color: #000000;
          margin: 0 0 6px 0;
        }
        .sidebar-card {
          background: #ffffff;
          border: 2px solid #e5e7eb;
          border-radius: var(--art-radius-md);
          padding: 12px;
          margin-bottom: 12px;
        }
        div[data-testid="stPageLink"] {
          margin-bottom: 0;
        }
        div[data-testid="stPageLink"] a {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 10px;
          height: 44px;
          padding: 0 12px;
          line-height: 1;
          border-radius: var(--art-radius-sm);
          text-decoration: none;
          color: #000000;
          font-weight: 600;
          border: 2px solid #e5e7eb;
          background: #ffffff;
          transition: all 0.2s ease;
        }
        div[data-testid="stPageLink"] a:hover {
          background: #f3f4f6;
          border-color: #000000;
          color: #000000;
        }
        div[data-testid="stPageLink"] a[aria-disabled="true"],
        div[data-testid="stPageLink"] a[aria-current="page"] {
          background: #000000;
          color: #ffffff !important;
          border-color: #000000;
          opacity: 1 !important;
          cursor: default;
          pointer-events: none;
        }
        div[data-testid="stPageLink"] a[aria-disabled="true"] *,
        div[data-testid="stPageLink"] a[aria-current="page"] * {
          color: #ffffff !important;
          opacity: 1 !important;
          filter: none !important;
        }
        .bullet-card {
          background: #ffffff;
          border: 2px solid #e5e7eb;
          border-radius: var(--art-radius-md);
          padding: 10px 12px;
        }
        .bullet-meta {
          color: var(--art-muted);
          font-size: 0.8rem;
          margin-bottom: 6px;
        }
        section[data-testid="stForm"] {
          background: #ffffff;
          border: 2px solid #e5e7eb;
          border-radius: var(--art-radius-lg);
          padding: 16px;
        }
        div[data-testid="stExpander"] {
          border-radius: var(--art-radius-lg);
          border: 2px solid #e5e7eb;
          background: #ffffff;
        }
        div[data-testid="stAlert"] {
          border-radius: var(--art-radius-md);
          border: 2px solid #e5e7eb;
        }
        div[data-testid="stDivider"] {
          margin: 1.5rem 0;
        }
        textarea, input, select {
          background-color: #ffffff !important;
          color: #000000 !important;
          border: 2px solid #e5e7eb !important;
          border-radius: var(--art-radius-sm) !important;
          padding: 10px 12px !important;
        }
        textarea::placeholder, input::placeholder {
          color: #9ca3af !important;
        }
        textarea:focus, input:focus, select:focus {
          border-color: #000000 !important;
          box-shadow: none !important;
        }
        div[data-baseweb="select"] input {
          background: transparent !important;
          border: 0 !important;
          box-shadow: none !important;
          padding: 0 !important;
          margin: 0 !important;
        }
        div[data-baseweb="select"] input:focus {
          box-shadow: none !important;
        }
        button, .stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
          border-radius: var(--art-radius-sm) !important;
          font-weight: 600 !important;
          transition: all 0.12s ease;
        }
        button[kind="primary"] {
          background: #000000 !important;
          color: #ffffff !important;
          border: 2px solid #000000 !important;
          box-shadow: none !important;
        }
        button[kind="primary"]:hover {
          background: #333333 !important;
          border-color: #333333 !important;
        }
        button[kind="secondary"], button[kind="tertiary"] {
          background: #ffffff !important;
          color: #000000 !important;
          border: 2px solid #e5e7eb !important;
          height: 44px;
          padding: 0 16px !important;
          display: inline-flex;
          align-items: center;
          justify-content: center;
        }
        button[kind="secondary"]:hover, button[kind="tertiary"]:hover {
          border-color: #000000 !important;
          background: #f9fafb !important;
          box-shadow: none !important;
        }
        a {
          color: #000000;
          text-decoration: underline;
        }
        @media (max-width: 768px) {
          .main .block-container {
            padding: 1.1rem 1rem 2rem;
          }
          .nav-link {
            padding: 10px;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_brand_bar(title: str = "Agentic Resume Tailor") -> None:
    """Render the app brand in a compact top bar."""
    st.markdown(
        f"""
        <div class="app-topbar">
          <div class="topbar-brand">{title}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar(api_url: str, active_page: str) -> Tuple[bool, Any]:
    """Render sidebar navigation + health.

    Args:
        api_url: Base URL for the API.
        active_page: Current page name for navigation state.

    Returns:
        Tuple of results including health info.
    """
    ok, info = get_health_cached(api_url)
    status = "Healthy" if ok else "Down"

    st.sidebar.markdown(
        f"""
        <div class="sidebar-card">
          <div class="sidebar-title">Server Health</div>
          <div class="topbar-status">{status}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.sidebar.button("Re-check", use_container_width=True):
        st.session_state["_health_force_refresh"] = time.time()
        ok, info = get_health_cached(api_url)

    st.sidebar.markdown("<div class='sidebar-title'>Navigation</div>", unsafe_allow_html=True)
    nav_items = [
        ("Generate", "✨", "app.py"),
        ("Resume Editor", "🧩", "pages/Resume_Editor.py"),
        ("Settings", "⚙️", "pages/Settings.py"),
    ]
    if hasattr(st.sidebar, "page_link"):
        for label, nav_icon, path in nav_items:
            st.sidebar.page_link(
                path,
                label=label,
                icon=nav_icon,
                disabled=(label == active_page),
            )
    else:
        for label, nav_icon, path in nav_items:
            if st.sidebar.button(
                f"{nav_icon} {label}",
                key=f"nav_{label}",
                use_container_width=True,
                disabled=(label == active_page),
            ):
                if hasattr(st, "switch_page"):
                    st.switch_page(path)
                else:
                    st.sidebar.warning("Upgrade Streamlit to enable navigation.")
    return ok, info
