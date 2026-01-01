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
          --art-bg: #f5f7fb;
          --art-bg-2: #eef2f8;
          --art-surface: #ffffff;
          --art-surface-glass: rgba(255, 255, 255, 0.88);
          --art-text: #0f172a;
          --art-muted: #5b6472;
          --art-border: rgba(148, 163, 184, 0.4);
          --art-accent: #0f766e;
          --art-accent-2: #2563eb;
          --art-accent-soft: rgba(15, 118, 110, 0.12);
          --art-shadow: 0 18px 40px rgba(15, 23, 42, 0.12);
          --art-shadow-soft: 0 10px 24px rgba(15, 23, 42, 0.08);
          --art-radius-lg: 18px;
          --art-radius-md: 12px;
          --art-radius-sm: 10px;
        }
        html, body, [class*="css"] {
          font-family: "Manrope", "Space Grotesk", sans-serif;
          color: var(--art-text);
        }
        h1, h2, h3, h4, h5 {
          font-family: "Space Grotesk", sans-serif;
          letter-spacing: -0.02em;
        }
        body {
          background: var(--art-bg);
        }
        .stApp {
          background:
            radial-gradient(1200px 520px at 12% -10%, rgba(37, 99, 235, 0.12), transparent 60%),
            radial-gradient(980px 520px at 88% 0%, rgba(15, 118, 110, 0.14), transparent 55%),
            linear-gradient(180deg, var(--art-bg) 0%, var(--art-bg-2) 100%);
          background-attachment: fixed;
        }
        header[data-testid="stHeader"],
        div[data-testid="stToolbar"],
        div[data-testid="stDecoration"],
        #MainMenu,
        footer {
          display: none;
        }
        section[data-testid="stSidebar"],
        div[data-testid="stSidebarNav"],
        nav[data-testid="stSidebarNav"] {
          display: none;
        }
        .main .block-container {
          padding-top: 1.5rem;
          padding-bottom: 2.5rem;
          max-width: 1200px;
          animation: rise-in 0.45s ease-out;
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
          color: var(--art-text);
          padding: 0;
          line-height: 1.1;
          position: relative;
          top: -8px;
        }
        .topbar-status {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          padding: 8px 12px;
          border-radius: 999px;
          background: var(--art-surface);
          border: 1px solid var(--art-border);
          font-size: 0.85rem;
          font-weight: 600;
          box-shadow: var(--art-shadow-soft);
          white-space: nowrap;
        }
        .art-card {
          background: var(--art-surface-glass);
          border: 1px solid var(--art-border);
          border-radius: var(--art-radius-lg);
          padding: 16px 18px;
          box-shadow: var(--art-shadow-soft);
          backdrop-filter: blur(6px);
        }
        .art-card + .art-card {
          margin-top: 12px;
        }
        .art-title {
          font-size: 1.1rem;
          font-weight: 600;
          color: var(--art-text);
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
          font-weight: 600;
          margin: 2px 6px 2px 0;
          border: 1px solid transparent;
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
          position: relative;
          display: flex;
          align-items: center;
          gap: 10px;
          padding: 10px 12px;
          border-radius: var(--art-radius-sm);
          text-decoration: none;
          color: var(--art-text);
          font-weight: 600;
          border: 1px solid transparent;
          transition: all 0.2s ease;
        }
        .nav-link:hover {
          background: var(--art-accent-soft);
          border-color: rgba(15, 118, 110, 0.2);
        }
        .nav-link.active {
          background: linear-gradient(120deg, rgba(15, 118, 110, 0.96), rgba(37, 99, 235, 0.96));
          color: #ffffff;
          box-shadow: var(--art-shadow-soft);
          border-color: transparent;
        }
        .nav-link.active::after {
          content: "";
          position: absolute;
          right: 12px;
          width: 7px;
          height: 7px;
          border-radius: 999px;
          background: rgba(255, 255, 255, 0.9);
        }
        .sidebar-title {
          font-size: 0.95rem;
          font-weight: 600;
          color: var(--art-text);
          margin: 0 0 6px 0;
        }
        .sidebar-card {
          background: var(--art-surface);
          border: 1px solid var(--art-border);
          border-radius: var(--art-radius-md);
          padding: 12px;
          margin-bottom: 12px;
          box-shadow: var(--art-shadow-soft);
        }
        div[data-testid="stPageLink"] {
          margin-bottom: 0;
        }
        div[data-testid="stPageLink"] a {
          display: flex;
          align-items: center;
          gap: 10px;
          padding: 10px 12px;
          border-radius: var(--art-radius-sm);
          text-decoration: none;
          color: var(--art-text);
          font-weight: 600;
          border: 1px solid rgba(148, 163, 184, 0.6);
          background: rgba(255, 255, 255, 0.78);
          box-shadow: 0 8px 18px rgba(15, 23, 42, 0.08);
          transition: all 0.2s ease;
        }
        div[data-testid="stPageLink"] a:hover {
          background: linear-gradient(120deg, rgba(37, 99, 235, 0.16), rgba(15, 118, 110, 0.18));
          border-color: rgba(15, 118, 110, 0.55);
          color: #0f172a;
          box-shadow: 0 12px 22px rgba(15, 23, 42, 0.12);
        }
        div[data-testid="stPageLink"] a[aria-disabled="true"],
        div[data-testid="stPageLink"] a[aria-current="page"] {
          background: linear-gradient(120deg, #0f766e 0%, #1d4ed8 100%);
          color: #ffffff !important;
          border-color: rgba(15, 118, 110, 0.6);
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
          background: var(--art-surface);
          border: 1px solid var(--art-border);
          border-radius: var(--art-radius-md);
          padding: 10px 12px;
          box-shadow: var(--art-shadow-soft);
        }
        .bullet-meta {
          color: var(--art-muted);
          font-size: 0.8rem;
          margin-bottom: 6px;
        }
        section[data-testid="stForm"] {
          background: var(--art-surface);
          border: 1px solid var(--art-border);
          border-radius: var(--art-radius-lg);
          padding: 16px;
          box-shadow: var(--art-shadow);
        }
        div[data-testid="stExpander"] {
          border-radius: var(--art-radius-lg);
          border: 1px solid var(--art-border);
          background: var(--art-surface);
          box-shadow: var(--art-shadow-soft);
        }
        div[data-testid="stAlert"] {
          border-radius: var(--art-radius-md);
          border: 1px solid var(--art-border);
          box-shadow: var(--art-shadow-soft);
        }
        div[data-testid="stDivider"] {
          margin: 1.5rem 0;
        }
        textarea, input, select {
          background-color: rgba(255, 255, 255, 0.96) !important;
          color: var(--art-text) !important;
          border: 1px solid rgba(148, 163, 184, 0.6) !important;
          border-radius: var(--art-radius-sm) !important;
          padding: 10px 12px !important;
          box-shadow: inset 0 0 0 1px rgba(148, 163, 184, 0.2);
        }
        textarea::placeholder, input::placeholder {
          color: rgba(91, 100, 114, 0.8) !important;
        }
        textarea:focus, input:focus, select:focus {
          border-color: var(--art-accent-2) !important;
          box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15) !important;
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
          transition: transform 0.12s ease, box-shadow 0.12s ease, background 0.12s ease;
        }
        button[kind="primary"] {
          background: linear-gradient(120deg, var(--art-accent) 0%, var(--art-accent-2) 100%) !important;
          color: #ffffff !important;
          border: 1px solid transparent !important;
          box-shadow: 0 12px 24px rgba(15, 118, 110, 0.2);
        }
        button[kind="primary"]:hover {
          transform: translateY(-1px);
          box-shadow: 0 14px 26px rgba(15, 118, 110, 0.25);
        }
        button[kind="secondary"], button[kind="tertiary"] {
          background: var(--art-surface) !important;
          color: var(--art-text) !important;
          border: 1px solid var(--art-border) !important;
        }
        button[kind="secondary"]:hover, button[kind="tertiary"]:hover {
          border-color: rgba(37, 99, 235, 0.4) !important;
          transform: translateY(-1px);
          box-shadow: var(--art-shadow-soft);
        }
        a {
          color: var(--art-accent-2);
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


def _render_topbar(api_url: str, active_page: str) -> Tuple[bool, Any]:
    """Render top navigation + health.

    Args:
        api_url: Base URL for the API.
        active_page: Current page name for navigation state.

    Returns:
        Tuple of results including health info.
    """
    ok, info = get_health_cached(api_url)
    status_icon = "🟢" if ok else "🔴"
    status = "Healthy" if ok else "Down"

    col_brand, col_nav, col_status, col_action = st.columns(
        [2.4, 4.6, 1.4, 1.1])
    col_brand.markdown(
        "<div class='topbar-brand'>Agentic Resume Tailor</div>", unsafe_allow_html=True)

    nav_cols = col_nav.columns([1.1, 1.6, 1.1])
    nav_items = [
        ("Generate", "✨", "app.py"),
        ("Resume Editor", "🧩", "pages/Resume_Editor.py"),
        ("Settings", "⚙️", "pages/Settings.py"),
    ]
    if hasattr(st, "page_link"):
        for col, (label, nav_icon, path) in zip(nav_cols, nav_items):
            with col:
                st.page_link(
                    path,
                    label=label,
                    icon=nav_icon,
                    disabled=(label == active_page),
                )
    else:
        for col, (label, nav_icon, path) in zip(nav_cols, nav_items):
            if col.button(
                f"{nav_icon} {label}",
                key=f"nav_{label}",
                use_container_width=True,
                disabled=(label == active_page),
            ):
                if hasattr(st, "switch_page"):
                    st.switch_page(path)
                else:
                    st.warning("Upgrade Streamlit to enable navigation.")

    col_status.markdown(
        f"<div class='topbar-status'>{status_icon} {status}</div>",
        unsafe_allow_html=True,
    )
    if col_action.button("Re-check", key="topbar_recheck", use_container_width=True):
        st.session_state["_health_force_refresh"] = time.time()
        ok, info = get_health_cached(api_url)

    st.divider()
    return ok, info
