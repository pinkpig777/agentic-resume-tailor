import streamlit as st

from agentic_resume_tailor.settings import get_settings
from agentic_resume_tailor.ui.common import _inject_app_styles, _render_topbar
from agentic_resume_tailor.ui.pages.settings import render_settings_page

settings = get_settings()
api_url = settings.api_url.rstrip("/")

st.set_page_config(
    page_title="Settings | Agentic Resume Tailor",
    layout="wide",
    initial_sidebar_state="collapsed",
)
_inject_app_styles()
_render_topbar(api_url, active_page="Settings")
render_settings_page(api_url)
