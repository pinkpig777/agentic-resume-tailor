import streamlit as st

from agentic_resume_tailor.settings import get_settings
from agentic_resume_tailor.ui.common import _inject_app_styles, _render_brand_bar, _render_sidebar
from agentic_resume_tailor.ui.pages.settings import render_settings_page

settings = get_settings()
api_url = settings.api_url.rstrip("/")

st.set_page_config(
    page_title="Settings | Agentic Resume Tailor",
    layout="wide",
    initial_sidebar_state="expanded",
)
_inject_app_styles()
_render_brand_bar()
_render_sidebar(api_url, active_page="Settings")
render_settings_page(api_url)
