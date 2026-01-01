import streamlit as st

from agentic_resume_tailor.settings import get_settings
from agentic_resume_tailor.ui.common import _inject_app_styles, _render_topbar
from agentic_resume_tailor.ui.pages.generate import render_generate_page
from agentic_resume_tailor.ui.pages.resume_editor import render_resume_editor
from agentic_resume_tailor.ui.pages.settings import render_settings_page


def main() -> None:
    """Run the Streamlit app."""
    settings = get_settings()
    api_url = settings.api_url.rstrip("/")

    st.set_page_config(
        page_title="Agentic Resume Tailor",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _inject_app_styles()
    _render_topbar(api_url, active_page="Generate")
    render_generate_page(api_url)
