from typing import Any, Dict, Tuple

import streamlit as st

from agentic_resume_tailor.ui.common import api_request, get_health_cached


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
