import time
from typing import Any, Tuple

import pandas as pd
import requests
import streamlit as st

from agentic_resume_tailor.settings import get_settings

settings = get_settings()
API_URL = settings.api_url.rstrip("/")

st.set_page_config(page_title="AI Resume Agent", layout="wide")
st.title("AI Resume Agent")


# ----------------------------
# Health check
# ----------------------------
def check_server_health(api_base: str, timeout_s: float = 1.5) -> Tuple[bool, Any]:
    """
    Returns (ok: bool, payload_or_error: dict|str)
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


st.sidebar.markdown("## Server Health")
col1, col2 = st.sidebar.columns([1, 1])

if col2.button("Re-check"):
    st.session_state["_health_force_refresh"] = time.time()

ok, info = get_health_cached(API_URL)
if not ok:
    st.error(
        f"API server is DOWN: {API_URL}\n\n"
        f"Health check failed: {info}\n\n"
        "Start FastAPI (server.py) and click Re-check in the sidebar.",
        icon="🚨",
    )
    st.stop()
st.sidebar.caption(f"Health URL: {API_URL}/health")
if ok:
    st.sidebar.success(f"UP: {API_URL}")
    st.sidebar.caption(f"Response: {info}")
else:
    st.sidebar.error(f"DOWN: {API_URL}")
    st.sidebar.caption(f"Error: {info}")

st.sidebar.divider()


# ----------------------------
# Settings
# ----------------------------
with st.sidebar:
    st.subheader("Settings")
    max_bullets = st.slider("Max bullets on page", min_value=8, max_value=24, value=16, step=1)
    max_iters = st.slider("Max loop iterations", min_value=1, max_value=6, value=3, step=1)
    threshold = st.slider(
        "Stop threshold (final score)", min_value=0, max_value=100, value=80, step=1
    )
    alpha = st.slider(
        "Alpha (retrieval weight)", min_value=0.0, max_value=1.0, value=0.7, step=0.05
    )
    must_weight = st.slider(
        "Must-have weight (coverage)", min_value=0.0, max_value=1.0, value=0.8, step=0.05
    )
    per_query_k = st.slider("per_query_k", min_value=3, max_value=30, value=10, step=1)
    final_k = st.slider("final_k", min_value=10, max_value=150, value=30, step=5)
    boost_weight = st.slider(
        "Boost query weight", min_value=0.5, max_value=3.0, value=1.6, step=0.1
    )
    boost_top_n_missing = st.slider(
        "Boost top-N missing must-have", min_value=1, max_value=20, value=6, step=1
    )


# ----------------------------
# JD input
# ----------------------------
st.subheader("Job Description")
jd_text = st.text_area("Paste the JD here", height=260, placeholder="Paste a job description...")


colA, colB = st.columns([1, 2], gap="large")

with colA:
    generate_disabled = not ok

    if generate_disabled:
        st.warning("API server is DOWN. Start FastAPI first, then Re-check.")

    if st.button("Generate", type="primary", use_container_width=True, disabled=generate_disabled):
        if not jd_text.strip():
            st.error("JD is empty.")
        else:
            payload = {
                "jd_text": jd_text.strip(),
                "max_bullets": max_bullets,
                "per_query_k": per_query_k,
                "final_k": final_k,
                "max_iters": max_iters,
                "threshold": threshold,
                "alpha": alpha,
                "must_weight": must_weight,
                "boost_weight": boost_weight,
                "boost_top_n_missing": boost_top_n_missing,
            }
            with st.spinner("Running agent..."):
                try:
                    resp = requests.post(f"{API_URL}/generate", json=payload, timeout=600)
                    resp.raise_for_status()
                    out = resp.json()
                    st.session_state["last_run"] = out
                except Exception as e:
                    st.exception(e)


with colB:
    st.subheader("Run Output")
    run = st.session_state.get("last_run")

    if not run:
        st.info("No run yet. Click Generate.")
    else:
        run_id = run.get("run_id")
        st.write(
            {
                "run_id": run_id,
                "profile_used": run.get("profile_used"),
                "best_iteration_index": run.get("best_iteration_index"),
            }
        )

        # Fetch report
        report = None
        try:
            r = requests.get(f"{API_URL}{run['report_url']}", timeout=60)
            r.raise_for_status()
            report = r.json()
        except Exception as e:
            st.warning("Failed to load report.json from API")
            st.exception(e)

        if report:
            best = report.get("best_score")
            if best:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Final score", best.get("final_score"))
                c2.metric("Retrieval", round(float(best.get("retrieval_score", 0.0)), 3))
                c3.metric(
                    "Coverage (bullets)", round(float(best.get("coverage_bullets_only", 0.0)), 3)
                )
                c4.metric("Coverage (all+skills)", round(float(best.get("coverage_all", 0.0)), 3))

            iters = report.get("iterations", [])
            if iters:
                rows = []
                for it in iters:
                    scores = it.get("scores") or {}
                    missing = it.get("missing") or {}
                    rows.append(
                        {
                            "iter": it.get("iteration"),
                            "final": scores.get("final"),
                            "retrieval": scores.get("retrieval"),
                            "cov_bullets": scores.get("coverage_bullets_only"),
                            "cov_all": scores.get("coverage_all"),
                            "missing_must_bullets": len(missing.get("must_bullets_only") or []),
                            "missing_nice_bullets": len(missing.get("nice_bullets_only") or []),
                        }
                    )

                df = pd.DataFrame(rows)
                st.subheader("Iterations")
                st.dataframe(df, use_container_width=True, hide_index=True)

                st.subheader("Iteration details")
                idxs = [int(x.get("iteration", 0)) for x in iters]
                default_idx = min(int(report.get("best_iteration_index", 0)), max(len(idxs) - 1, 0))
                pick = st.selectbox("Pick iteration", idxs, index=default_idx)

                chosen = next((x for x in iters if int(x.get("iteration", -1)) == int(pick)), None)
                if chosen:
                    st.markdown("**Queries used**")
                    st.code("\n".join(chosen.get("queries_used") or []))

                    missing = chosen.get("missing") or {}
                    st.markdown("**Missing must-have (bullets only)**")
                    st.write(missing.get("must_bullets_only") or [])
                    st.markdown("**Missing nice-to-have (bullets only)**")
                    st.write(missing.get("nice_bullets_only") or [])

                    st.markdown("**Selected IDs**")
                    st.code("\n".join(chosen.get("selected_ids") or []))

            with st.expander("Raw report.json"):
                st.json(report)

        # PDF download
        st.subheader("Download")
        try:
            pdf = requests.get(f"{API_URL}{run['pdf_url']}", timeout=120).content
            st.download_button(
                "Download tailored_resume.pdf",
                data=pdf,
                file_name="tailored_resume.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception:
            st.warning("PDF not ready yet or download failed.")

        try:
            rep_bytes = requests.get(f"{API_URL}{run['report_url']}", timeout=60).content
            st.download_button(
                "Download resume_report.json",
                data=rep_bytes,
                file_name="resume_report.json",
                mime="application/json",
                use_container_width=True,
            )
        except Exception:
            st.warning("report.json not ready yet or download failed.")


st.markdown("---")
st.caption(
    "Tip: run API on :8000 and Streamlit on :8501. Set ART_API_URL if your API runs elsewhere."
)
