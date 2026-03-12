import asyncio
import json
import os
import signal
import subprocess
import sys
import tempfile
from collections import defaultdict

import dotenv
import streamlit as st
import streamlit.components.v1 as components

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from bot.config import BotConfig
from forecasting_tools import MetaculusClient
from forecasting_tools.helpers.metaculus_client import ApiFilter
from utils.run_history import load_run_history
from utils.worker_state import (
    clear_worker_result,
    load_worker_result,
    load_worker_status,
    save_worker_status,
)

dotenv.load_dotenv()

st.set_page_config(page_title="Metaculus Bot UI", layout="wide")
st.title("Metaculus Bot Control Panel")

DEFAULT_URL = "https://www.metaculus.com/questions/578/human-extinction-by-2100/"


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def format_prediction_for_display(prediction) -> str:
    if isinstance(prediction, (int, float)):
        return f"{prediction * 100:.2f}%"
    if isinstance(prediction, dict):
        if "aggregate_probability" in prediction:
            return f"{safe_float(prediction['aggregate_probability']) * 100:.2f}%"
        return "Structured prediction"
    return str(prediction)


def render_prediction_block(prediction) -> None:
    if isinstance(prediction, (int, float)):
        st.write(f"**Prediction:** {prediction * 100:.2f}%")
    elif isinstance(prediction, dict):
        st.write("**Prediction:** Structured prediction")
        st.json(prediction)
    else:
        st.write(f"**Prediction:** {prediction}")


@st.cache_data(ttl=300)
def get_live_questions(limit: int = 20):
    client = MetaculusClient()
    api_filter = ApiFilter(
        allowed_statuses=["open"],
        order_by="-published_time",
        is_in_main_feed=True,
    )
    return asyncio.run(
        client.get_questions_matching_filter(
            api_filter=api_filter,
            num_questions=limit,
        )
    )


def pid_is_running(pid: int | None) -> bool:
    if not pid:
        return False

    try:
        pid = int(pid)
    except Exception:
        return False

    try:
        if os.name == "nt":
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}"],
                capture_output=True,
                text=True,
                check=False,
            )
            stdout = (result.stdout or "").lower()
            return str(pid) in stdout and "no tasks are running" not in stdout
        else:
            os.kill(pid, 0)
            return True
    except Exception:
        return False


def worker_is_running() -> bool:
    status = load_worker_status()
    return status.get("state") == "running" and pid_is_running(status.get("pid"))


def _write_temp_config(config: BotConfig) -> str:
    fd, path = tempfile.mkstemp(prefix="metac_bot_config_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(config.__dict__, f)
    except Exception:
        try:
            os.close(fd)
        except Exception:
            pass
        raise
    return path


def start_worker(question_url: str, config: BotConfig) -> None:
    clear_worker_result()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    worker_script = os.path.join(project_root, "run_forecast_worker.py")
    config_path = _write_temp_config(config)

    process = subprocess.Popen(
        [
            sys.executable,
            worker_script,
            "--url",
            question_url,
            "--config-path",
            config_path,
        ],
        cwd=project_root,
    )

    save_worker_status(
        {
            "state": "running",
            "pid": process.pid,
            "question_url": question_url,
            "config": config.__dict__,
            "config_path": config_path,
        }
    )


def stop_worker() -> None:
    status = load_worker_status()
    pid = status.get("pid")

    if not pid:
        return

    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                check=False,
                capture_output=True,
            )
        else:
            os.kill(pid, signal.SIGTERM)
    except Exception:
        pass

    save_worker_status({**status, "state": "stopped"})


def maybe_reconcile_worker_state() -> tuple[dict, dict]:
    status = load_worker_status()
    result = load_worker_result()

    if status.get("state") == "running":
        if not pid_is_running(status.get("pid")):
            if result.get("state") in {"completed", "error", "stopped"}:
                save_worker_status(
                    {
                        **status,
                        "state": result.get("state"),
                        "completed_at": result.get("completed_at"),
                        "error": result.get("error"),
                    }
                )
                status = load_worker_status()
            else:
                save_worker_status(
                    {
                        **status,
                        "state": "unknown",
                        "error": "Worker exited before writing a final result.",
                    }
                )
                status = load_worker_status()

    return status, result


status, result = maybe_reconcile_worker_state()

if status.get("state") == "running":
    refresh_ms = int(status.get("config", {}).get("ui_refresh_interval_seconds", 4)) * 1000
    components.html(
        f"""
        <script>
        setTimeout(function() {{
            window.parent.location.reload();
        }}, {refresh_ms});
        </script>
        """,
        height=0,
    )

with st.sidebar:
    st.header("Settings")

    question_source = st.selectbox(
        "Question source",
        ["Test questions", "Live questions", "Custom URL"],
    )

    if question_source == "Test questions":
        question_options = {
            "Human extinction": "https://www.metaculus.com/questions/578/human-extinction-by-2100/",
            "AI labor strikes": "https://www.metaculus.com/questions/38880/",
            "Gas prices all-time high": "https://www.metaculus.com/questions/42548",
        }
        question_choice = st.selectbox(
            "Select a test question",
            list(question_options.keys()),
        )
        question_url = question_options[question_choice]

    elif question_source == "Live questions":
        try:
            live_questions = get_live_questions(limit=20)
            question_map = {
                q.question_text: q.page_url
                for q in live_questions
                if getattr(q, "page_url", None)
            }

            if question_map:
                question_choice = st.selectbox(
                    "Select a live question",
                    list(question_map.keys()),
                )
                question_url = question_map[question_choice]
            else:
                st.warning("No live questions could be loaded. Falling back to default URL.")
                question_url = DEFAULT_URL

        except Exception as e:
            st.error(f"Could not load live questions: {e}")
            question_url = DEFAULT_URL
    else:
        question_url = st.text_input("Metaculus question URL", value=DEFAULT_URL)

    st.subheader("Forecasting")

    research_reports = st.number_input(
        "Research reports per question",
        min_value=1,
        max_value=5,
        value=2,
    )

    predictions_per_report = st.number_input(
        "Predictions per research report",
        min_value=1,
        max_value=5,
        value=3,
    )

    max_concurrent = st.number_input(
        "Max concurrent questions",
        min_value=1,
        max_value=10,
        value=1,
    )

    required_successful_predictions = st.slider(
        "Required successful prediction fraction",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
    )

    st.subheader("Research")

    use_asknews = st.checkbox("Use AskNews", value=True)
    use_sequential_research = st.checkbox("Use sequential research", value=True)
    use_question_routing = st.checkbox("Use question routing", value=True)
    show_route_debug = st.checkbox("Show route debug", value=True)
    enable_research_summary = st.checkbox("Enable research summary", value=True)
    use_research_summary_to_forecast = st.checkbox(
        "Use research summary to forecast",
        value=True,
    )

    forecast_diversity_enabled = st.checkbox(
        "Enable forecast diversity lenses",
        value=True,
    )

    research_cache_dir = st.text_input(
        "Research cache directory",
        value=".cache/research",
    )

    asknews_cache_ttl_hours = st.number_input(
        "AskNews cache TTL (hours)",
        min_value=1,
        max_value=168,
        value=6,
    )

    max_search_queries = st.number_input(
        "Max search queries",
        min_value=1,
        max_value=10,
        value=4,
    )

    max_results_per_query = st.number_input(
        "Max results per query",
        min_value=1,
        max_value=20,
        value=5,
    )

    research_temperature = st.slider(
        "Research temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05,
    )

    st.subheader("Aggregation")

    binary_aggregation_method = st.selectbox(
        "Binary aggregation method",
        options=["trimmed_mean_logit", "median", "median_logit", "mean", "trimmed_mean"],
        index=0,
    )

    binary_trim_fraction = st.slider(
        "Binary trim fraction",
        min_value=0.0,
        max_value=0.4,
        value=0.2,
        step=0.05,
    )

    community_anchor_weight = st.slider(
        "Community anchor weight",
        min_value=0.0,
        max_value=0.5,
        value=0.15,
        step=0.01,
    )

    anti_rounding = st.checkbox("Anti-rounding", value=True)

    st.subheader("Models")

    default_model = st.selectbox(
        "Default model",
        options=["gpt-4o-mini", "gpt-4o"],
        index=0,
    )

    summarizer_model = st.selectbox(
        "Summarizer model",
        options=["gpt-4o-mini", "gpt-4o"],
        index=0,
    )

    parser_model = st.selectbox(
        "Parser model",
        options=["gpt-4o-mini", "gpt-4o"],
        index=0,
    )

    research_model = st.selectbox(
        "Research model",
        options=["gpt-4o-mini", "gpt-4o"],
        index=0,
    )

    ui_refresh_interval_seconds = st.number_input(
        "UI refresh interval while running (seconds)",
        min_value=2,
        max_value=30,
        value=4,
    )

    config = BotConfig(
        bot_name="RobBot",
        research_reports_per_question=int(research_reports),
        predictions_per_research_report=int(predictions_per_report),
        max_concurrent_questions=int(max_concurrent),
        required_successful_predictions=float(required_successful_predictions),
        default_model=default_model,
        summarizer_model=summarizer_model,
        parser_model=parser_model,
        use_asknews=use_asknews,
        use_sequential_research=use_sequential_research,
        use_question_routing=use_question_routing,
        show_route_debug=show_route_debug,
        enable_research_summary=enable_research_summary,
        use_research_summary_to_forecast=use_research_summary_to_forecast,
        research_model=research_model,
        research_temperature=float(research_temperature),
        research_cache_dir=research_cache_dir,
        max_search_queries=int(max_search_queries),
        max_results_per_query=int(max_results_per_query),
        asknews_cache_ttl_hours=int(asknews_cache_ttl_hours),
        binary_aggregation_method=binary_aggregation_method,
        binary_trim_fraction=float(binary_trim_fraction),
        anti_rounding=anti_rounding,
        forecast_diversity_enabled=forecast_diversity_enabled,
        community_anchor_weight=float(community_anchor_weight),
        ui_refresh_interval_seconds=int(ui_refresh_interval_seconds),
    )

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Run Forecast", width="stretch"):
        if worker_is_running():
            st.warning("A forecast is already running.")
        else:
            start_worker(question_url, config)
            st.rerun()

with col2:
    if st.button("Refresh Status", width="stretch"):
        st.rerun()

with col3:
    if st.button("Stop Forecast", width="stretch"):
        if worker_is_running():
            stop_worker()
            st.rerun()
        else:
            st.info("No running forecast to stop.")

st.divider()
st.subheader("Worker Status")

status, result = maybe_reconcile_worker_state()

if not status:
    st.write("No worker activity yet.")
else:
    st.write(f"**State:** {status.get('state', 'unknown')}")
    st.write(f"**PID:** {status.get('pid', 'n/a')}")
    st.write(f"**Question URL:** {status.get('question_url', 'n/a')}")

    if status.get("state") == "running":
        st.info(
            "A forecast is currently running. The UI is checking whether the worker has ended and will refresh automatically."
        )
    elif status.get("state") == "stopped":
        st.warning("The forecast was stopped.")
    elif status.get("state") == "error":
        st.error(f"Worker error: {status.get('error', 'Unknown error')}")

if result.get("state") == "completed":
    st.divider()
    st.subheader("Forecast Summary")
    st.write(f"**Question:** {result['question_text']}")
    st.write(f"**URL:** {result['question_url']}")
    render_prediction_block(result.get("prediction"))
    st.write(f"**Cost estimate:** ${safe_float(result.get('price_estimate')):.2f}")
    st.write(f"**Minutes taken:** {safe_float(result.get('minutes_taken')):.2f}")
    st.write(f"**Errors:** {result.get('errors', [])}")

    aggregation_summary = result.get("aggregation_summary") or {}
    if aggregation_summary:
        raw_agg = aggregation_summary.get("raw_aggregate_probability")
        anchored = aggregation_summary.get("anchored_probability")
        community = aggregation_summary.get("community_prediction")
        anchor_weight = aggregation_summary.get("community_anchor_weight")

        if isinstance(raw_agg, (int, float)):
            st.write(f"**Raw ensemble probability:** {raw_agg * 100:.2f}%")
        if isinstance(community, (int, float)):
            st.write(f"**Community prediction:** {community * 100:.2f}%")
        if isinstance(anchor_weight, (int, float)):
            st.write(f"**Community anchor weight:** {anchor_weight:.2f}")
        if isinstance(anchored, (int, float)):
            st.write(f"**Anchored final probability:** {anchored * 100:.2f}%")

    with st.expander("Explanation", expanded=True):
        st.text(result.get("explanation") or "(No explanation returned)")

    if aggregation_summary:
        with st.expander("Aggregation diagnostics", expanded=True):
            st.json(aggregation_summary)

    question_state = result.get("question_state") or {}
    if question_state:
        with st.expander("Question state", expanded=False):
            st.json(question_state)

elif result.get("state") == "error":
    st.error(f"Error: {result.get('error', 'Unknown error')}")
    with st.expander("Traceback", expanded=False):
        st.code(result.get("traceback", ""), language="text")

st.divider()
st.subheader("Run History")

history = load_run_history()
if not history:
    st.write("No saved runs yet.")
else:
    grouped = defaultdict(list)
    for entry in reversed(history):
        grouped[entry.get("question_url", "unknown")].append(entry)

    for question_url, entries in grouped.items():
        latest = entries[0]
        header = latest.get("question_text") or question_url
        with st.expander(header, expanded=False):
            for idx, entry in enumerate(entries[:10], start=1):
                st.markdown(f"**Run {idx}**")
                st.write(f"Timestamp: {entry.get('timestamp', '')}")
                st.write(f"Prediction: {format_prediction_for_display(entry.get('prediction'))}")
                st.write(f"Cost: ${safe_float(entry.get('price_estimate')):.2f}")
                st.write(f"Minutes: {safe_float(entry.get('minutes_taken')):.2f}")
                if entry.get("aggregation_summary"):
                    st.json(entry["aggregation_summary"])
                st.divider()