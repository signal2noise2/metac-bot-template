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

default_url = "https://www.metaculus.com/questions/578/human-extinction-by-2100/"


def extract_question_id(url: str) -> str:
    try:
        parts = url.rstrip("/").split("/")
        idx = parts.index("questions")
        return parts[idx + 1]
    except Exception:
        return "?"


def shorten(text: str, length: int = 48) -> str:
    if not text:
        return "(No title)"
    if len(text) <= length:
        return text
    return text[:length].rstrip() + "..."


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


def worker_is_running() -> bool:
    status = load_worker_status()
    return status.get("state") == "running"


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

    save_worker_status(
        {
            **status,
            "state": "stopped",
        }
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
                question_url = default_url

        except Exception as e:
            st.error(f"Could not load live questions: {e}")
            question_url = default_url

    else:
        question_url = st.text_input("Metaculus question URL", value=default_url)

    research_reports = st.number_input(
        "Research reports per question",
        min_value=1,
        max_value=5,
        value=1,
    )

    predictions_per_report = st.number_input(
        "Predictions per research report",
        min_value=1,
        max_value=5,
        value=1,
    )

    max_concurrent = st.number_input(
        "Max concurrent questions",
        min_value=1,
        max_value=10,
        value=1,
    )

    use_asknews = st.checkbox("Use AskNews", value=True)
    use_sequential_research = st.checkbox("Use sequential research", value=True)

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

    config = BotConfig(
        bot_name="RobBot",
        research_reports_per_question=research_reports,
        predictions_per_research_report=predictions_per_report,
        max_concurrent_questions=max_concurrent,
        default_model=default_model,
        summarizer_model=summarizer_model,
        parser_model=parser_model,
        use_asknews=use_asknews,
        use_sequential_research=use_sequential_research,
        research_model=research_model,
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

status = load_worker_status()
result = load_worker_result()

if not status:
    st.write("No worker activity yet.")
else:
    st.write(f"**State:** {status.get('state', 'unknown')}")
    st.write(f"**PID:** {status.get('pid', 'n/a')}")
    st.write(f"**Question URL:** {status.get('question_url', 'n/a')}")

    if status.get("state") == "running":
        st.info("A forecast is currently running. Use Refresh Status to check progress or Stop Forecast to cancel.")
    elif status.get("state") == "stopped":
        st.warning("The forecast was stopped.")
    elif status.get("state") == "error":
        st.error(f"Worker error: {status.get('error', 'Unknown error')}")

if result.get("state") == "completed":
    st.divider()
    st.subheader("Forecast Summary")
    st.write(f"**Question:** {result['question_text']}")
    st.write(f"**URL:** {result['question_url']}")
    st.write(f"**Prediction:** {result['prediction'] * 100:.2f}%")
    st.write(f"**Cost estimate:** ${result['price_estimate']:.2f}")
    st.write(f"**Minutes taken:** {result['minutes_taken']:.2f}")
    st.write(f"**Errors:** {result['errors']}")

    with st.expander("Explanation", expanded=True):
        st.text(result.get("explanation") or "(No explanation returned)")

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
    for item in reversed(history[-20:]):
        qid = extract_question_id(item["question_url"])
        subject = shorten(item.get("question_text", ""))
        title = (
            f"{qid} | {subject} | "
            f"P={item['prediction'] * 100:.2f}% | "
            f"${item['price_estimate']:.2f}"
        )

        with st.expander(title):
            st.write(f"**Question:** {item['question_text']}")
            st.write(f"**URL:** {item['question_url']}")
            st.write(f"**Prediction:** {item['prediction'] * 100:.2f}%")
            st.write(f"**Cost:** ${item['price_estimate']:.2f}")
            st.write(f"**Minutes:** {item['minutes_taken']:.2f}")
            st.write(f"**Errors:** {item['errors']}")
            st.write("**Config:**")
            st.json(item["config"])

st.divider()
st.subheader("Run Comparison")

if not history:
    st.write("No runs available for comparison yet.")
else:
    grouped = defaultdict(list)
    for item in history:
        grouped[item["question_url"]].append(item)

    comparison_options = []
    label_to_url = {}

    for url, items in grouped.items():
        latest_item = items[-1]
        qid = extract_question_id(url)
        subject = shorten(latest_item.get("question_text", ""), 60)
        label = f"{qid} | {subject} ({len(items)} runs)"
        comparison_options.append(label)
        label_to_url[label] = url

    selected_label = st.selectbox(
        "Choose a question to compare runs",
        comparison_options,
    )
    selected_url = label_to_url[selected_label]
    selected_runs = grouped[selected_url]

    comparison_rows = []
    for item in reversed(selected_runs):
        cfg = item.get("config", {})
        comparison_rows.append(
            {
                "Timestamp": str(item.get("timestamp", "")),
                "Prediction %": float(round(item.get("prediction", 0) * 100, 2)),
                "Cost $": float(round(item.get("price_estimate", 0), 2)),
                "Minutes": float(round(item.get("minutes_taken", 0), 2)),
                "Default model": str(cfg.get("default_model", "")),
                "Summarizer": str(cfg.get("summarizer_model", "")),
                "Parser": str(cfg.get("parser_model", "")),
                "Research model": str(cfg.get("research_model", "")),
                "AskNews": str(cfg.get("use_asknews", "")),
                "Sequential": str(cfg.get("use_sequential_research", "")),
                "Research reports": int(cfg.get("research_reports_per_question", 0)),
                "Predictions/report": int(cfg.get("predictions_per_research_report", 0)),
            }
        )

    st.dataframe(comparison_rows, width="stretch")