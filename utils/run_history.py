import json
from datetime import datetime
from pathlib import Path


RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(exist_ok=True)

RUN_HISTORY_FILE = RUNS_DIR / "run_history.json"


def load_run_history() -> list[dict]:
    if not RUN_HISTORY_FILE.exists():
        return []

    try:
        with open(RUN_HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_run_result(entry: dict) -> None:
    history = load_run_history()
    history.append(entry)

    with open(RUN_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def make_run_entry(
    *,
    question_text: str,
    question_url: str,
    prediction,
    price_estimate,
    minutes_taken,
    errors,
    explanation,
    config: dict,
) -> dict:
    return {
        "timestamp": datetime.now().isoformat(),
        "question_text": question_text,
        "question_url": question_url,
        "prediction": prediction,
        "price_estimate": price_estimate,
        "minutes_taken": minutes_taken,
        "errors": errors,
        "explanation": explanation,
        "config": config,
    }