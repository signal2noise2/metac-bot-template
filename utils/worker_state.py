import json
from pathlib import Path


RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(exist_ok=True)

WORKER_STATUS_FILE = RUNS_DIR / "worker_status.json"
WORKER_RESULT_FILE = RUNS_DIR / "worker_result.json"


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def clear_json(path: Path) -> None:
    if path.exists():
        path.unlink()


def load_worker_status() -> dict:
    return load_json(WORKER_STATUS_FILE)


def save_worker_status(data: dict) -> None:
    save_json(WORKER_STATUS_FILE, data)


def clear_worker_status() -> None:
    clear_json(WORKER_STATUS_FILE)


def load_worker_result() -> dict:
    return load_json(WORKER_RESULT_FILE)


def save_worker_result(data: dict) -> None:
    save_json(WORKER_RESULT_FILE, data)


def clear_worker_result() -> None:
    clear_json(WORKER_RESULT_FILE)