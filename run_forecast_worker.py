import argparse
import asyncio
import json
import os
import traceback
from datetime import datetime

import dotenv

from bot.config import BotConfig
from bot.my_bot import MyBot
from forecasting_tools import MetaculusClient
from utils.run_history import make_run_entry, save_run_result
from utils.worker_state import (
    clear_worker_result,
    save_worker_result,
    save_worker_status,
)

dotenv.load_dotenv()


async def run_forecast(url: str, config: BotConfig):
    bot = MyBot(config=config)
    client = MetaculusClient()
    question = client.get_question_by_url(url)
    reports = await bot.forecast_questions([question])
    return reports[0]


def _load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--config-path", required=True)
    args = parser.parse_args()

    config_dict = _load_config(args.config_path)
    config = BotConfig(**config_dict)

    clear_worker_result()
    save_worker_status(
        {
            "state": "running",
            "pid": os.getpid(),
            "started_at": datetime.now().isoformat(),
            "question_url": args.url,
            "config": config_dict,
        }
    )

    try:
        report = asyncio.run(run_forecast(args.url, config))

        result = {
            "state": "completed",
            "completed_at": datetime.now().isoformat(),
            "question_text": report.question.question_text,
            "question_url": report.question.page_url,
            "prediction": report.prediction,
            "price_estimate": report.price_estimate,
            "minutes_taken": report.minutes_taken,
            "errors": report.errors,
            "explanation": report.explanation,
            "config": config_dict,
        }
        save_worker_result(result)

        entry = make_run_entry(
            question_text=report.question.question_text,
            question_url=report.question.page_url,
            prediction=report.prediction,
            price_estimate=report.price_estimate,
            minutes_taken=report.minutes_taken,
            errors=report.errors,
            explanation=report.explanation,
            config=config_dict,
        )
        save_run_result(entry)

        save_worker_status(
            {
                "state": "completed",
                "pid": os.getpid(),
                "completed_at": datetime.now().isoformat(),
                "question_url": args.url,
                "config": config_dict,
            }
        )

    except Exception as e:
        error_text = "".join(traceback.format_exception(type(e), e, e.__traceback__))

        save_worker_result(
            {
                "state": "error",
                "completed_at": datetime.now().isoformat(),
                "question_url": args.url,
                "error": str(e),
                "traceback": error_text,
                "config": config_dict,
            }
        )

        save_worker_status(
            {
                "state": "error",
                "pid": os.getpid(),
                "completed_at": datetime.now().isoformat(),
                "question_url": args.url,
                "config": config_dict,
                "error": str(e),
            }
        )
        raise

    finally:
        try:
            if os.path.exists(args.config_path):
                os.remove(args.config_path)
        except Exception:
            pass


if __name__ == "__main__":
    main()