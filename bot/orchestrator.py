from __future__ import annotations

import asyncio
import re
from typing import Any

from forecasting_tools import MetaculusClient

from bot.aggregator import aggregate_prediction_samples, summarize_prediction
from bot.config import BotConfig
from bot.my_bot import MyBot
from bot.state import QuestionState


FORECAST_PATTERN = re.compile(r"Forecaster\s+\d+\*?:\s*([0-9]+(?:\.[0-9]+)?)%")


def extract_forecasts_from_text(text: str) -> list[float]:
    matches = FORECAST_PATTERN.findall(text or "")
    return [float(m) / 100.0 for m in matches]


def _json_safe(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "model_dump"):
        try:
            return _json_safe(obj.model_dump())
        except Exception:
            pass
    if hasattr(obj, "dict"):
        try:
            return _json_safe(obj.dict())
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        try:
            return _json_safe(vars(obj))
        except Exception:
            pass
    return str(obj)


def _blend_with_community(
    aggregate_probability: float,
    community_prediction: Any,
    anchor_weight: float,
) -> float:
    if not isinstance(aggregate_probability, (int, float)):
        return aggregate_probability

    if not isinstance(community_prediction, (int, float)):
        return float(aggregate_probability)

    w = max(0.0, min(float(anchor_weight), 1.0))
    return ((1.0 - w) * float(aggregate_probability)) + (w * float(community_prediction))


class ForecastOrchestrator:
    def __init__(self, config: BotConfig):
        self.config = config
        self.client = MetaculusClient()

    async def run_question_url(self, url: str) -> dict[str, Any]:
        question = self.client.get_question_by_url(url)

        state = QuestionState(
            question_text=getattr(question, "question_text", ""),
            question_url=getattr(question, "page_url", ""),
            config_snapshot=self.config.__dict__.copy(),
        )

        bot = MyBot(config=self.config, state=state)
        bot.prepare_question_state(question)

        report = await bot.forecast_question(question)
        state = bot.get_question_state(question) or state

        safe_prediction = _json_safe(report.prediction)
        safe_price_estimate = _json_safe(report.price_estimate)
        safe_minutes_taken = _json_safe(report.minutes_taken)
        safe_errors = _json_safe(report.errors)
        safe_explanation = _json_safe(report.explanation)

        explanation_text = safe_explanation if isinstance(safe_explanation, str) else ""

        parsed_samples = extract_forecasts_from_text(explanation_text)
        if not state.predictions:
            for p in parsed_samples:
                state.add_prediction(p, "Parsed from explanation")

        if state.predictions:
            aggregation_summary = aggregate_prediction_samples(
                state.predictions,
                method=getattr(
                    self.config,
                    "binary_aggregation_method",
                    "trimmed_mean_logit",
                ),
                trim_fraction=getattr(self.config, "binary_trim_fraction", 0.2),
            )
        else:
            aggregation_summary = summarize_prediction(safe_prediction)

        aggregate_probability = aggregation_summary.get("aggregate_probability")
        final_prediction = safe_prediction

        if isinstance(aggregate_probability, (int, float)):
            final_prediction = float(aggregate_probability)

        community_prediction = getattr(
            report.question,
            "community_prediction_at_access_time",
            None,
        )

        if isinstance(aggregate_probability, (int, float)):
            aggregation_summary["raw_aggregate_probability"] = float(aggregate_probability)

        anchor_weight = float(getattr(self.config, "community_anchor_weight", 0.15) or 0.0)
        anchored_prediction = final_prediction
        if isinstance(aggregate_probability, (int, float)) and isinstance(
            community_prediction, (int, float)
        ):
            anchored_prediction = _blend_with_community(
                float(aggregate_probability),
                float(community_prediction),
                anchor_weight,
            )
            aggregation_summary["community_prediction"] = float(community_prediction)
            aggregation_summary["community_anchor_weight"] = anchor_weight
            aggregation_summary["anchored_probability"] = round(float(anchored_prediction), 6)
            aggregation_summary["deviation_from_community"] = round(
                abs(float(aggregate_probability) - float(community_prediction)),
                6,
            )
            aggregation_summary["deviation_after_anchoring"] = round(
                abs(float(anchored_prediction) - float(community_prediction)),
                6,
            )

        final_prediction = anchored_prediction

        state.finalize_forecast(
            prediction=final_prediction,
            price_estimate=safe_price_estimate,
            minutes_taken=safe_minutes_taken,
            aggregation_summary=aggregation_summary,
        )

        for error in safe_errors or []:
            state.add_error(str(error))

        return {
            "question_text": getattr(report.question, "question_text", ""),
            "question_url": getattr(report.question, "page_url", ""),
            "prediction": final_prediction,
            "price_estimate": safe_price_estimate,
            "minutes_taken": safe_minutes_taken,
            "errors": safe_errors,
            "explanation": safe_explanation,
            "question_state": state.to_dict(),
            "aggregation_summary": aggregation_summary,
        }


def run_orchestrated_forecast(url: str, config: BotConfig) -> dict[str, Any]:
    orchestrator = ForecastOrchestrator(config)
    return asyncio.run(orchestrator.run_question_url(url))