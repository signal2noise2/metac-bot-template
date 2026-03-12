from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ResearchRunRecord:
    route_metadata: dict[str, Any]

    started_at: str = field(default_factory=_utc_now_iso)
    completed_at: str | None = None

    provider_name: str = ""
    provider_available: bool = False
    cache_used: bool = False

    queries: list[str] = field(default_factory=list)
    evidence_count: int = 0

    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    research: str = ""
    summary: str = ""

    elapsed_seconds: float | None = None
    completed: bool = False

    def finalize(
        self,
        *,
        provider_name: str,
        provider_available: bool,
        cache_used: bool,
        queries: list[str],
        evidence_count: int,
        warnings: list[str],
        errors: list[str],
        research: str,
        summary: str,
        elapsed_seconds: float | None,
    ) -> None:
        self.provider_name = provider_name
        self.provider_available = provider_available
        self.cache_used = cache_used
        self.queries = queries
        self.evidence_count = evidence_count
        self.warnings = warnings
        self.errors = errors
        self.research = research
        self.summary = summary
        self.elapsed_seconds = elapsed_seconds
        self.completed = True
        self.completed_at = _utc_now_iso()

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "route": self.route_metadata,
            "provider_name": self.provider_name,
            "provider_available": self.provider_available,
            "cache_used": self.cache_used,
            "queries": self.queries,
            "evidence_count": self.evidence_count,
            "warnings": self.warnings,
            "errors": self.errors,
            "elapsed_seconds": self.elapsed_seconds,
            "research_preview": self.research[:800] if self.research else "",
            "summary_preview": self.summary[:800] if self.summary else "",
            "completed": self.completed,
        }


@dataclass
class PredictionSample:
    prediction_value: float
    reasoning_preview: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction_value": self.prediction_value,
            "reasoning_preview": self.reasoning_preview,
        }


class QuestionState:
    def __init__(
        self,
        question_text: str = "",
        question_url: str = "",
        config_snapshot: dict[str, Any] | None = None,
    ):
        self.question_text = question_text
        self.question_url = question_url
        self.config_snapshot = config_snapshot or {}

        self.started_at = _utc_now_iso()
        self.completed_at: str | None = None

        self.route_decision: dict[str, Any] = {}
        self.research_runs: list[ResearchRunRecord] = []
        self.predictions: list[PredictionSample] = []

        self.warnings: list[str] = []
        self.errors: list[str] = []

        self.final_prediction: Any = None
        self.final_price_estimate: Any = None
        self.final_minutes_taken: Any = None
        self.aggregation_summary: dict[str, Any] = {}

    def start_research_run(self, route_metadata: dict[str, Any]) -> ResearchRunRecord:
        if route_metadata and not self.route_decision:
            self.route_decision = dict(route_metadata)
        run = ResearchRunRecord(route_metadata=route_metadata)
        self.research_runs.append(run)
        return run

    def add_prediction(self, probability: float, reasoning: str = "") -> None:
        try:
            prob = float(probability)
        except Exception:
            return
        self.predictions.append(
            PredictionSample(
                prediction_value=prob,
                reasoning_preview=(reasoning or "")[:1200],
            )
        )

    def add_warning(self, warning: str) -> None:
        if warning:
            self.warnings.append(str(warning))

    def add_error(self, error: str) -> None:
        if error:
            self.errors.append(str(error))

    def finalize_forecast(
        self,
        *,
        prediction: Any,
        price_estimate: Any,
        minutes_taken: Any,
        aggregation_summary: dict[str, Any] | None = None,
    ) -> None:
        self.final_prediction = prediction
        self.final_price_estimate = price_estimate
        self.final_minutes_taken = minutes_taken
        self.aggregation_summary = aggregation_summary or {}
        self.completed_at = _utc_now_iso()

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_text": self.question_text,
            "question_url": self.question_url,
            "config_snapshot": self.config_snapshot,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "route_decision": self.route_decision,
            "research_runs": [run.to_dict() for run in self.research_runs],
            "warnings": self.warnings,
            "errors": self.errors,
            "final_prediction": self.final_prediction,
            "final_price_estimate": self.final_price_estimate,
            "final_minutes_taken": self.final_minutes_taken,
            "aggregation_summary": self.aggregation_summary,
            "research_cache_hits": sum(1 for run in self.research_runs if run.cache_used),
            "research_runs_completed": sum(1 for run in self.research_runs if run.completed),
            "total_prediction_samples": len(self.predictions),
            "predictions": [prediction.to_dict() for prediction in self.predictions],
        }