from __future__ import annotations

import logging
from typing import Any

from bot.question_router import route_question
from bot.state import QuestionState
from research_pipeline import SequentialResearchPipeline

logger = logging.getLogger(__name__)


FORECAST_LENSES = [
    {
        "name": "base_rate",
        "guidance": (
            "Forecasting lens: Base-rate-first. Emphasize historical frequencies, "
            "outside view, prior rates, analogous episodes, and structural limits. "
            "Be conservative about dramatic short-term moves unless evidence is strong."
        ),
    },
    {
        "name": "trend",
        "guidance": (
            "Forecasting lens: Trend-and-drivers. Emphasize current momentum, "
            "near-term catalysts, operational bottlenecks, and the most important "
            "causal drivers that could move the outcome before resolution."
        ),
    },
    {
        "name": "adversarial",
        "guidance": (
            "Forecasting lens: Adversarial stress test. Actively search for the strongest "
            "reasons the consensus could be wrong, including neglected tail risks, "
            "sudden regime shifts, and overlooked downside or upside scenarios."
        ),
    },
    {
        "name": "skeptical",
        "guidance": (
            "Forecasting lens: Skeptical / anti-hype. Prioritize reasons why the outcome "
            "may fail to happen on time, including inertia, delays, institutional friction, "
            "false signals, and the tendency of markets and events to mean-revert."
        ),
    },
]


def _build_routed_question_prompt(
    question_text: str,
    question_type: str,
    lens_guidance: str = "",
) -> str:
    if question_type == "long_horizon_binary":
        guidance = (
            "Research guidance: This is a long-horizon forecasting question. "
            "Focus on base rates, historical analogies, structural drivers, "
            "expert priors, and long-run trends. Do not overweight recent news."
        )
    elif question_type == "numerical":
        guidance = (
            "Research guidance: This is a numerical forecasting question. "
            "Focus on historical data, trend extrapolation, comparable baselines, "
            "and plausible upper/lower bounds."
        )
    elif question_type == "science_tech":
        guidance = (
            "Research guidance: This is a science/technology forecasting question. "
            "Focus on technical progress, bottlenecks, benchmarks, expert commentary, "
            "and historical pace of advancement."
        )
    elif question_type == "current_events":
        guidance = (
            "Research guidance: This is a current-events question. "
            "Focus on recent developments, official announcements, key actors, "
            "and short-term catalysts."
        )
    else:
        guidance = (
            "Research guidance: Focus on the most decision-relevant evidence including "
            "base rates, developments, key drivers, and uncertainties."
        )

    extra = f"\n{lens_guidance}" if lens_guidance else ""
    return f"{guidance}{extra}\n\nForecasting question:\n{question_text}"


def _coerce_research_text(result: Any) -> str:
    raw_evidence = getattr(result, "raw_evidence", None) or []
    if raw_evidence:
        joined = "\n\n".join(block for block in raw_evidence if block and block.strip())
        if joined.strip():
            return joined.strip()

    summary = getattr(result, "summary", "") or ""
    if summary.strip():
        return summary.strip()

    return "No research evidence was retrieved."


def _pick_lens(state: QuestionState | None, enabled: bool) -> dict[str, str]:
    if not enabled:
        return {"name": "default", "guidance": ""}

    if state is None:
        return FORECAST_LENSES[0]

    idx = len(state.research_runs) % len(FORECAST_LENSES)
    return FORECAST_LENSES[idx]


async def get_research(
    question,
    *,
    state: QuestionState | None = None,
    use_asknews: bool = True,
    use_sequential_research: bool = True,
    use_question_routing: bool = True,
    show_route_debug: bool = True,
    research_model: str = "gpt-4o-mini",
    research_temperature: float = 0.2,
    research_cache_dir: str = ".cache/research",
    max_search_queries: int = 4,
    max_results_per_query: int = 5,
    asknews_cache_ttl_hours: int = 6,
    forecast_diversity_enabled: bool = True,
    **_ignored_kwargs,
) -> str:
    question_text = getattr(question, "question_text", str(question))
    selected_lens = _pick_lens(state, forecast_diversity_enabled)

    if not use_sequential_research:
        message = "Sequential research disabled in config."
        if state is not None:
            route_metadata = {
                "question_type": "research_disabled",
                "use_asknews": False,
                "rationale": message,
                "forecast_lens": selected_lens["name"],
            }
            research_run = state.start_research_run(route_metadata)
            research_run.finalize(
                provider_name="disabled",
                provider_available=False,
                cache_used=False,
                queries=[],
                evidence_count=0,
                warnings=[message],
                errors=[],
                elapsed_seconds=0.0,
                research=message,
                summary=message,
            )
            state.add_warning(message)
        return message

    if use_question_routing:
        route = route_question(question_text)
        routed_use_asknews = use_asknews and route.use_asknews
        routed_question_text = _build_routed_question_prompt(
            question_text=question_text,
            question_type=route.question_type,
            lens_guidance=selected_lens["guidance"],
        )
        route_metadata = {
            "question_type": route.question_type,
            "use_asknews": routed_use_asknews,
            "rationale": route.rationale,
            "forecast_lens": selected_lens["name"],
        }
    else:
        routed_use_asknews = use_asknews
        routed_question_text = _build_routed_question_prompt(
            question_text=question_text,
            question_type="routing_disabled",
            lens_guidance=selected_lens["guidance"],
        )
        route_metadata = {
            "question_type": "routing_disabled",
            "use_asknews": routed_use_asknews,
            "rationale": "Question routing disabled in config.",
            "forecast_lens": selected_lens["name"],
        }

    if show_route_debug:
        logger.info(
            "Research route selected | question_type=%s | use_asknews=%s | lens=%s | rationale=%s",
            route_metadata["question_type"],
            route_metadata["use_asknews"],
            route_metadata["forecast_lens"],
            route_metadata["rationale"],
        )

    research_run = state.start_research_run(route_metadata) if state is not None else None

    try:
        pipeline = SequentialResearchPipeline(
            model=research_model,
            use_asknews=routed_use_asknews,
            max_search_queries=max_search_queries,
            max_results_per_query=max_results_per_query,
            research_cache_dir=research_cache_dir,
            cache_ttl_hours=asknews_cache_ttl_hours,
            research_temperature=research_temperature,
        )

        result = await pipeline.run(routed_question_text)

        research_text = _coerce_research_text(result)
        summary_text = getattr(result, "summary", "") or ""
        warnings = list(getattr(result, "warnings", []) or [])
        errors: list[str] = []

        if state is not None:
            for warning in warnings:
                state.add_warning(warning)

        if research_run is not None:
            research_run.finalize(
                provider_name=getattr(result, "provider_name", "unknown"),
                provider_available=bool(getattr(result, "provider_available", False)),
                cache_used=bool(getattr(result, "cache_used", False)),
                queries=list(getattr(result, "queries", []) or []),
                evidence_count=int(getattr(result, "evidence_count", 0)),
                warnings=warnings,
                errors=errors,
                elapsed_seconds=getattr(result, "elapsed_seconds", None),
                research=research_text,
                summary=summary_text,
            )

        return research_text

    except Exception as e:
        error_message = f"{type(e).__name__}: {e}"
        logger.exception("Sequential research failed")

        fallback_text = f"Sequential research unavailable due to error: {error_message}"

        if state is not None:
            state.add_error(fallback_text)

        if research_run is not None:
            research_run.finalize(
                provider_name="error",
                provider_available=False,
                cache_used=False,
                queries=[],
                evidence_count=0,
                warnings=[],
                errors=[fallback_text],
                elapsed_seconds=None,
                research=fallback_text,
                summary=fallback_text,
            )

        return fallback_text