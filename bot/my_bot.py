from __future__ import annotations

from typing import Any

from forecasting_tools.forecast_bots.template_bot import TemplateBot

from bot.research import get_research
from bot.state import QuestionState


class MyBot(TemplateBot):
    def __init__(self, config, state: QuestionState | None = None):
        super().__init__(
            research_reports_per_question=config.research_reports_per_question,
            predictions_per_research_report=config.predictions_per_research_report,
            use_research_summary_to_forecast=getattr(
                config, "use_research_summary_to_forecast", True
            ),
            publish_reports_to_metaculus=False,
        )

        self.config = config
        self.state = state

    def prepare_question_state(self, question: Any) -> None:
        if self.state is None:
            return

        self.state.question_text = getattr(question, "question_text", "") or getattr(
            question, "title", ""
        )
        self.state.question_url = getattr(question, "page_url", "") or getattr(
            question, "url", ""
        )

    def get_question_state(self, question: Any) -> QuestionState | None:
        self.prepare_question_state(question)
        return self.state

    async def run_research(self, question: Any) -> str:
        return await get_research(
            question,
            state=self.state,
            use_asknews=self.config.use_asknews,
            use_sequential_research=self.config.use_sequential_research,
            use_question_routing=getattr(self.config, "use_question_routing", True),
            show_route_debug=getattr(self.config, "show_route_debug", True),
            research_model=self.config.research_model,
            research_temperature=self.config.research_temperature,
            research_cache_dir=self.config.research_cache_dir,
            max_search_queries=self.config.max_search_queries,
            max_results_per_query=self.config.max_results_per_query,
            asknews_cache_ttl_hours=self.config.asknews_cache_ttl_hours,
            forecast_diversity_enabled=getattr(
                self.config,
                "forecast_diversity_enabled",
                True,
            ),
        )