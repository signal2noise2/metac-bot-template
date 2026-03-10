from forecasting_tools.forecast_bots.template_bot import TemplateBot

from bot.config import BotConfig
from bot.research import get_research


class MyBot(TemplateBot):
    def __init__(self, config: BotConfig | None = None):
        self.config = config or BotConfig()

        super().__init__(
            research_reports_per_question=self.config.research_reports_per_question,
            predictions_per_research_report=self.config.predictions_per_research_report,
        )

        self.llms = {
            "default": self.config.default_model,
            "summarizer": self.config.summarizer_model,
            "parser": self.config.parser_model,
        }

    @property
    def bot_name(self) -> str:
        return self.config.bot_name

    @property
    def _max_concurrent_questions(self) -> int:
        return self.config.max_concurrent_questions

    async def run_research(self, question) -> str:
        prompt = f"""
You are an assistant to a superforecaster.
Provide a concise but detailed roundup of the most relevant recent information.

Question:
{question.question_text}
"""
        return await get_research(
            question=question,
            prompt=prompt,
            use_asknews=self.config.use_asknews,
        )