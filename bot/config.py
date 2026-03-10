from dataclasses import dataclass


@dataclass
class BotConfig:
    bot_name: str = "RobBot"
    research_reports_per_question: int = 1
    predictions_per_research_report: int = 1
    max_concurrent_questions: int = 1

    default_model: str = "gpt-4o-mini"
    summarizer_model: str = "gpt-4o-mini"
    parser_model: str = "gpt-4o-mini"

    use_asknews: bool = True