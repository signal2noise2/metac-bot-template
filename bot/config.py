from dataclasses import dataclass


@dataclass
class BotConfig:
    bot_name: str = "RobBot"

    # Forecasting behaviour
    research_reports_per_question: int = 1
    predictions_per_research_report: int = 1
    max_concurrent_questions: int = 1

    # Models used by TemplateBot / forecasting-tools
    default_model: str = "gpt-4o-mini"
    summarizer_model: str = "gpt-4o-mini"
    parser_model: str = "gpt-4o-mini"

    # Research settings
    use_asknews: bool = True
    use_sequential_research: bool = True
    research_model: str = "gpt-4o-mini"
    research_temperature: float = 0.2
    research_cache_dir: str = ".cache/research"
    max_search_queries: int = 4
    max_results_per_query: int = 5