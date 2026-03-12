from dataclasses import dataclass


@dataclass
class BotConfig:
    bot_name: str = "RobBot"

    # Forecasting behaviour
    research_reports_per_question: int = 2
    predictions_per_research_report: int = 3
    max_concurrent_questions: int = 1
    required_successful_predictions: float = 0.5

    # Models used by TemplateBot / forecasting-tools
    default_model: str = "gpt-4o-mini"
    summarizer_model: str = "gpt-4o-mini"
    parser_model: str = "gpt-4o-mini"

    # Research settings
    use_asknews: bool = True
    use_sequential_research: bool = True
    use_question_routing: bool = True
    show_route_debug: bool = True
    enable_research_summary: bool = True
    use_research_summary_to_forecast: bool = True

    research_model: str = "gpt-4o-mini"
    research_temperature: float = 0.2
    research_cache_dir: str = ".cache/research"
    max_search_queries: int = 4
    max_results_per_query: int = 5

    # Cache / provider behaviour
    asknews_cache_ttl_hours: int = 6

    # Aggregation / anti-coarsening
    binary_aggregation_method: str = "trimmed_mean_logit"
    binary_trim_fraction: float = 0.2
    anti_rounding: bool = True

    # Forecast diversity
    forecast_diversity_enabled: bool = True

    # Community anchoring
    community_anchor_weight: float = 0.15

    # UI refresh / monitoring
    ui_refresh_interval_seconds: int = 4