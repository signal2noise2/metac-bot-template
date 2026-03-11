import logging

from research_pipeline import SequentialResearchPipeline

logger = logging.getLogger(__name__)


async def get_research(
    question,
    *,
    use_asknews: bool = True,
    use_sequential_research: bool = True,
    research_model: str = "gpt-4o-mini",
    research_temperature: float = 0.2,
    research_cache_dir: str = ".cache/research",
    max_search_queries: int = 4,
    max_results_per_query: int = 5,
) -> str:
    try:
        question_text = getattr(question, "question_text", str(question))

        if not use_sequential_research:
            return "Sequential research disabled in config."

        pipeline = SequentialResearchPipeline(
            model=research_model,
            use_asknews=use_asknews,
            max_search_queries=max_search_queries,
            max_results_per_query=max_results_per_query,
        )

        result = await pipeline.run(question_text)
        return result.summary

    except Exception as e:
        logger.exception("Sequential research failed")
        return f"Sequential research unavailable due to error: {str(e)}"