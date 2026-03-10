from forecasting_tools import AskNewsSearcher


async def get_research(question, prompt: str, use_asknews: bool = True) -> str:
    if not use_asknews:
        return "Research disabled in settings."

    try:
        return await AskNewsSearcher().get_formatted_news_async(prompt)
    except Exception as e:
        return f"Research unavailable due to error: {e}"