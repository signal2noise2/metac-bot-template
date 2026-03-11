import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import List, Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class BaseSearchProvider:
    async def search(self, query: str, max_results: int = 5) -> str:
        raise NotImplementedError


class AskNewsProvider(BaseSearchProvider):
    """
    AskNews partner / free-plan safe provider.

    AskNews docs say the Metaculus-partner/free plan is limited to:
    - 1 request per 10 seconds
    - concurrency limit 1

    So we enforce:
    - one request at a time with a lock
    - minimum 10 seconds between requests
    """

    _lock = asyncio.Lock()
    _last_request_time = 0.0
    _min_interval_seconds = 10.5  # slight buffer above 10s

    def __init__(self):
        self.client_id = os.getenv("ASKNEWS_CLIENT_ID")
        self.client_secret = os.getenv("ASKNEWS_SECRET")

        if not self.client_id or not self.client_secret:
            raise RuntimeError("ASKNEWS_CLIENT_ID or ASKNEWS_SECRET is missing")

    async def _wait_for_slot(self) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.__class__._last_request_time
            wait_time = self.__class__._min_interval_seconds - elapsed

            if wait_time > 0:
                logger.info("AskNews throttling: sleeping %.2f seconds", wait_time)
                await asyncio.sleep(wait_time)

            self.__class__._last_request_time = time.monotonic()

    async def search(self, query: str, max_results: int = 5) -> str:
        try:
            from asknews_sdk import AsyncAskNewsSDK

            await self._wait_for_slot()

            ask = AsyncAskNewsSDK(
                client_id=self.client_id,
                client_secret=self.client_secret,
                scopes=["news", "stories", "analytics", "chat"],
            )

            response = await asyncio.wait_for(
                ask.news.search_news(
                    query=query,
                    n_articles=max_results,
                    return_type="both",
                    strategy="latest news",
                ),
                timeout=45,
            )

            return response.as_string

        except Exception as e:
            logger.warning("AskNews search failed for query %r: %s", query, e)
            return ""


class NullSearchProvider(BaseSearchProvider):
    async def search(self, query: str, max_results: int = 5) -> str:
        return ""


@dataclass
class ResearchResult:
    summary: str
    raw_evidence: List[str]


class SequentialResearchPipeline:
    def __init__(
        self,
        model: str,
        use_asknews: bool = True,
        max_search_queries: int = 4,
        max_results_per_query: int = 5,
    ):
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY or OPENROUTER_API_KEY must be set")

        base_url = os.getenv("OPENROUTER_BASE_URL")

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = model
        self.max_search_queries = max_search_queries
        self.max_results_per_query = max_results_per_query

        if use_asknews:
            try:
                self.search_provider = AskNewsProvider()
            except Exception as e:
                logger.warning("Falling back to NullSearchProvider: %s", e)
                self.search_provider = NullSearchProvider()
        else:
            self.search_provider = NullSearchProvider()

    async def run(self, question: str) -> ResearchResult:
        queries = await self._build_queries(question)
        queries = queries[: self.max_search_queries]

        evidence = []

        for q in queries:
            search_results = await self.search_provider.search(
                q,
                max_results=self.max_results_per_query,
            )
            combined = f"Query: {q}\n\n{search_results}"
            evidence.append(combined)

        summary = await self._synthesise(question, evidence)
        return ResearchResult(summary=summary, raw_evidence=evidence)

    async def _build_queries(self, question: str) -> List[str]:
        prompt = f"""
You are planning research for a forecasting question.

Question:
{question}

Generate up to {self.max_search_queries} useful search queries.
Focus on:
- base rates
- recent developments
- expert opinion
- key drivers

Return one query per line, with no numbering.
"""

        response = await asyncio.wait_for(
            self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            ),
            timeout=60,
        )

        text = response.choices[0].message.content or ""
        queries = [q.strip("-• \t") for q in text.split("\n") if q.strip()]

        if not queries:
            logger.warning("Falling back to default search plan")
            queries = [
                f"{question} latest developments",
                f"{question} expert analysis",
                f"{question} historical precedent",
                f"{question} base rate probability",
            ]

        return queries[: self.max_search_queries]

    async def _synthesise(self, question: str, evidence: List[str]) -> str:
        joined = "\n\n".join(evidence)

        prompt = f"""
You are preparing a research dossier for a forecasting bot.

Question:
{question}

Evidence gathered:
{joined}

Write a structured research summary covering:
- Base rates
- Recent developments
- Arguments for YES
- Arguments for NO
- Key drivers
- Major uncertainties

Write clearly and concisely.
"""

        response = await asyncio.wait_for(
            self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            ),
            timeout=90,
        )

        return response.choices[0].message.content or ""