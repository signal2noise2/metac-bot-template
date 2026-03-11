import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import List

from openai import AsyncOpenAI

from utils.research_cache import ResearchCache

logger = logging.getLogger(__name__)


class BaseSearchProvider:
    name = "base"

    async def search(self, query: str, max_results: int = 5) -> str:
        raise NotImplementedError


class AskNewsProvider(BaseSearchProvider):
    """
    AskNews partner-safe provider.

    Metaculus AskNews plan limits:
    - 1 request every 10 seconds
    - concurrency limit = 1
    """

    name = "asknews"

    _lock = asyncio.Lock()
    _last_request_time = 0.0
    _min_interval_seconds = 10.5

    def __init__(self):
        self.client_id = os.getenv("ASKNEWS_CLIENT_ID")
        self.client_secret = os.getenv("ASKNEWS_SECRET")

        if not self.client_id or not self.client_secret:
            raise RuntimeError("ASKNEWS_CLIENT_ID or ASKNEWS_SECRET missing")

    async def _wait_for_slot(self):
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.__class__._last_request_time
            wait_time = self.__class__._min_interval_seconds - elapsed

            if wait_time > 0:
                logger.info("AskNews throttling %.2fs", wait_time)
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
            logger.warning("AskNews failed: %s", e)
            return ""


class NullSearchProvider(BaseSearchProvider):
    name = "null"

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
        research_cache_dir: str = ".cache/research",
    ):
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")

        if not api_key:
            raise RuntimeError("OPENAI_API_KEY or OPENROUTER_API_KEY required")

        base_url = os.getenv("OPENROUTER_BASE_URL")

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        self.model = model
        self.max_search_queries = max_search_queries
        self.max_results_per_query = max_results_per_query

        self.cache = ResearchCache(research_cache_dir)

        if use_asknews:
            try:
                self.search_provider = AskNewsProvider()
            except Exception as e:
                logger.warning("AskNews unavailable: %s", e)
                self.search_provider = NullSearchProvider()
        else:
            self.search_provider = NullSearchProvider()

    async def run(self, question: str) -> ResearchResult:
        """
        Research pipeline:

        1. Generate search queries
        2. Fuse them into one AskNews query
        3. Cache lookup
        4. Fetch evidence if needed
        5. Summarise
        """

        queries = await self._build_queries(question)

        # fuse queries into one AskNews-safe query
        fused_query = " | ".join(queries[: self.max_search_queries])

        evidence = []

        cached = self.cache.get(
            provider=self.search_provider.name,
            query=fused_query,
            max_results=self.max_results_per_query,
            ttl_seconds=6 * 3600,
        )

        if cached:
            logger.info("Research cache hit")
            evidence.append(cached)

        else:
            logger.info("Research cache miss")

            search_results = await self.search_provider.search(
                fused_query,
                max_results=self.max_results_per_query,
            )

            combined = f"Query: {fused_query}\n\n{search_results}"

            evidence.append(combined)

            self.cache.set(
                provider=self.search_provider.name,
                query=fused_query,
                max_results=self.max_results_per_query,
                result=combined,
            )

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

Return one query per line.
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