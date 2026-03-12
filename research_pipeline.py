from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from typing import List

from openai import AsyncOpenAI

from utils.research_cache import ResearchCache

logger = logging.getLogger(__name__)


class BaseSearchProvider:
    name = "base"
    available = True

    async def search(self, query: str, max_results: int = 5) -> str:
        raise NotImplementedError


class AskNewsProvider(BaseSearchProvider):
    """
    AskNews provider with safe throttling.

    Metaculus / AskNews practical constraints:
    - 1 request every ~10 seconds
    - no concurrent requests
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

    async def _wait_for_slot(self) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.__class__._last_request_time
            wait_time = self.__class__._min_interval_seconds - elapsed
            if wait_time > 0:
                logger.info("AskNews throttling | wait_time=%.2fs", wait_time)
                await asyncio.sleep(wait_time)
            self.__class__._last_request_time = time.monotonic()

    async def search(self, query: str, max_results: int = 5) -> str:
        started = time.monotonic()
        logger.info(
            "AskNews search starting | query=%r | max_results=%s",
            query,
            max_results,
        )
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

            result = response.as_string or ""
            logger.info(
                "AskNews search completed | chars=%s | elapsed=%.2fs",
                len(result),
                time.monotonic() - started,
            )
            return result
        except Exception as e:
            logger.warning("AskNews search failed | query=%r | error=%s", query, e)
            return ""


class NullSearchProvider(BaseSearchProvider):
    name = "null"
    available = False

    async def search(self, query: str, max_results: int = 5) -> str:
        logger.info("NullSearchProvider used")
        return ""


@dataclass
class ResearchResult:
    summary: str
    raw_evidence: List[str]
    cache_used: bool
    provider_name: str
    provider_available: bool
    queries: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    elapsed_seconds: float | None = None

    @property
    def evidence_count(self) -> int:
        return len([item for item in self.raw_evidence if item.strip()])


class SequentialResearchPipeline:
    def __init__(
        self,
        model: str,
        use_asknews: bool = True,
        max_search_queries: int = 4,
        max_results_per_query: int = 5,
        research_cache_dir: str = ".cache/research",
        cache_ttl_hours: int = 6,
        research_temperature: float = 0.2,
    ):
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY or OPENROUTER_API_KEY required")

        base_url = os.getenv("OPENROUTER_BASE_URL")
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_search_queries = max_search_queries
        self.max_results_per_query = max_results_per_query
        self.cache_ttl_seconds = max(int(cache_ttl_hours), 1) * 3600
        self.research_temperature = research_temperature
        self.cache = ResearchCache(research_cache_dir)
        self.initialization_warnings: list[str] = []

        if use_asknews:
            try:
                self.search_provider = AskNewsProvider()
            except Exception as e:
                warning = f"AskNews unavailable: {e}"
                logger.warning(warning)
                self.initialization_warnings.append(warning)
                self.search_provider = NullSearchProvider()
        else:
            self.search_provider = NullSearchProvider()
            self.initialization_warnings.append(
                "AskNews disabled by config; using null provider."
            )

        logger.info(
            "Research pipeline initialised | provider=%s | cache_dir=%s | ttl_hours=%s",
            self.search_provider.name,
            research_cache_dir,
            cache_ttl_hours,
        )

    def _make_question_cache_key(self, question: str) -> str:
        normalized = " ".join((question or "").strip().lower().split())
        raw_key = "|".join(
            [
                self.search_provider.name,
                normalized,
                str(self.max_search_queries),
                str(self.max_results_per_query),
            ]
        )
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    async def run(self, question: str) -> ResearchResult:
        start = time.monotonic()
        logger.info("Research started | question=%r", question)

        cache_used = False
        evidence: list[str] = []
        warnings = list(self.initialization_warnings)
        queries: list[str] = []
        question_cache_key = self._make_question_cache_key(question)

        cached = self.cache.get(
            provider=f"{self.search_provider.name}_question",
            query=question_cache_key,
            max_results=self.max_results_per_query,
            ttl_seconds=self.cache_ttl_seconds,
        )

        if cached:
            cache_used = True
            evidence.append(cached)
            logger.info(
                "Research cache HIT | provider=%s | question_cache_key=%s",
                self.search_provider.name,
                question_cache_key,
            )
        else:
            logger.info(
                "Research cache MISS | provider=%s | question_cache_key=%s",
                self.search_provider.name,
                question_cache_key,
            )
            queries = await self._build_queries(question)
            logger.info(
                "Generated research queries | count=%s | queries=%s",
                len(queries),
                queries,
            )

            if self.search_provider.available:
                for query in queries:
                    search_results = await self.search_provider.search(
                        query,
                        max_results=self.max_results_per_query,
                    )
                    if search_results.strip():
                        evidence.append(f"Query: {query}\n\n{search_results}")
                    else:
                        warnings.append(f"No results returned for query: {query}")
            else:
                warnings.append(
                    "External search provider unavailable; synthesis will rely on model priors only."
                )

            combined = "\n\n".join(evidence)
            self.cache.set(
                provider=f"{self.search_provider.name}_question",
                query=question_cache_key,
                max_results=self.max_results_per_query,
                result=combined,
            )
            logger.info(
                "Research cache WRITE | provider=%s | question_cache_key=%s",
                self.search_provider.name,
                question_cache_key,
            )

        if not evidence:
            warnings.append("No external evidence retrieved for this question.")

        summary = await self._synthesise(question, evidence, warnings)
        elapsed_seconds = time.monotonic() - start

        logger.info(
            "Research completed | elapsed=%.2fs | cache_used=%s | evidence_count=%s",
            elapsed_seconds,
            cache_used,
            len(evidence),
        )

        return ResearchResult(
            summary=summary,
            raw_evidence=evidence,
            cache_used=cache_used,
            provider_name=self.search_provider.name,
            provider_available=self.search_provider.available,
            queries=queries,
            warnings=warnings,
            elapsed_seconds=elapsed_seconds,
        )

    async def _build_queries(self, question: str) -> List[str]:
        prompt = f"""
You are planning research for a forecasting question.

Question:
{question}

Generate up to {self.max_search_queries} useful search queries.

Requirements:
- Include at least one query aimed at recent developments.
- Include at least one query aimed at base rates or historical precedent.
- Include at least one query aimed at expert analysis, drivers, or constraints.
- Keep each query concise and specific.

Return one query per line, with no numbering.
"""

        response = await asyncio.wait_for(
            self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            ),
            timeout=60,
        )
        text = response.choices[0].message.content or ""

        queries: list[str] = []
        seen: set[str] = set()

        for raw_line in text.split("\n"):
            query = raw_line.strip().lstrip("-*•0123456789. ").strip()
            normalized = query.lower()
            if query and normalized not in seen:
                seen.add(normalized)
                queries.append(query)

        if not queries:
            queries = [
                f"{question} latest developments",
                f"{question} historical precedent",
                f"{question} expert analysis",
                f"{question} key drivers",
            ]

        return queries[: self.max_search_queries]

    async def _synthesise(
        self,
        question: str,
        evidence: List[str],
        warnings: List[str],
    ) -> str:
        joined = "\n\n".join(evidence) if evidence else "No external evidence was retrieved."
        warning_block = "\n".join(f"- {warning}" for warning in warnings) if warnings else "- None"

        prompt = f"""
You are preparing research for a forecasting bot.

Question:
{question}

Warnings:
{warning_block}

Evidence:
{joined}

Write a concise research brief with these headings:

Base rates
Arguments for YES
Arguments for NO
Key drivers
Uncertainties

Rules:
- Do not fabricate sources.
- If evidence is sparse, say so clearly.
- Distinguish recent developments from structural/background information when relevant.
- Keep the brief compact and decision-relevant.
"""

        response = await asyncio.wait_for(
            self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.research_temperature,
            ),
            timeout=90,
        )
        return response.choices[0].message.content or ""