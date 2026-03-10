from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol

from openai import OpenAI

logger = logging.getLogger(__name__)


# ---------------------------
# Data models
# ---------------------------

@dataclass
class SearchTask:
    query: str
    purpose: str
    recency: str = "general"   # "recent", "historical", "general"
    max_results: int = 5


@dataclass
class SearchResult:
    query: str
    purpose: str
    provider: str
    content: str


@dataclass
class ResearchDossier:
    question_text: str
    category: str
    key_drivers: list[str] = field(default_factory=list)
    base_rate_notes: list[str] = field(default_factory=list)
    recent_developments: list[str] = field(default_factory=list)
    supporting_evidence: list[str] = field(default_factory=list)
    counter_evidence: list[str] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)
    search_tasks: list[SearchTask] = field(default_factory=list)
    raw_results: list[SearchResult] = field(default_factory=list)
    synthesis_markdown: str = ""

    def to_markdown(self) -> str:
        if self.synthesis_markdown:
            return self.synthesis_markdown

        def bullets(items: list[str]) -> str:
            return "\n".join(f"- {x}" for x in items) if items else "- None"

        return f"""
# Research dossier

## Category
{self.category}

## Key drivers
{bullets(self.key_drivers)}

## Base rates / historical context
{bullets(self.base_rate_notes)}

## Recent developments
{bullets(self.recent_developments)}

## Supporting evidence
{bullets(self.supporting_evidence)}

## Counter-evidence
{bullets(self.counter_evidence)}

## Uncertainties
{bullets(self.uncertainties)}
""".strip()


# ---------------------------
# Search provider abstraction
# ---------------------------

class SearchProvider(Protocol):
    name: str

    def search(self, task: SearchTask) -> SearchResult:
        ...


class NullSearchProvider:
    """
    Safe fallback when no provider is configured.
    Lets the rest of the pipeline run while you test structure.
    """
    name = "null"

    def search(self, task: SearchTask) -> SearchResult:
        return SearchResult(
            query=task.query,
            purpose=task.purpose,
            provider=self.name,
            content=f"[No search provider configured] Query would have been: {task.query}"
        )


class AskNewsProvider:
    """
    Minimal AskNews integration based on the example in the template README.
    Requires:
      ASKNEWS_CLIENT_ID
      ASKNEWS_SECRET
    """
    name = "asknews"

    def __init__(self) -> None:
        self.client_id = os.getenv("ASKNEWS_CLIENT_ID")
        self.client_secret = os.getenv("ASKNEWS_SECRET")
        if not self.client_id or not self.client_secret:
            raise ValueError("AskNews credentials missing")

    async def _search_async(self, task: SearchTask) -> str:
        from asknews_sdk import AsyncAskNewsSDK

        ask = AsyncAskNewsSDK(
            client_id=self.client_id,
            client_secret=self.client_secret,
            scopes=["news", "stories", "analytics", "chat"],
        )

        strategy = {
            "recent": "latest news",
            "historical": "news knowledge",
            "general": "news knowledge",
        }.get(task.recency, "news knowledge")

        response = await ask.news.search_news(
            query=task.query,
            n_articles=task.max_results,
            return_type="both",
            strategy=strategy,
        )
        return response.as_string

    def search(self, task: SearchTask) -> SearchResult:
        content = asyncio.run(self._search_async(task))
        return SearchResult(
            query=task.query,
            purpose=task.purpose,
            provider=self.name,
            content=content,
        )


# ---------------------------
# LLM helper
# ---------------------------

class LlmHelper:
    """
    OpenAI-compatible client.
    Works with OpenAI directly or OpenRouter if you set OPENAI_BASE_URL / OPENROUTER_BASE_URL.
    """
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.2,
    ) -> None:
        api_key = (
            os.getenv("OPENAI_API_KEY")
            or os.getenv("OPENROUTER_API_KEY")
        )
        if not api_key:
            raise ValueError("Need OPENAI_API_KEY or OPENROUTER_API_KEY")

        base_url = (
            os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENROUTER_BASE_URL")
            or ("https://openrouter.ai/api/v1" if os.getenv("OPENROUTER_API_KEY") else None)
        )

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model or os.getenv("RESEARCH_MODEL", "openai/gpt-4.1-mini")
        self.temperature = temperature

    def complete(self, system: str, user: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""


# ---------------------------
# Cache
# ---------------------------

class ResearchCache:
    def __init__(self, cache_dir: str = ".cache/research") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path_for_key(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def get(self, key: str) -> Optional[ResearchDossier]:
        path = self._path_for_key(key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            dossier = ResearchDossier(
                question_text=data["question_text"],
                category=data["category"],
                key_drivers=data.get("key_drivers", []),
                base_rate_notes=data.get("base_rate_notes", []),
                recent_developments=data.get("recent_developments", []),
                supporting_evidence=data.get("supporting_evidence", []),
                counter_evidence=data.get("counter_evidence", []),
                uncertainties=data.get("uncertainties", []),
                search_tasks=[SearchTask(**x) for x in data.get("search_tasks", [])],
                raw_results=[SearchResult(**x) for x in data.get("raw_results", [])],
                synthesis_markdown=data.get("synthesis_markdown", ""),
            )
            return dossier
        except Exception as exc:
            logger.warning("Failed to load cached research: %s", exc)
            return None

    def set(self, key: str, dossier: ResearchDossier) -> None:
        payload = asdict(dossier)
        path = self._path_for_key(key)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------
# Sequential research pipeline
# ---------------------------

class SequentialResearchPipeline:
    def __init__(
        self,
        llm: LlmHelper,
        search_provider: Optional[SearchProvider] = None,
        cache: Optional[ResearchCache] = None,
    ) -> None:
        self.llm = llm
        self.search_provider = search_provider or NullSearchProvider()
        self.cache = cache or ResearchCache()

    def run(
        self,
        question_text: str,
        question_id: Optional[str] = None,
        resolution_criteria: str = "",
        fine_print: str = "",
        use_cache: bool = True,
    ) -> ResearchDossier:
        cache_key = json.dumps(
            {
                "question_id": question_id,
                "question_text": question_text,
                "resolution_criteria": resolution_criteria,
                "fine_print": fine_print,
                "provider": self.search_provider.name,
                "model": self.llm.model,
                "version": 1,
            },
            sort_keys=True,
        )

        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        category = self._classify_question(question_text, resolution_criteria, fine_print)
        search_tasks = self._make_search_plan(question_text, category, resolution_criteria, fine_print)

        raw_results: list[SearchResult] = []
        for task in search_tasks:
            try:
                raw_results.append(self.search_provider.search(task))
            except Exception as exc:
                logger.warning("Search failed for '%s': %s", task.query, exc)
                raw_results.append(
                    SearchResult(
                        query=task.query,
                        purpose=task.purpose,
                        provider=self.search_provider.name,
                        content=f"[Search failed] {exc}",
                    )
                )

        dossier = self._synthesise(
            question_text=question_text,
            category=category,
            resolution_criteria=resolution_criteria,
            fine_print=fine_print,
            search_tasks=search_tasks,
            raw_results=raw_results,
        )

        if use_cache:
            self.cache.set(cache_key, dossier)

        return dossier

    def _classify_question(
        self,
        question_text: str,
        resolution_criteria: str,
        fine_print: str,
    ) -> str:
        system = (
            "You classify forecasting questions. "
            "Return exactly one lowercase label from this list:\n"
            "geopolitics, economics, technology, science, health, climate, sports, elections, business, general"
        )
        user = f"""
Question:
{question_text}

Resolution criteria:
{resolution_criteria}

Fine print:
{fine_print}
""".strip()

        label = self.llm.complete(system, user).strip().lower()
        allowed = {
            "geopolitics", "economics", "technology", "science", "health",
            "climate", "sports", "elections", "business", "general"
        }
        return label if label in allowed else "general"

    def _make_search_plan(
        self,
        question_text: str,
        category: str,
        resolution_criteria: str,
        fine_print: str,
    ) -> list[SearchTask]:
        system = (
            "You are a forecasting research planner.\n"
            "Produce between 3 and 5 search tasks as JSON only.\n"
            "Each task must have keys: query, purpose, recency, max_results.\n"
            "Use a mix of:\n"
            "- one broad background query\n"
            "- one recent developments query\n"
            "- one historical/base-rate query\n"
            "- optionally one query targeting a decisive driver\n"
            "Valid recency values: recent, historical, general."
        )

        user = f"""
Question:
{question_text}

Category:
{category}

Resolution criteria:
{resolution_criteria}

Fine print:
{fine_print}
""".strip()

        raw = self.llm.complete(system, user)

        try:
            data = json.loads(raw)
            tasks = [SearchTask(**item) for item in data]
            return tasks[:5]
        except Exception:
            logger.warning("Falling back to default search plan")
            return [
                SearchTask(
                    query=question_text,
                    purpose="Broad background on the question",
                    recency="general",
                    max_results=5,
                ),
                SearchTask(
                    query=f"{question_text} latest developments",
                    purpose="Recent developments most likely to move the forecast",
                    recency="recent",
                    max_results=5,
                ),
                SearchTask(
                    query=f"{question_text} history similar cases base rate",
                    purpose="Historical analogies and base rates",
                    recency="historical",
                    max_results=5,
                ),
            ]

    def _synthesise(
        self,
        question_text: str,
        category: str,
        resolution_criteria: str,
        fine_print: str,
        search_tasks: list[SearchTask],
        raw_results: list[SearchResult],
    ) -> ResearchDossier:
        joined_results = "\n\n".join(
            [
                f"### Query: {r.query}\nPurpose: {r.purpose}\nProvider: {r.provider}\n\n{r.content}"
                for r in raw_results
            ]
        )

        system = (
            "You are a forecasting research synthesiser.\n"
            "Your job is NOT to produce a probability yet.\n"
            "Instead produce a structured JSON object with keys:\n"
            "key_drivers, base_rate_notes, recent_developments, supporting_evidence, "
            "counter_evidence, uncertainties, synthesis_markdown.\n"
            "All list fields should contain short bullet-sized strings.\n"
            "Be conservative and avoid inventing facts."
        )

        user = f"""
Question:
{question_text}

Category:
{category}

Resolution criteria:
{resolution_criteria}

Fine print:
{fine_print}

Search results:
{joined_results}
""".strip()

        raw = self.llm.complete(system, user)

        try:
            data = json.loads(raw)
            return ResearchDossier(
                question_text=question_text,
                category=category,
                key_drivers=data.get("key_drivers", []),
                base_rate_notes=data.get("base_rate_notes", []),
                recent_developments=data.get("recent_developments", []),
                supporting_evidence=data.get("supporting_evidence", []),
                counter_evidence=data.get("counter_evidence", []),
                uncertainties=data.get("uncertainties", []),
                search_tasks=search_tasks,
                raw_results=raw_results,
                synthesis_markdown=data.get("synthesis_markdown", ""),
            )
        except Exception:
            logger.warning("Falling back to plain-text synthesis")
            synthesis = self.llm.complete(
                "Produce a concise structured markdown research dossier for a forecasting question.",
                f"Question:\n{question_text}\n\nSearch results:\n{joined_results}",
            )
            return ResearchDossier(
                question_text=question_text,
                category=category,
                search_tasks=search_tasks,
                raw_results=raw_results,
                synthesis_markdown=synthesis,
            )