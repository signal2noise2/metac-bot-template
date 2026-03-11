import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional


class ResearchCache:
    """
    Simple filesystem cache for research queries.

    Each cached entry stores:
        - provider
        - query
        - max_results
        - timestamp
        - result text

    Cache layout:

        .cache/research/
            asknews/
                <hash>.json
            web/
                <hash>.json
    """

    def __init__(self, cache_dir: str = ".cache/research"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _provider_dir(self, provider: str) -> Path:
        p = self.cache_dir / provider
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _make_key(self, provider: str, query: str, max_results: int) -> str:
        key = f"{provider}:{query}:{max_results}"
        return hashlib.sha256(key.encode()).hexdigest()

    def _cache_path(self, provider: str, query: str, max_results: int) -> Path:
        key = self._make_key(provider, query, max_results)
        return self._provider_dir(provider) / f"{key}.json"

    def get(
        self,
        provider: str,
        query: str,
        max_results: int,
        ttl_seconds: Optional[int] = None,
    ) -> Optional[str]:
        """
        Return cached result if present and not expired.
        """
        path = self._cache_path(provider, query, max_results)

        if not path.exists():
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            timestamp = payload.get("timestamp", 0)

            if ttl_seconds is not None:
                if time.time() - timestamp > ttl_seconds:
                    return None

            return payload.get("result")

        except Exception:
            return None

    def set(
        self,
        provider: str,
        query: str,
        max_results: int,
        result: str,
    ) -> None:
        """
        Store query result in cache.
        """
        path = self._cache_path(provider, query, max_results)

        payload = {
            "provider": provider,
            "query": query,
            "max_results": max_results,
            "timestamp": time.time(),
            "result": result,
        }

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
        except Exception:
            pass