"""Offline provider: the hypothesis comes from the case file itself.

Use it to grade translations produced elsewhere (a vendor export, a human draft, a model dump)
and for CI, where no key may exist.
"""

from __future__ import annotations

from mtverify.config import Settings
from mtverify.models import HealthResult
from mtverify.providers.base import Provider, TranslationError


class FileProvider(Provider):
    name = "file"
    max_batch = 10_000

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self._by_id: dict[str, str] = {}

    def load(self, cases) -> None:
        for c in cases:
            if c.hypothesis is None:
                raise TranslationError(f"case {c.id}: file provider needs a 'hypothesis' field")
            self._by_id[c.id] = c.hypothesis

    def translate_cases(self, cases) -> list[str]:
        """Per-case lookup, so two cases with the same source text stay distinct."""
        return [self._by_id[c.id] for c in cases]

    def health(self) -> HealthResult:
        return HealthResult(ok=True, detail=f"{len(self._by_id)} hypotheses loaded from file", latency_ms=0.0)

    def translate(self, texts: list[str], source: str, target: str) -> list[str]:
        raise TranslationError("file provider translates by case id; the runner calls translate_cases()")
