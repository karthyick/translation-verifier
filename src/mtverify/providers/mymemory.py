"""MyMemory (api.mymemory.translated.net): a free public translation API run by Translated.

Not an in-memory store: it is a website API, so it needs internet. No key needed. Anonymous use is
rate-limited per day (about 5,000 words); set MTVERIFY_MYMEMORY_EMAIL to raise the limit.
Text sent here leaves your machine, so do not send private data.
"""

from __future__ import annotations

import os

from mtverify.models import HealthResult
from mtverify.patterns import protect, restore
from mtverify.providers.base import HttpProvider, TranslationError

_URL = "https://api.mymemory.translated.net/get"
_MAX_CHARS = 500  # the API rejects longer queries


class MyMemoryProvider(HttpProvider):
    name = "mymemory"
    max_batch = 1  # one text per request

    def _one(self, text: str, source: str, target: str) -> str:
        if len(text) > _MAX_CHARS:
            raise TranslationError(f"mymemory: text longer than {_MAX_CHARS} chars")
        params = {"q": text, "langpair": f"{source}|{target}"}
        email = os.environ.get("MTVERIFY_MYMEMORY_EMAIL")
        if email:
            params["de"] = email
        r = self.request("GET", _URL, params=params)
        try:
            d = r.json()
            status = int(d.get("responseStatus", 0))
            out = d["responseData"]["translatedText"]
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            raise TranslationError(f"mymemory: unexpected response {r.text[:200]!r}") from e
        if status != 200:
            raise TranslationError(f"mymemory: {d.get('responseDetails') or status}")
        return out

    def health(self) -> HealthResult:
        return self._timed(lambda: f"'Hello' -> {self._one('Hello', 'en', 'fr')!r}")

    def translate(self, texts: list[str], source: str, target: str) -> list[str]:
        out = []
        for t in texts:
            masked, saved = protect(t)
            out.append(restore(self._one(masked, source, target), saved))
        return out
