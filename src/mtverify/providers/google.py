"""Google Cloud Translation, Basic (v2) REST endpoint with an API key."""

from __future__ import annotations

from mtverify.models import HealthResult
from mtverify.providers.base import HttpProvider, TranslationError

_URL = "https://translation.googleapis.com/language/translate/v2"


class GoogleProvider(HttpProvider):
    name = "google"
    max_batch = 100

    def _headers(self) -> dict[str, str]:
        if not self.settings.google_key:
            raise TranslationError("google: MTVERIFY_GOOGLE_KEY is not set")
        # header, never the query string: httpx logs request URLs at INFO
        return {"X-goog-api-key": self.settings.google_key}

    def health(self) -> HealthResult:
        def probe() -> str:
            r = self.request("GET", f"{_URL}/languages", params={"target": "en"}, headers=self._headers())
            n = len(r.json().get("data", {}).get("languages", []))
            return f"{n} languages available"

        return self._timed(probe)

    def translate(self, texts: list[str], source: str, target: str) -> list[str]:
        payload = {"q": texts, "source": source, "target": target, "format": "text"}
        r = self.request("POST", _URL, headers=self._headers(), json=payload)
        try:
            out = [t["translatedText"] for t in r.json()["data"]["translations"]]
        except (KeyError, TypeError, ValueError) as e:
            raise TranslationError(f"google: unexpected response {r.text[:200]!r}") from e
        if len(out) != len(texts):
            raise TranslationError(f"google: sent {len(texts)} texts, got {len(out)} back")
        return out
