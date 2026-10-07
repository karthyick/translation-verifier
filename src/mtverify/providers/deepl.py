"""DeepL API v2. Free keys end in ':fx' and use api-free.deepl.com."""

from __future__ import annotations

from mtverify.models import HealthResult
from mtverify.providers.base import HttpProvider, TranslationError


class DeepLProvider(HttpProvider):
    name = "deepl"
    max_batch = 50

    @property
    def base_url(self) -> str:
        key = self.settings.deepl_key or ""
        return "https://api-free.deepl.com" if key.endswith(":fx") else "https://api.deepl.com"

    def _headers(self) -> dict[str, str]:
        if not self.settings.deepl_key:
            raise TranslationError("deepl: MTVERIFY_DEEPL_KEY is not set")
        return {"Authorization": f"DeepL-Auth-Key {self.settings.deepl_key}"}

    def health(self) -> HealthResult:
        def probe() -> str:
            r = self.request("GET", f"{self.base_url}/v2/usage", headers=self._headers())
            d = r.json()
            return f"usage {d.get('character_count')}/{d.get('character_limit')} chars"

        return self._timed(probe)

    def translate(self, texts: list[str], source: str, target: str) -> list[str]:
        payload = {
            "text": texts,
            "source_lang": source.upper(),
            "target_lang": _deepl_target(target),
            "preserve_formatting": True,
        }
        r = self.request("POST", f"{self.base_url}/v2/translate", headers=self._headers(), json=payload)
        try:
            items = r.json()["translations"]
            out = [i["text"] for i in items]
        except (KeyError, TypeError, ValueError) as e:
            raise TranslationError(f"deepl: unexpected response {r.text[:200]!r}") from e
        if len(out) != len(texts):
            raise TranslationError(f"deepl: sent {len(texts)} texts, got {len(out)} back")
        return out


def _deepl_target(code: str) -> str:
    # DeepL wants a variant for these two.
    return {"pt": "PT-BR", "zh": "ZH-HANS"}.get(code, code.upper())
