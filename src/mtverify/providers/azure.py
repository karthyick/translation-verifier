"""Azure AI Translator, Text Translation REST v3.0."""

from __future__ import annotations

from mtverify.models import HealthResult
from mtverify.providers.base import HttpProvider, TranslationError


class AzureProvider(HttpProvider):
    name = "azure"
    max_batch = 100  # API limit is 1000 items / 50k chars; keep well below

    def _headers(self) -> dict[str, str]:
        if not self.settings.azure_key:
            raise TranslationError("azure: MTVERIFY_AZURE_KEY is not set")
        h = {"Ocp-Apim-Subscription-Key": self.settings.azure_key, "Content-Type": "application/json"}
        if self.settings.azure_region:
            h["Ocp-Apim-Subscription-Region"] = self.settings.azure_region
        return h

    def health(self) -> HealthResult:
        def probe() -> str:
            # /languages needs no key, so send one real one-word translate as the probe.
            out = self.translate(["ping"], "en", "fr")
            return f"translate ok: 'ping' -> {out[0]!r}"

        return self._timed(probe)

    def translate(self, texts: list[str], source: str, target: str) -> list[str]:
        url = f"{self.settings.azure_endpoint.rstrip('/')}/translate"
        params = {"api-version": "3.0", "from": _azure_target(source), "to": _azure_target(target), "textType": "plain"}
        body = [{"Text": t} for t in texts]
        r = self.request("POST", url, params=params, headers=self._headers(), json=body)
        try:
            out = [item["translations"][0]["text"] for item in r.json()]
        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise TranslationError(f"azure: unexpected response {r.text[:200]!r}") from e
        if len(out) != len(texts):
            raise TranslationError(f"azure: sent {len(texts)} texts, got {len(out)} back")
        return out


def _azure_target(code: str) -> str:
    return {"zh": "zh-Hans", "pt": "pt-br"}.get(code, code)
