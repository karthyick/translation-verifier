"""Runtime settings. Everything comes from environment variables so no key ever lands in a file."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(*names: str, default: str | None = None) -> str | None:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v.strip()
    return default


def _num(name: str, default: str, cast):
    raw = _env(name, default=default)
    try:
        return cast(raw)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name} must be a {cast.__name__}, got {raw!r}") from e


@dataclass(frozen=True)
class Settings:
    deepl_key: str | None = field(default_factory=lambda: _env("MTVERIFY_DEEPL_KEY", "DEEPL_AUTH_KEY"))
    google_key: str | None = field(
        default_factory=lambda: _env("MTVERIFY_GOOGLE_KEY", "GOOGLE_TRANSLATE_API_KEY")
    )
    azure_key: str | None = field(
        default_factory=lambda: _env("MTVERIFY_AZURE_KEY", "AZURE_TRANSLATOR_KEY")
    )
    azure_region: str | None = field(
        default_factory=lambda: _env("MTVERIFY_AZURE_REGION", "AZURE_TRANSLATOR_REGION")
    )
    azure_endpoint: str = field(
        default_factory=lambda: _env(
            "MTVERIFY_AZURE_ENDPOINT", default="https://api.cognitive.microsofttranslator.com"
        )
        or "https://api.cognitive.microsofttranslator.com"
    )
    timeout_s: float = field(default_factory=lambda: _num("MTVERIFY_TIMEOUT", "30", float))
    retries: int = field(default_factory=lambda: _num("MTVERIFY_RETRIES", "3", int))

    def redacted(self) -> dict[str, str]:
        """Safe to log: never shows a key, only whether one is set."""
        return {
            "deepl_key": "set" if self.deepl_key else "unset",
            "google_key": "set" if self.google_key else "unset",
            "azure_key": "set" if self.azure_key else "unset",
            "azure_region": self.azure_region or "unset",
            "timeout_s": str(self.timeout_s),
            "retries": str(self.retries),
        }
