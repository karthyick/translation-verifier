"""Provider registry. Add a provider: subclass Provider, register it here."""

from __future__ import annotations

from mtverify.config import Settings
from mtverify.providers.azure import AzureProvider
from mtverify.providers.base import Provider, TransientError, TranslationError
from mtverify.providers.deepl import DeepLProvider
from mtverify.providers.file import FileProvider
from mtverify.providers.google import GoogleProvider
from mtverify.providers.local import LocalProvider
from mtverify.providers.mymemory import MyMemoryProvider

PROVIDERS: dict[str, type[Provider]] = {
    "file": FileProvider,
    "deepl": DeepLProvider,
    "google": GoogleProvider,
    "azure": AzureProvider,
    "local": LocalProvider,
    "mymemory": MyMemoryProvider,
}


def get_provider(name: str, settings: Settings | None = None) -> Provider:
    try:
        cls = PROVIDERS[name.lower()]
    except KeyError as e:
        raise TranslationError(f"unknown provider {name!r}; choose from {', '.join(PROVIDERS)}") from e
    return cls(settings or Settings())


__all__ = [
    "PROVIDERS",
    "Provider",
    "TransientError",
    "TranslationError",
    "get_provider",
]
