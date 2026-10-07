import json

import httpx
import pytest

from mtverify.providers import get_provider
from mtverify.providers.azure import AzureProvider
from mtverify.providers.base import TransientError, TranslationError
from mtverify.providers.deepl import DeepLProvider
from mtverify.providers.google import GoogleProvider


def _provider(cls, settings, handler):
    return cls(settings, transport=httpx.MockTransport(handler), sleep=lambda s: None)


def test_deepl_translate_and_health(settings):
    seen = {}

    def handler(req: httpx.Request):
        seen["auth"] = req.headers["Authorization"]
        seen["host"] = req.url.host
        if req.url.path == "/v2/usage":
            return httpx.Response(200, json={"character_count": 10, "character_limit": 500000})
        seen["payload"] = json.loads(req.read())
        return httpx.Response(200, json={"translations": [{"text": "Hallo"}, {"text": "Welt"}]})

    p = _provider(DeepLProvider, settings, handler)
    assert p.health().ok
    assert p.translate(["Hello", "World"], "en", "de") == ["Hallo", "Welt"]
    assert seen["auth"] == "DeepL-Auth-Key k:fx"
    assert seen["host"] == "api-free.deepl.com"
    assert seen["payload"]["target_lang"] == "DE" and seen["payload"]["text"] == ["Hello", "World"]


def test_google_parses_v2_shape(settings):
    def handler(req: httpx.Request):
        assert req.headers["X-goog-api-key"] == "g"
        return httpx.Response(200, json={"data": {"translations": [{"translatedText": "Bonjour"}]}})

    p = _provider(GoogleProvider, settings, handler)
    assert p.translate(["Hello"], "en", "fr") == ["Bonjour"]


def test_azure_headers_and_shape(settings):
    def handler(req: httpx.Request):
        assert req.headers["Ocp-Apim-Subscription-Key"] == "a"
        assert req.headers["Ocp-Apim-Subscription-Region"] == "eastus"
        assert req.url.params["to"] == "zh-Hans"
        return httpx.Response(200, json=[{"translations": [{"text": "你好", "to": "zh-Hans"}]}])

    p = _provider(AzureProvider, settings, handler)
    assert p.translate(["Hello"], "en", "zh") == ["你好"]


def test_retry_on_429_then_success(settings):
    calls = {"n": 0}

    def handler(req: httpx.Request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "1"}, text="slow down")
        return httpx.Response(200, json={"translations": [{"text": "ok"}]})

    p = _provider(DeepLProvider, settings, handler)
    assert p.translate(["x"], "en", "de") == ["ok"]
    assert calls["n"] == 2


def test_gives_up_after_retries_with_transient_error(settings):
    def handler(req: httpx.Request):
        return httpx.Response(503, text="down")

    p = _provider(DeepLProvider, settings, handler)
    with pytest.raises(TransientError, match="503"):
        p.translate(["x"], "en", "de")


def test_4xx_fails_immediately(settings):
    calls = {"n": 0}

    def handler(req: httpx.Request):
        calls["n"] += 1
        return httpx.Response(403, text="bad key")

    p = _provider(DeepLProvider, settings, handler)
    with pytest.raises(TranslationError, match="403"):
        p.translate(["x"], "en", "de")
    assert calls["n"] == 1


def test_count_mismatch_is_an_error(settings):
    def handler(req: httpx.Request):
        return httpx.Response(200, json={"translations": [{"text": "only one"}]})

    p = _provider(DeepLProvider, settings, handler)
    with pytest.raises(TranslationError, match="got 1"):
        p.translate(["a", "b"], "en", "de")


def test_missing_key_is_clear(settings):
    from mtverify.config import Settings

    p = DeepLProvider(Settings(deepl_key=None))
    with pytest.raises(TranslationError, match="MTVERIFY_DEEPL_KEY"):
        p.translate(["x"], "en", "de")


def test_unknown_provider():
    with pytest.raises(TranslationError, match="unknown provider"):
        get_provider("nope")
