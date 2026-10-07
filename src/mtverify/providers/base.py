"""Provider contract plus a shared HTTP helper with retry and backoff."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable

import httpx

from mtverify.config import Settings
from mtverify.models import HealthResult

log = logging.getLogger("mtverify.providers")


class TranslationError(Exception):
    """Permanent failure: bad key, bad language, malformed response."""


class TransientError(TranslationError):
    """Retryable failure: 429, 5xx, timeout, connection reset."""


class Provider(ABC):
    name: str = "base"
    max_batch: int = 25

    def __init__(self, settings: Settings):
        self.settings = settings

    @abstractmethod
    def health(self) -> HealthResult: ...

    @abstractmethod
    def translate(self, texts: list[str], source: str, target: str) -> list[str]: ...

    def close(self) -> None:
        """Release connections. Providers without a transport have nothing to do."""
        return None


class HttpProvider(Provider):
    """Base for REST providers. Subclasses build the request, this class handles transport."""

    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        super().__init__(settings)
        self._client = httpx.Client(timeout=settings.timeout_s, transport=transport)
        self._sleep = sleep

    def close(self) -> None:
        self._client.close()

    def request(self, method: str, url: str, **kw) -> httpx.Response:
        """Send with retry. 429 and 5xx retry with exponential backoff; 4xx fail at once."""
        attempts = max(1, self.settings.retries + 1)
        delay = 0.5
        last: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                resp = self._client.request(method, url, **kw)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last = TransientError(f"{self.name}: {type(e).__name__}: {e}")
            else:
                if resp.status_code < 400:
                    return resp
                body = resp.text[:300]
                if resp.status_code == 429 or resp.status_code >= 500:
                    last = TransientError(f"{self.name}: HTTP {resp.status_code}: {body}")
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        delay = max(delay, float(retry_after))
                else:
                    raise TranslationError(f"{self.name}: HTTP {resp.status_code}: {body}")
            if attempt < attempts:
                log.warning("%s attempt %d/%d failed (%s); retry in %.1fs", self.name, attempt,
                            attempts, last, delay)
                self._sleep(delay)
                delay = min(delay * 2, 30.0)
        assert last is not None
        raise last

    def _timed(self, fn: Callable[[], str]) -> HealthResult:
        t0 = time.perf_counter()
        try:
            detail = fn()
        except TranslationError as e:
            return HealthResult(ok=False, detail=str(e), latency_ms=(time.perf_counter() - t0) * 1000)
        return HealthResult(ok=True, detail=detail, latency_ms=(time.perf_counter() - t0) * 1000)
