"""Gate 1: is the service alive and is the key valid."""

from __future__ import annotations

from mtverify.models import CheckResult, GateResult, HealthResult
from mtverify.providers.base import Provider


def health_gate(provider: Provider) -> tuple[HealthResult, GateResult]:
    try:
        hr = provider.health()
    except Exception as e:  # a provider bug must not take the run down
        hr = HealthResult(ok=False, detail=f"{type(e).__name__}: {e}")
    check = CheckResult(
        name="health",
        status="pass" if hr.ok else "fail",
        detail=hr.detail,
        value=hr.latency_ms,
    )
    return hr, GateResult.from_checks("health", [check])
