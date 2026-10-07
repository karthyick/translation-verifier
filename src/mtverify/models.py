"""Data contracts shared by every gate, provider and report."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Status = Literal["pass", "fail", "skip", "error"]


class TestCase(BaseModel):
    """One source string that must be translated into one target language."""

    __test__ = False  # keep pytest from collecting this model
    model_config = ConfigDict(extra="forbid")  # a typo like "refrence" must not silently skip gate 3

    id: str
    source: str
    source_lang: str = "en"
    target_lang: str
    reference: str | None = None
    hypothesis: str | None = None
    glossary: dict[str, str] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)

    @field_validator("source_lang", "target_lang")
    @classmethod
    def _lower(cls, v: str) -> str:
        return v.strip().lower()

    @field_validator("id")
    @classmethod
    def _non_empty_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("case id must not be empty")
        return v


class CheckResult(BaseModel):
    name: str
    status: Status
    detail: str = ""
    value: float | None = None
    threshold: float | None = None


class GateResult(BaseModel):
    gate: Literal["health", "functional", "quality"]
    status: Status
    checks: list[CheckResult] = Field(default_factory=list)

    @staticmethod
    def from_checks(gate: str, checks: list[CheckResult]) -> GateResult:
        return GateResult(gate=gate, status=worst_status([c.status for c in checks]), checks=checks)


class CaseReport(BaseModel):
    case_id: str
    source_lang: str
    target_lang: str
    source: str
    hypothesis: str | None
    reference: str | None
    latency_ms: float | None = None
    gates: list[GateResult] = Field(default_factory=list)
    error: str | None = None

    @property
    def status(self) -> Status:
        if self.error:
            return "error"
        return worst_status([g.status for g in self.gates])


class HealthResult(BaseModel):
    ok: bool
    detail: str = ""
    latency_ms: float | None = None


class RunReport(BaseModel):
    provider: str
    started_at: datetime
    finished_at: datetime
    health: HealthResult
    cases: list[CaseReport]
    summary: dict = Field(default_factory=dict)

    @property
    def exit_code(self) -> int:
        """0 all pass, 1 some case failed, 2 health failed or run-level error."""
        if not self.health.ok:
            return 2
        statuses = [c.status for c in self.cases]
        if any(s == "error" for s in statuses):
            return 2
        if any(s == "fail" for s in statuses):
            return 1
        return 0


# skip < pass so a gate with real passes and some skips reads "pass"; an all-skip gate reads "skip"
_RANK = {"skip": 0, "pass": 1, "fail": 2, "error": 3}


def worst_status(statuses: list[Status]) -> Status:
    if not statuses:
        return "skip"
    return max(statuses, key=lambda s: _RANK[s])
