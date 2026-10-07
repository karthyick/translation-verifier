"""Orchestrates one verification run: load cases, gate 1, translate, gate 2, gate 3, report."""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from mtverify import languages
from mtverify.config import Settings
from mtverify.gates import QualityConfig, QualityScorer, functional_gate, health_gate
from mtverify.models import CaseReport, GateResult, HealthResult, RunReport, TestCase
from mtverify.providers import get_provider
from mtverify.providers.base import Provider, TranslationError
from mtverify.providers.file import FileProvider

log = logging.getLogger("mtverify.runner")


@dataclass
class RunConfig:
    provider: str = "file"
    quality: QualityConfig = field(default_factory=QualityConfig)
    repeat: bool = False  # translate twice and demand identical output
    batch_size: int | None = None
    fail_fast: bool = False
    settings: Settings = field(default_factory=Settings)


class CaseFileError(ValueError):
    pass


def load_cases(path: str | Path) -> list[TestCase]:
    """Read JSON Lines. One bad line names the line number and stops the run before any API call."""
    cases: list[TestCase] = []
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                case = TestCase.model_validate(json.loads(line))
            except (json.JSONDecodeError, ValidationError) as e:
                raise CaseFileError(f"{path}:{n}: {e}") from e
            if case.id in seen:
                raise CaseFileError(f"{path}:{n}: duplicate case id {case.id!r}")
            for code in (case.source_lang, case.target_lang):
                languages.get(code)  # raises UnknownLanguage with the full list
            seen.add(case.id)
            cases.append(case)
    if not cases:
        raise CaseFileError(f"{path}: no cases")
    return cases


def _batches(items: list[TestCase], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _translate_all(provider: Provider, cases: list[TestCase], cfg: RunConfig
                   ) -> tuple[dict[str, str], dict[str, str], dict[str, float], dict[str, str]]:
    """Returns hyps, second hyps (if repeat), latency per case, error per case."""
    hyps: dict[str, str] = {}
    seconds: dict[str, str] = {}
    latency: dict[str, float] = {}
    errors: dict[str, str] = {}
    if isinstance(provider, FileProvider):
        for c, h in zip(cases, provider.translate_cases(cases), strict=True):
            hyps[c.id], latency[c.id] = h, 0.0
            if cfg.repeat:
                seconds[c.id] = h
        return hyps, seconds, latency, errors
    by_pair: dict[tuple[str, str], list[TestCase]] = defaultdict(list)
    for c in cases:
        by_pair[(c.source_lang, c.target_lang)].append(c)
    size = cfg.batch_size or provider.max_batch
    for (src, tgt), group in by_pair.items():
        for batch in _batches(group, size):
            texts = [c.source for c in batch]
            t0 = time.perf_counter()
            try:
                out = provider.translate(texts, src, tgt)
                second = provider.translate(texts, src, tgt) if cfg.repeat else None
            except TranslationError as e:
                log.error("%s->%s batch of %d failed: %s", src, tgt, len(batch), e)
                for c in batch:
                    errors[c.id] = str(e)
                if cfg.fail_fast:
                    return hyps, seconds, latency, errors
                continue
            per = (time.perf_counter() - t0) * 1000 / len(batch)
            for i, c in enumerate(batch):
                hyps[c.id] = out[i]
                latency[c.id] = per
                if second is not None:
                    seconds[c.id] = second[i]
    return hyps, seconds, latency, errors


def _summarize(report_cases: list[CaseReport], health: HealthResult) -> dict:
    by_status = defaultdict(int)
    by_lang: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    failing_checks: dict[str, int] = defaultdict(int)
    metric_vals: dict[str, list[float]] = defaultdict(list)
    for c in report_cases:
        by_status[c.status] += 1
        by_lang[c.target_lang][c.status] += 1
        for g in c.gates:
            for ch in g.checks:
                if ch.status == "fail":
                    failing_checks[f"{g.gate}.{ch.name}"] += 1
                if ch.value is not None and ch.name in ("chrf", "bleu", "comet", "labse"):
                    metric_vals[ch.name].append(ch.value)
    means = {k: round(sum(v) / len(v), 2) for k, v in metric_vals.items() if v}
    lat = [c.latency_ms for c in report_cases if c.latency_ms is not None]
    lat_sorted = sorted(lat)
    p95 = lat_sorted[int(0.95 * (len(lat_sorted) - 1))] if lat_sorted else None
    return {
        "health_ok": health.ok,
        "total": len(report_cases),
        "by_status": dict(by_status),
        "languages": len(by_lang),
        "by_language": {k: dict(v) for k, v in sorted(by_lang.items())},
        "failing_checks": dict(sorted(failing_checks.items(), key=lambda kv: -kv[1])),
        "metric_means": means,
        "latency_ms_p95": round(p95, 1) if p95 is not None else None,
    }


def run(cases_path: str | Path, cfg: RunConfig | None = None) -> RunReport:
    cfg = cfg or RunConfig()
    started = datetime.now(timezone.utc)
    cases = load_cases(cases_path)
    provider = get_provider(cfg.provider, cfg.settings)
    if isinstance(provider, FileProvider):
        provider.load(cases)
    log.info("run: provider=%s cases=%d settings=%s", provider.name, len(cases), cfg.settings.redacted())
    try:
        health, health_gate_result = health_gate(provider)
        if not health.ok:
            log.error("health failed: %s", health.detail)
            return RunReport(provider=provider.name, started_at=started,
                             finished_at=datetime.now(timezone.utc), health=health, cases=[],
                             summary={"health_ok": False, "total": 0})
        hyps, seconds, latency, errors = _translate_all(provider, cases, cfg)
    finally:
        provider.close()

    scored = [c for c in cases if c.id in hyps]
    quality = QualityScorer(cfg.quality).score(scored, [hyps[c.id] for c in scored]) if scored else []
    q_by_id = {c.id: g for c, g in zip(scored, quality, strict=True)}

    reports: list[CaseReport] = []
    for c in cases:
        if c.id in errors:
            reports.append(CaseReport(case_id=c.id, source_lang=c.source_lang, target_lang=c.target_lang,
                                      source=c.source, hypothesis=None, reference=c.reference,
                                      error=errors[c.id], gates=[health_gate_result]))
            continue
        if c.id not in hyps:  # fail_fast skipped it
            reports.append(CaseReport(case_id=c.id, source_lang=c.source_lang, target_lang=c.target_lang,
                                      source=c.source, hypothesis=None, reference=c.reference,
                                      error="not run (fail-fast)", gates=[]))
            continue
        h = hyps[c.id]
        gates: list[GateResult] = [
            health_gate_result,
            functional_gate(c, h, seconds.get(c.id) if cfg.repeat else None),
            q_by_id[c.id],
        ]
        reports.append(CaseReport(case_id=c.id, source_lang=c.source_lang, target_lang=c.target_lang,
                                  source=c.source, hypothesis=h, reference=c.reference,
                                  latency_ms=round(latency[c.id], 1), gates=gates))
    return RunReport(provider=provider.name, started_at=started, finished_at=datetime.now(timezone.utc),
                     health=health, cases=reports, summary=_summarize(reports, health))
