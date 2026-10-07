"""Write a run as JSON (for machines) and Markdown (for people)."""

from __future__ import annotations

from pathlib import Path

from mtverify.models import RunReport

_ICON = {"pass": "PASS", "fail": "FAIL", "skip": "skip", "error": "ERROR"}


def to_json(report: RunReport) -> str:
    data = report.model_dump(mode="json")
    for c, cr in zip(data["cases"], report.cases, strict=True):
        c["status"] = cr.status
    data["exit_code"] = report.exit_code
    import json

    return json.dumps(data, ensure_ascii=False, indent=2)


def to_markdown(report: RunReport) -> str:
    s = report.summary
    lines = [
        f"# mtverify run: {report.provider}",
        "",
        f"started {report.started_at.isoformat(timespec='seconds')}  "
        f"finished {report.finished_at.isoformat(timespec='seconds')}  exit code {report.exit_code}",
        "",
        f"health: {'ok' if report.health.ok else 'FAIL'} ({report.health.detail})",
        "",
        "| total | pass | fail | skip | error | languages | p95 ms |",
        "|---|---|---|---|---|---|---|",
    ]
    bs = s.get("by_status", {})
    lines.append(
        f"| {s.get('total', 0)} | {bs.get('pass', 0)} | {bs.get('fail', 0)} | {bs.get('skip', 0)} | "
        f"{bs.get('error', 0)} | {s.get('languages', 0)} | {s.get('latency_ms_p95')} |"
    )
    if s.get("metric_means"):
        lines += ["", "| metric | mean |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in s["metric_means"].items()]
    if s.get("failing_checks"):
        lines += ["", "| failing check | cases |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in s["failing_checks"].items()]
    if s.get("by_language"):
        lines += ["", "| lang | pass | fail | error |", "|---|---|---|---|"]
        for lang, counts in s["by_language"].items():
            lines.append(f"| {lang} | {counts.get('pass', 0)} | {counts.get('fail', 0)} | {counts.get('error', 0)} |")
    lines += ["", "## Cases", ""]
    for c in report.cases:
        lines.append(f"### {_ICON[c.status]} {c.case_id} ({c.source_lang}->{c.target_lang})")
        lines.append("")
        lines.append(f"- source: `{c.source}`")
        if c.hypothesis is not None:
            lines.append(f"- output: `{c.hypothesis}`")
        if c.reference:
            lines.append(f"- reference: `{c.reference}`")
        if c.error:
            lines.append(f"- error: {c.error}")
        for g in c.gates:
            bad = [ch for ch in g.checks if ch.status in ("fail", "error")]
            if bad:
                for ch in bad:
                    lines.append(f"- {g.gate}.{ch.name}: **{ch.status}** {ch.detail}")
            else:
                vals = ", ".join(f"{ch.name} {ch.detail}" for ch in g.checks if ch.value is not None)
                lines.append(f"- {g.gate}: {g.status}" + (f" ({vals})" if vals else ""))
        lines.append("")
    return "\n".join(lines)


def write(report: RunReport, out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = report.started_at.strftime("%Y%m%dT%H%M%SZ")
    j = out / f"mtverify-{report.provider}-{stamp}.json"
    m = out / f"mtverify-{report.provider}-{stamp}.md"
    j.write_text(to_json(report), encoding="utf-8")
    m.write_text(to_markdown(report), encoding="utf-8")
    (out / "latest.md").write_text(to_markdown(report), encoding="utf-8")
    (out / "latest.json").write_text(to_json(report), encoding="utf-8")
    return j, m
