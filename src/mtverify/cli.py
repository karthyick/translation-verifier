"""Command line entry point: mtverify run | translate | health | languages | init-cases."""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from importlib import resources
from pathlib import Path

from mtverify import __version__, languages
from mtverify import report as report_mod
from mtverify.config import Settings
from mtverify.gates import QualityConfig
from mtverify.models import TestCase
from mtverify.providers import PROVIDERS, TranslationError, get_provider
from mtverify.runner import CaseFileError, RunConfig, run, run_cases


def _build() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mtverify", description=__doc__)
    p.add_argument("--version", action="version", version=f"mtverify {__version__}")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run gates 1-3 over a case file")
    r.add_argument("--cases", required=True, help="JSON Lines file of test cases")
    r.add_argument("--provider", default="file", choices=sorted(PROVIDERS))
    r.add_argument("--report-dir", default="reports")
    r.add_argument("--chrf-min", type=float, default=50.0)
    r.add_argument("--bleu-min", type=float, default=None)
    r.add_argument("--comet", default=None, metavar="MODEL",
                   help="e.g. Unbabel/wmt22-comet-da or Unbabel/wmt22-cometkiwi-da (needs [comet])")
    r.add_argument("--comet-min", type=float, default=0.75)
    r.add_argument("--labse", action="store_true", help="source/output LaBSE cosine (needs [labse])")
    r.add_argument("--labse-min", type=float, default=0.75)
    r.add_argument("--gpus", type=int, default=0)
    r.add_argument("--repeat", action="store_true", help="translate twice, require identical output")
    r.add_argument("--batch-size", type=_positive_int, default=None)
    r.add_argument("--fail-fast", action="store_true")

    t = sub.add_parser("translate", help="translate text with a free provider and verify it through the gates")
    t.add_argument("text", nargs="*", help="one or more English strings")
    t.add_argument("--file", help="text file, one string per line")
    t.add_argument("--to", dest="target", required=True, help="target language code, e.g. ja")
    t.add_argument("--from", dest="source", default="en")
    t.add_argument("--provider", default="local", choices=sorted(k for k in PROVIDERS if k != "file"))
    t.add_argument("--model", default=None, help="local model id (default google/madlad400-3b-mt)")
    t.add_argument("--no-labse", action="store_true", help="skip the meaning check")
    t.add_argument("--labse-min", type=float, default=0.75)
    t.add_argument("--report-dir", default="reports")

    h = sub.add_parser("health", help="gate 1 only")
    h.add_argument("--provider", required=True, choices=sorted(k for k in PROVIDERS if k != "file"))

    sub.add_parser("languages", help="list supported target languages")

    i = sub.add_parser("init-cases", help="copy the bundled 27-language smoke set to a path")
    i.add_argument("--out", default="cases/smoke.jsonl")
    return p


def _positive_int(v: str) -> int:
    n = int(v)
    if n < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return n


def main(argv: list[str] | None = None) -> int:
    args = _build().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    # httpx logs every request URL at INFO; keep provider traffic out of CI logs
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.cmd == "languages":
        print(f"{'code':<5}{'name':<12}{'script':<12}{'detector':<14}bleu-tok")
        for lang in languages.REGISTRY.values():
            det = "script+lingua" if lang.lingua else "script only"
            print(f"{lang.code:<5}{lang.name:<12}{lang.script:<12}{det:<14}{lang.bleu_tokenize}")
        print(f"\n{len(languages.REGISTRY)} languages")
        return 0

    if args.cmd == "init-cases":
        src = resources.files("mtverify").joinpath("data/smoke.jsonl")
        dst = Path(args.out)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with resources.as_file(src) as p:
            shutil.copyfile(p, dst)
        print(f"wrote {dst}")
        return 0

    try:
        settings = Settings()
    except ValueError as e:
        print(f"settings error: {e}", file=sys.stderr)
        return 2

    if args.cmd == "translate":
        return _translate(args, settings)

    if args.cmd == "health":
        prov = get_provider(args.provider, settings)
        try:
            hr = prov.health()
        finally:
            prov.close()
        print(f"{prov.name}: {'ok' if hr.ok else 'FAIL'}  {hr.detail}  ({hr.latency_ms:.0f} ms)")
        return 0 if hr.ok else 2

    cfg = RunConfig(
        provider=args.provider,
        quality=QualityConfig(chrf_min=args.chrf_min, bleu_min=args.bleu_min, comet_model=args.comet,
                              comet_min=args.comet_min, labse=args.labse, labse_min=args.labse_min,
                              gpus=args.gpus),
        repeat=args.repeat,
        batch_size=args.batch_size,
        fail_fast=args.fail_fast,
        settings=settings,
    )
    try:
        rep = run(args.cases, cfg)
    except (CaseFileError, languages.UnknownLanguage, OSError, UnicodeDecodeError) as e:
        print(f"case file error: {e}", file=sys.stderr)
        return 2
    except TranslationError as e:
        print(f"provider error: {e}", file=sys.stderr)
        return 2
    j, m = report_mod.write(rep, args.report_dir)
    s = rep.summary
    bs = s.get("by_status", {})
    print(f"provider {rep.provider}  health {'ok' if rep.health.ok else 'FAIL'}  "
          f"cases {s.get('total', 0)}  pass {bs.get('pass', 0)}  fail {bs.get('fail', 0)}  "
          f"error {bs.get('error', 0)}  languages {s.get('languages', 0)}")
    if s.get("total") and not s.get("quality_scored"):
        print("WARNING quality gate scored 0 cases: add references or --comet Unbabel/wmt22-cometkiwi-da",
              file=sys.stderr)
    if s.get("metric_means"):
        print("means  " + "  ".join(f"{k} {v}" for k, v in s["metric_means"].items()))
    if s.get("failing_checks"):
        print("failing checks  " + "  ".join(f"{k}={v}" for k, v in s["failing_checks"].items()))
    print(f"report {m}")
    return rep.exit_code


def _translate(args, settings: Settings) -> int:
    import dataclasses

    texts = list(args.text)
    if args.file:
        try:
            texts += [ln.strip() for ln in Path(args.file).read_text(encoding="utf-8").splitlines() if ln.strip()]
        except (OSError, UnicodeDecodeError) as e:
            print(f"input error: {e}", file=sys.stderr)
            return 2
    if not texts:
        print("input error: give text arguments or --file", file=sys.stderr)
        return 2
    try:
        languages.get(args.source)
        languages.get(args.target)
    except languages.UnknownLanguage as e:
        print(f"input error: {e}", file=sys.stderr)
        return 2
    if args.model:
        settings = dataclasses.replace(settings, local_model=args.model)
    cases = [TestCase(id=f"t{i}", source=s, source_lang=args.source, target_lang=args.target)
             for i, s in enumerate(texts, 1)]
    cfg = RunConfig(provider=args.provider, settings=settings,
                    quality=QualityConfig(labse=not args.no_labse, labse_min=args.labse_min))
    try:
        rep = run_cases(cases, cfg)
    except TranslationError as e:
        print(f"provider error: {e}", file=sys.stderr)
        return 2
    _, m = report_mod.write(rep, args.report_dir)
    print(f"provider {rep.provider}  health {'ok' if rep.health.ok else 'FAIL'}  ({rep.health.detail})")
    if not rep.health.ok:
        return rep.exit_code
    print(f"{'#':<4}{'status':<7}{'source':<32}{'translation':<34}notes")
    for c in rep.cases:
        notes = []
        for g in c.gates:
            for ch in g.checks:
                if ch.status in ("fail", "error"):
                    notes.append(f"{ch.name}: {ch.detail}")
                elif ch.name == "labse" and ch.value is not None:
                    notes.append(f"meaning {ch.value:.2f}")
        out = c.hypothesis if c.hypothesis is not None else (c.error or "")
        print(f"{c.case_id:<4}{c.status.upper():<7}{c.source[:30]:<32}{out[:32]:<34}{'; '.join(notes)}")
    bs = rep.summary.get("by_status", {})
    print(f"pass {bs.get('pass', 0)}  fail {bs.get('fail', 0)}  error {bs.get('error', 0)}  report {m}")
    return rep.exit_code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
