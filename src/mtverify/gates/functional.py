"""Gate 2: did the service break anything that is not language.

Every check is pure, deterministic and offline. Each returns one CheckResult; a check never
raises, so one bad string cannot stop the run.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

from mtverify import languages
from mtverify.models import CheckResult, GateResult, TestCase
from mtverify.patterns import BAD_CHARS as _BAD_CHARS
from mtverify.patterns import EMAIL as _EMAIL
from mtverify.patterns import HTML_TAG as _HTML_TAG
from mtverify.patterns import NUMBER as _NUMBER
from mtverify.patterns import NUMBER_SEPARATORS as _NUMBER_SEPARATORS
from mtverify.patterns import PLACEHOLDER as _PLACEHOLDER
from mtverify.patterns import URL as _URL
from mtverify.patterns import find_urls


def _normalize_digits(text: str) -> str:
    """Turn any Unicode digit (Arabic-Indic, Devanagari, Bengali ...) into ASCII."""
    out = []
    for ch in text:
        d = unicodedata.digit(ch, None)
        out.append(str(d) if d is not None else ch)
    return "".join(out)


def _counter_diff(expected: Counter, got: Counter) -> str:
    missing = expected - got
    extra = got - expected
    parts = []
    if missing:
        parts.append("missing " + ", ".join(sorted(missing.elements())))
    if extra:
        parts.append("extra " + ", ".join(sorted(extra.elements())))
    return "; ".join(parts)


def check_not_empty(src: str, hyp: str) -> CheckResult:
    if src.strip() and not hyp.strip():
        return CheckResult(name="not_empty", status="fail", detail="translation is empty")
    return CheckResult(name="not_empty", status="pass")


def check_placeholders(src: str, hyp: str) -> CheckResult:
    exp, got = Counter(_PLACEHOLDER.findall(src)), Counter(_PLACEHOLDER.findall(hyp))
    if exp == got:
        return CheckResult(name="placeholders", status="pass", detail=f"{sum(exp.values())} kept")
    return CheckResult(name="placeholders", status="fail", detail=_counter_diff(exp, got))


def _tag_key(m: re.Match) -> str:
    closing, name = m.group(1), m.group(2).lower()
    return f"</{name}>" if closing else f"<{name}>"  # <br> and <br/> count as the same tag


def check_html_tags(src: str, hyp: str) -> CheckResult:
    exp = Counter(_tag_key(m) for m in _HTML_TAG.finditer(src))
    got = Counter(_tag_key(m) for m in _HTML_TAG.finditer(hyp))
    if not exp and not got:
        return CheckResult(name="html_tags", status="skip", detail="no tags in source")
    if exp == got:
        return CheckResult(name="html_tags", status="pass", detail=f"{sum(exp.values())} tags kept")
    return CheckResult(name="html_tags", status="fail", detail=_counter_diff(exp, got))


def check_numbers(src: str, hyp: str) -> CheckResult:
    exp = Counter(_NUMBER.findall(_normalize_digits(src)))
    got = Counter(_NUMBER.findall(_normalize_digits(hyp)))
    if not exp:
        return CheckResult(name="numbers", status="skip", detail="no numbers in source")
    # Separators may change (1,000 -> 1.000). Compare on digits only as the fallback.
    strip = _NUMBER_SEPARATORS.sub
    if exp == got or Counter(strip("", n) for n in exp.elements()) == Counter(strip("", n) for n in got.elements()):
        return CheckResult(name="numbers", status="pass", detail=f"{sum(exp.values())} kept")
    return CheckResult(name="numbers", status="fail", detail=_counter_diff(exp, got))


def check_urls_emails(src: str, hyp: str) -> CheckResult:
    exp = Counter(find_urls(src) + _EMAIL.findall(src))
    got = Counter(find_urls(hyp) + _EMAIL.findall(hyp))
    if not exp:
        return CheckResult(name="urls_emails", status="skip", detail="none in source")
    if exp == got:
        return CheckResult(name="urls_emails", status="pass", detail=f"{sum(exp.values())} kept")
    return CheckResult(name="urls_emails", status="fail", detail=_counter_diff(exp, got))


def check_language(hyp: str, target: str) -> CheckResult:
    try:
        d = languages.detect(hyp, target)
    except languages.UnknownLanguage as e:
        return CheckResult(name="language", status="error", detail=str(e))
    return CheckResult(
        name="language",
        status="pass" if d.ok else "fail",
        detail=d.detail if d.ok else f"expected {d.expected}, {d.detail}",
        value=d.script_ratio,
    )


def check_untranslated(src: str, hyp: str, source_lang: str, target: str) -> CheckResult:
    if source_lang == target:
        return CheckResult(name="untranslated", status="skip", detail="same language")
    stripped = _PLACEHOLDER.sub("", _NUMBER.sub("", _URL.sub("", src)))
    if not any(ch.isalpha() for ch in stripped):
        return CheckResult(name="untranslated", status="skip", detail="nothing translatable")
    if src.strip().casefold() == hyp.strip().casefold():
        return CheckResult(name="untranslated", status="fail", detail="output equals input")
    return CheckResult(name="untranslated", status="pass")


_WORD = re.compile(r"[A-Za-z]{3,}")


def check_leftover_source(src: str, hyp: str, source_lang: str, target: str) -> CheckResult:
    """Non-Latin target still carrying source words, e.g. ja output with 'Pay now' left in English.

    Placeholders, tags, URLs and emails are ignored, and so are ALL-CAPS tokens like USD or API.
    """
    tgt = languages.get(target)
    if source_lang == target or tgt.script == "Latin" or languages.get(source_lang).script != "Latin":
        return CheckResult(name="leftover_source", status="skip", detail="same script family")
    from mtverify.patterns import strip_non_language

    def words(text: str) -> set[str]:
        return {w.lower() for w in _WORD.findall(strip_non_language(text)) if not w.isupper()}

    left = sorted(words(src) & words(hyp))
    if len(left) >= 2:
        return CheckResult(name="leftover_source", status="fail", detail="untranslated: " + ", ".join(left))
    return CheckResult(name="leftover_source", status="pass")


def check_length_ratio(src: str, hyp: str, target: str) -> CheckResult:
    if len(src.strip()) < 10:
        return CheckResult(name="length_ratio", status="skip", detail="source too short to judge")
    lo, hi = languages.get(target).length_ratio
    ratio = len(hyp) / max(1, len(src))
    ok = lo <= ratio <= hi
    return CheckResult(
        name="length_ratio",
        status="pass" if ok else "fail",
        detail=f"{ratio:.2f} (allowed {lo}-{hi})",
        value=ratio,
    )


def check_glossary(hyp: str, glossary: dict[str, str]) -> CheckResult:
    if not glossary:
        return CheckResult(name="glossary", status="skip", detail="no glossary")
    low = hyp.casefold()
    missing = [f"{s}->{t}" for s, t in glossary.items() if t.casefold() not in low]
    if missing:
        return CheckResult(name="glossary", status="fail", detail="missing " + ", ".join(missing))
    return CheckResult(name="glossary", status="pass", detail=f"{len(glossary)} terms present")


def check_encoding(hyp: str) -> CheckResult:
    if _BAD_CHARS.search(hyp):
        return CheckResult(name="encoding", status="fail", detail="replacement or control chars")
    return CheckResult(name="encoding", status="pass")


def check_idempotent(first: str, second: str | None) -> CheckResult:
    if second is None:
        return CheckResult(name="idempotent", status="skip", detail="not repeated")
    if first == second:
        return CheckResult(name="idempotent", status="pass")
    return CheckResult(name="idempotent", status="fail", detail="second call differs")


def functional_gate(case: TestCase, hyp: str, second_hyp: str | None = None) -> GateResult:
    src, tgt = case.source, case.target_lang
    checks = [
        check_not_empty(src, hyp),
        check_placeholders(src, hyp),
        check_html_tags(src, hyp),
        check_numbers(src, hyp),
        check_urls_emails(src, hyp),
        check_language(hyp, tgt),
        check_untranslated(src, hyp, case.source_lang, tgt),
        check_leftover_source(src, hyp, case.source_lang, tgt),
        check_length_ratio(src, hyp, tgt),
        check_glossary(hyp, case.glossary),
        check_encoding(hyp),
        check_idempotent(hyp, second_hyp),
    ]
    return GateResult.from_checks("functional", checks)
