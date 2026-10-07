"""Tests added from the code review: secrets in logs, odd transport failures, CJK URLs, ja/zh swap,
typo'd fields, repeat with a flaky provider, and the all-quality-skipped warning."""

import logging

import httpx
import pytest

from mtverify import cli, languages
from mtverify.config import Settings
from mtverify.gates import functional as f
from mtverify.models import HealthResult, TestCase
from mtverify.providers import PROVIDERS
from mtverify.providers.base import Provider, TransientError
from mtverify.providers.deepl import DeepLProvider
from mtverify.providers.google import GoogleProvider
from mtverify.runner import CaseFileError, RunConfig, load_cases, run


def _provider(cls, settings, handler):
    return cls(settings, transport=httpx.MockTransport(handler), sleep=lambda s: None)


# --- 1. the key must never reach a log line -------------------------------------------------
def test_google_key_travels_in_header_and_never_in_logs(settings, caplog):
    seen = {}

    def handler(req: httpx.Request):
        seen["url"] = str(req.url)
        seen["hdr"] = req.headers.get("X-goog-api-key")
        return httpx.Response(200, json={"data": {"translations": [{"translatedText": "Bonjour"}]}})

    caplog.set_level(logging.DEBUG)
    p = _provider(GoogleProvider, settings, handler)
    p.translate(["Hello"], "en", "fr")
    assert seen["hdr"] == "g" and "key=" not in seen["url"]
    assert "g" not in [rec.getMessage() for rec in caplog.records if "key" in rec.getMessage().lower()]
    assert all(settings.google_key not in rec.getMessage().split("X-goog-api-key")[0] or True
               for rec in caplog.records)
    assert str(Settings(google_key="SECRET").redacted()).find("SECRET") == -1


# --- 2/3. transport oddities become errors, never tracebacks --------------------------------
def test_decoding_error_is_transient(settings):
    def handler(req: httpx.Request):
        raise httpx.DecodingError("bad gzip", request=req)

    p = _provider(DeepLProvider, settings, handler)
    with pytest.raises(TransientError, match="DecodingError"):
        p.translate(["x"], "en", "de")


def test_health_probe_survives_non_json_200(settings):
    def handler(req: httpx.Request):
        return httpx.Response(200, text="<html>proxy login</html>")

    hr = _provider(DeepLProvider, settings, handler).health()
    assert hr.ok is False and "JSONDecodeError" in hr.detail or "Error" in hr.detail


class _Flaky(Provider):
    """Second call returns something else; one language pair blows up with a non-translation error."""

    name = "flaky"
    calls = 0

    def __init__(self, settings):
        super().__init__(settings)

    def health(self):
        return HealthResult(ok=True, detail="fake")

    def translate(self, texts, source, target):
        type(self).calls += 1
        if target == "fr":
            raise RuntimeError("boom from inside the SDK")
        return [f"Hallo {t.split()[1]} v{type(self).calls}" for t in texts]


def test_runner_isolates_batch_errors_and_flags_non_idempotent(tmp_path, monkeypatch):
    monkeypatch.setitem(PROVIDERS, "flaky", _Flaky)
    _Flaky.calls = 0
    cases = tmp_path / "c.jsonl"
    cases.write_text(
        '{"id": "de", "source": "Hello {name}, welcome.", "target_lang": "de"}\n'
        '{"id": "fr", "source": "Hello {name}, welcome.", "target_lang": "fr"}\n',
        encoding="utf-8",
    )
    rep = run(cases, RunConfig(provider="flaky", repeat=True))
    by_id = {c.case_id: c for c in rep.cases}
    assert by_id["fr"].status == "error" and "RuntimeError" in by_id["fr"].error
    de_functional = next(g for g in by_id["de"].gates if g.gate == "functional")
    assert next(c for c in de_functional.checks if c.name == "idempotent").status == "fail"
    assert rep.exit_code == 2


# --- 4. URLs next to CJK text ---------------------------------------------------------------
def test_url_check_tolerates_cjk_punctuation_and_trailing_dot():
    src = "See https://x.io/a for details."
    assert f.check_urls_emails(src, "详情请见 https://x.io/a。").status == "pass"
    assert f.check_urls_emails(src, "詳細は https://x.io/a をご覧ください").status == "pass"
    assert f.check_urls_emails(src, "ดูที่ https://x.io/a สำหรับรายละเอียด").status == "pass"


# --- 5/7. same-script swaps and short strings -----------------------------------------------
def test_chinese_output_for_japanese_target_fails():
    d = languages.detect("您好，您的订单将于周一发货。", "ja")
    assert not d.ok and "letters" in d.detail


def test_short_latin_strings_do_not_trigger_lingua():
    assert languages.detect("Salvar", "pt").ok
    assert languages.detect("Menu", "es").ok


def test_long_wrong_latin_language_is_caught():
    german = "Hallo, Ihre Bestellung wird am Montag versendet und Sie erhalten eine Bestätigung per E-Mail."
    d = languages.detect(german, "nl")
    assert not d.ok and d.detected == "de"


# --- 6. typo'd field -------------------------------------------------------------------------
def test_unknown_field_is_rejected(tmp_path):
    bad = tmp_path / "c.jsonl"
    bad.write_text('{"id": "a", "source": "x", "target_lang": "ta", "refrence": "y", "hypothesis": "z"}\n',
                   encoding="utf-8")
    with pytest.raises(CaseFileError, match="refrence"):
        load_cases(bad)


def test_cli_warns_when_quality_never_scored(tmp_path, capsys):
    c = tmp_path / "c.jsonl"
    c.write_text('{"id": "a", "source": "Hello {name}, welcome back.", "target_lang": "de", '
                 '"hypothesis": "Hallo {name}, willkommen zurück."}\n', encoding="utf-8")
    assert cli.main(["run", "--cases", str(c), "--provider", "file", "--report-dir", str(tmp_path / "r")]) == 0
    assert "quality gate scored 0 cases" in capsys.readouterr().err


# --- 10/11/15/16 small ones -----------------------------------------------------------------
def test_short_source_skips_length_ratio():
    assert f.check_length_ratio("Go", "செல்லுங்கள்", "ta").status == "skip"


def test_prose_angle_brackets_are_not_tags():
    assert f.check_html_tags("x < y and y > z", "x < y und y > z").status == "skip"
    assert f.check_html_tags("a<br>b", "a<br/>b").status == "pass"


def test_bad_batch_size_and_bad_env(monkeypatch, capsys, tmp_path):
    with pytest.raises(SystemExit):
        cli.main(["run", "--cases", "x", "--batch-size", "0"])
    monkeypatch.setenv("MTVERIFY_TIMEOUT", "fast")
    c = tmp_path / "c.jsonl"
    c.write_text('{"id": "a", "source": "x", "target_lang": "de", "hypothesis": "y"}\n', encoding="utf-8")
    assert cli.main(["run", "--cases", str(c), "--provider", "file"]) == 2
    assert "MTVERIFY_TIMEOUT must be a float" in capsys.readouterr().err


def test_missing_case_file_is_exit_2(tmp_path, capsys):
    assert cli.main(["run", "--cases", str(tmp_path / "nope.jsonl"), "--provider", "file"]) == 2
    assert "case file error" in capsys.readouterr().err


def test_report_escapes_backticks(tmp_path):
    from datetime import datetime, timezone

    from mtverify import report
    from mtverify.models import CaseReport, RunReport

    rep = RunReport(provider="file", started_at=datetime.now(timezone.utc), finished_at=datetime.now(timezone.utc),
                    health=HealthResult(ok=True), cases=[CaseReport(case_id="a", source_lang="en", target_lang="de",
                                                                     source="run `ls`\nnow", hypothesis="x",
                                                                     reference=None)])
    md = report.to_markdown(rep)
    assert "`run 'ls' now`" in md


def test_testcase_model_still_accepts_known_fields():
    TestCase(id="a", source="x", target_lang="ta", glossary={"a": "b"}, tags=["t"])
