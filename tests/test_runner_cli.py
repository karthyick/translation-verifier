import json
from importlib import resources

import pytest

from mtverify import cli, report
from mtverify.runner import CaseFileError, RunConfig, load_cases, run

SMOKE = resources.files("mtverify").joinpath("data/smoke.jsonl")


def test_smoke_set_covers_20_plus_languages_and_flags_negatives(tmp_path):
    with resources.as_file(SMOKE) as p:
        rep = run(p, RunConfig(provider="file"))
    assert rep.health.ok
    assert rep.summary["languages"] >= 20
    by_id = {c.case_id: c for c in rep.cases}
    for cid in ("ta-bad-placeholder", "hi-bad-language", "de-bad-tag", "ja-bad-number", "fr-bad-glossary"):
        assert by_id[cid].status == "fail", cid
    good = [c for c in rep.cases if c.case_id.endswith("-good")]
    assert good and all(c.status == "pass" for c in good), [c.case_id for c in good if c.status != "pass"]
    assert rep.exit_code == 1
    j, m = report.write(rep, tmp_path)
    data = json.loads(j.read_text(encoding="utf-8"))
    assert data["exit_code"] == 1 and len(data["cases"]) == len(rep.cases)
    assert "ta-bad-placeholder" in m.read_text(encoding="utf-8")


def test_load_cases_rejects_bad_line(tmp_path):
    bad = tmp_path / "c.jsonl"
    bad.write_text('{"id": "a", "source": "x", "target_lang": "ta", "hypothesis": "y"}\nnot json\n',
                   encoding="utf-8")
    with pytest.raises(CaseFileError, match=":2:"):
        load_cases(bad)


def test_load_cases_rejects_duplicate_and_unknown_lang(tmp_path):
    f = tmp_path / "c.jsonl"
    f.write_text('{"id": "a", "source": "x", "target_lang": "ta"}\n{"id": "a", "source": "x", "target_lang": "ta"}\n',
                 encoding="utf-8")
    with pytest.raises(CaseFileError, match="duplicate"):
        load_cases(f)
    f.write_text('{"id": "a", "source": "x", "target_lang": "xx"}\n', encoding="utf-8")
    from mtverify.languages import UnknownLanguage

    with pytest.raises(UnknownLanguage):
        load_cases(f)


def test_file_provider_needs_hypothesis(tmp_path):
    f = tmp_path / "c.jsonl"
    f.write_text('{"id": "a", "source": "Hello", "target_lang": "ta"}\n', encoding="utf-8")
    from mtverify.providers.base import TranslationError

    with pytest.raises(TranslationError, match="hypothesis"):
        run(f, RunConfig(provider="file"))


def test_cli_run_and_exit_codes(tmp_path, capsys):
    out = tmp_path / "cases.jsonl"
    assert cli.main(["init-cases", "--out", str(out)]) == 0
    code = cli.main(["run", "--cases", str(out), "--provider", "file", "--report-dir", str(tmp_path / "r")])
    assert code == 1
    printed = capsys.readouterr().out
    assert "languages 27" in printed and "fail 5" in printed
    assert (tmp_path / "r" / "latest.md").exists()


def test_cli_languages_lists_registry(capsys):
    assert cli.main(["languages"]) == 0
    assert "28 languages" in capsys.readouterr().out


def test_cli_bad_case_file_is_exit_2(tmp_path, capsys):
    f = tmp_path / "x.jsonl"
    f.write_text("{bad", encoding="utf-8")
    assert cli.main(["run", "--cases", str(f), "--provider", "file"]) == 2
    assert "case file error" in capsys.readouterr().err
