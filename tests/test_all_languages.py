"""Every language in the smoke set: the good translation passes, and four kinds of damage are caught.

The damage is generated from the good translation itself, so each language is tested on its own script.
"""

import json
from importlib import resources

import pytest

from mtverify import languages
from mtverify.gates import QualityConfig, QualityScorer, functional_gate
from mtverify.models import TestCase

_SMOKE = resources.files("mtverify").joinpath("data/smoke.jsonl")


def _good_cases() -> list[TestCase]:
    out = []
    for line in _SMOKE.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            c = TestCase.model_validate(json.loads(line))
            if c.id.endswith("-good"):
                out.append(c)
    return out


GOOD = _good_cases()
IDS = [c.target_lang for c in GOOD]


def _failed(gate, name):
    return next(ch for ch in gate.checks if ch.name == name).status == "fail"


def test_smoke_set_covers_every_registry_language_except_source():
    assert sorted(IDS) == sorted(code for code in languages.REGISTRY if code != "en")
    assert len(IDS) >= 20


@pytest.mark.parametrize("case", GOOD, ids=IDS)
def test_good_translation_passes_gate2_and_gate3(case):
    assert functional_gate(case, case.hypothesis).status == "pass"
    q = QualityScorer(QualityConfig()).score([case], [case.hypothesis])[0]
    assert q.status == "pass", q


@pytest.mark.parametrize("case", GOOD, ids=IDS)
def test_dropped_placeholder_is_caught(case):
    broken = case.hypothesis.replace("{name}", "NAME")
    assert _failed(functional_gate(case, broken), "placeholders")


@pytest.mark.parametrize("case", GOOD, ids=IDS)
def test_changed_number_is_caught(case):
    broken = case.hypothesis.replace("123", "132")
    assert broken != case.hypothesis
    assert _failed(functional_gate(case, broken), "numbers")


@pytest.mark.parametrize("case", GOOD, ids=IDS)
def test_untranslated_english_is_caught(case):
    gate = functional_gate(case, case.source)
    assert _failed(gate, "untranslated")
    assert _failed(gate, "language")


@pytest.mark.parametrize("case", GOOD, ids=IDS)
def test_empty_output_is_caught(case):
    assert _failed(functional_gate(case, ""), "not_empty")


@pytest.mark.parametrize("case", GOOD, ids=IDS)
def test_wrong_target_language_is_caught(case):
    """Feed another language's good output. Different script must always fail the language check."""
    other = next(o for o in GOOD if languages.get(o.target_lang).script != languages.get(case.target_lang).script)
    assert _failed(functional_gate(case, other.hypothesis), "language")
