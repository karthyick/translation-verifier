from mtverify.gates import QualityConfig, QualityScorer
from mtverify.models import TestCase


def _check(gate, name):
    return next(c for c in gate.checks if c.name == name)


def test_chrf_orders_good_above_bad(ta_case):
    good = "வணக்கம் {name}, உங்கள் ஆர்டர் #123 திங்கள் அன்று அனுப்பப்படும்."
    bad = "வணக்கம் பெயர், உங்கள் ஆர்டர் 123 செவ்வாய் அன்று அனுப்பப்படும்."
    gates = QualityScorer(QualityConfig(chrf_min=60)).score([ta_case, ta_case], [good, bad])
    g, b = _check(gates[0], "chrf"), _check(gates[1], "chrf")
    assert g.value > b.value
    assert g.status == "pass" and b.status == "fail"
    assert g.threshold == 60


def test_bleu_informational_by_default(ta_case):
    gate = QualityScorer(QualityConfig()).score([ta_case], ["தவறு"])[0]
    assert _check(gate, "bleu").status == "pass"
    gate = QualityScorer(QualityConfig(bleu_min=10)).score([ta_case], ["தவறு"])[0]
    assert _check(gate, "bleu").status == "fail"


def test_cjk_uses_char_tokenizer():
    case = TestCase(id="ja", source="Thanks", target_lang="ja", reference="ありがとうございます。")
    gate = QualityScorer(QualityConfig(chrf_min=10)).score([case], ["ありがとうございます。"])[0]
    assert _check(gate, "bleu").value == 100.0


def test_no_reference_skips_not_fails():
    case = TestCase(id="x", source="Thanks", target_lang="de")
    gate = QualityScorer(QualityConfig()).score([case], ["Danke"])[0]
    assert gate.status == "skip"


def test_optional_models_skip_when_not_installed(monkeypatch, ta_case):
    import builtins

    real_import = builtins.__import__

    def fake(name, *a, **k):
        if name in ("comet", "sentence_transformers"):
            raise ImportError(name)
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    cfg = QualityConfig(comet_model="Unbabel/wmt22-cometkiwi-da", labse=True)
    gate = QualityScorer(cfg).score([ta_case], [ta_case.reference])[0]
    assert _check(gate, "comet").status == "skip"
    assert _check(gate, "labse").status == "skip"
    assert gate.status == "pass"  # skips never fail a gate


def test_comet_runs_with_one_worker_off_windows(monkeypatch, ta_case):
    """COMET 2.2 crashes on macOS MPS with num_workers=0; we must pass a worker count."""
    import sys
    import types

    seen = {}

    class FakeModel:
        def predict(self, data, **kw):
            seen.update(kw)
            return types.SimpleNamespace(scores=[0.91] * len(data))

    fake = types.ModuleType("comet")
    fake.download_model = lambda name: "/tmp/fake.ckpt"
    fake.load_from_checkpoint = lambda path: FakeModel()
    monkeypatch.setitem(sys.modules, "comet", fake)
    gate = QualityScorer(QualityConfig(comet_model="Unbabel/wmt22-comet-da")).score(
        [ta_case], [ta_case.reference])[0]
    assert _check(gate, "comet").status == "pass" and _check(gate, "comet").value == 0.91
    assert seen["num_workers"] == (0 if sys.platform == "win32" else 1)
    assert _check(gate, "comet").threshold == 0.75
