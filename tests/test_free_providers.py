"""Free providers: local model (faked here, real run is in the README) and MyMemory (mocked HTTP)."""

import sys
import types

import httpx
import pytest

from mtverify import cli
from mtverify.config import Settings
from mtverify.patterns import protect, restore
from mtverify.providers import PROVIDERS
from mtverify.providers.base import TranslationError
from mtverify.providers.local import LocalProvider, model_family
from mtverify.providers.mymemory import MyMemoryProvider


# ---------- protect / restore ----------
def test_protect_masks_everything_that_must_not_be_translated():
    src = "Hi {name}, open <b>https://x.io/a</b> or mail kr@x.io, code %s"
    masked, saved = protect(src)
    assert "{name}" not in masked and "https://x.io/a" not in masked and "<b>" not in masked
    assert masked.count("<x") == len(saved) == 6
    assert restore(masked, saved) == src


def test_restore_tolerates_model_spacing_and_leaves_lost_ones_missing():
    _, saved = protect("Hello {name}, order {id}")
    assert restore("こんにちは < x0 / >、注文", saved) == "こんにちは {name}、注文"
    assert "{id}" not in restore("こんにちは <x0/>", saved)
    assert restore("no masks", []) == "no masks"


# ---------- local provider with a fake transformers ----------
class _Batch(dict):
    def to(self, device):
        return self


class _Tensor:
    def __init__(self, n):
        self.shape = (1, n)

    def to(self, device):
        return self


class FakeTok:
    unk_token_id = 3
    src_lang = None

    @classmethod
    def from_pretrained(cls, mid):
        return cls()

    def get_lang_id(self, code):
        if code == "xx":
            raise KeyError(code)
        return 100

    def convert_tokens_to_ids(self, code):
        return 200

    def __call__(self, texts, **kw):
        self.last = list(texts)
        return {"input_ids": _Tensor(8), "attention_mask": _Tensor(8)}

    def batch_decode(self, out, skip_special_tokens=True):
        return [f"JA[{t}]" for t in self.last]


class FakeModel:
    @classmethod
    def from_pretrained(cls, mid, **kw):
        FakeModel.load_kw = kw
        return cls()

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def generate(self, **kw):
        FakeModel.kw = kw
        return object()


@pytest.fixture
def fake_hf(monkeypatch):
    tr = types.ModuleType("transformers")
    tr.AutoTokenizer, tr.AutoModelForSeq2SeqLM = FakeTok, FakeModel
    torch = types.ModuleType("torch")

    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    torch.inference_mode = _NoGrad
    torch.bfloat16, torch.float32 = "bf16", "fp32"
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "transformers", tr)
    monkeypatch.setitem(sys.modules, "torch", torch)


def test_local_m2m100_masks_and_restores(fake_hf):
    p = LocalProvider(Settings(local_model="facebook/m2m100_418M", device="auto"))
    out = p.translate(["Hello {name}, see https://x.io"], "en", "ja")
    assert out == ["JA[Hello {name}, see https://x.io]"]
    assert p._tok.last == ["Hello <x0/>, see <x1/>"]  # the model never saw the placeholder
    assert FakeModel.kw["forced_bos_token_id"] == 100 and p._device == "cpu"
    assert p.health().ok


def test_local_nllb_and_madlad_codes(fake_hf):
    n = LocalProvider(Settings(local_model="facebook/nllb-200-distilled-600M"))
    n.translate(["Hi"], "en", "ja")
    assert n._tok.src_lang == "eng_Latn" and FakeModel.kw["forced_bos_token_id"] == 200
    m = LocalProvider(Settings(local_model="google/madlad400-3b-mt"))
    m.translate(["Hi"], "en", "ja")
    assert m._tok.last == ["<2ja> Hi"] and "forced_bos_token_id" not in FakeModel.kw
    assert FakeModel.load_kw["dtype"] == "bf16"


def test_local_errors_are_clear(fake_hf, monkeypatch):
    with pytest.raises(TranslationError, match="unsupported model"):
        model_family("gpt2")
    p = LocalProvider(Settings(local_model="facebook/m2m100_418M"))
    with pytest.raises(TranslationError, match="no language 'xx'"):
        p.translate(["Hi"], "en", "xx")
    monkeypatch.setitem(sys.modules, "transformers", None)
    q = LocalProvider(Settings(local_model="facebook/m2m100_418M"))
    with pytest.raises(TranslationError, match="pip install mtverify\\[local\\]"):
        q.translate(["Hi"], "en", "ja")


# ---------- MyMemory ----------
def _mm(handler):
    return MyMemoryProvider(Settings(retries=1), transport=httpx.MockTransport(handler), sleep=lambda s: None)


def test_mymemory_masks_and_parses():
    seen = {}

    def handler(req):
        seen["q"], seen["pair"] = req.url.params["q"], req.url.params["langpair"]
        return httpx.Response(200, json={"responseStatus": 200, "responseData": {"translatedText": "こんにちは <x0/>"}})

    assert _mm(handler).translate(["Hello {name}"], "en", "ja") == ["こんにちは {name}"]
    assert seen == {"q": "Hello <x0/>", "pair": "en|ja"}


def test_mymemory_quota_message_is_an_error():
    def handler(req):
        return httpx.Response(200, json={"responseStatus": 429, "responseDetails": "MYMEMORY WARNING: YOU USED ALL"})

    with pytest.raises(TranslationError, match="YOU USED ALL"):
        _mm(handler).translate(["x"], "en", "ja")


# ---------- translate command ----------
class _FakeJa:
    name = "fakeja"
    max_batch = 10

    def __init__(self, settings):
        pass

    def health(self):
        from mtverify.models import HealthResult

        return HealthResult(ok=True, detail="fake")

    def translate(self, texts, s, t):
        table = {"water": "水", "Hello {name}": "こんにちは"}
        return [table[x] for x in texts]

    def close(self):
        pass


def test_translate_command_runs_gates(monkeypatch, tmp_path, capsys):
    monkeypatch.setitem(PROVIDERS, "fakeja", _FakeJa)
    code = cli.main(["translate", "--to", "ja", "--provider", "fakeja", "--no-labse",
                     "--report-dir", str(tmp_path), "water", "Hello {name}"])
    out = capsys.readouterr().out
    assert code == 1
    assert "PASS" in out and "水" in out
    assert "FAIL" in out and "missing {name}" in out


def test_translate_command_input_errors(capsys):
    assert cli.main(["translate", "--to", "ja"]) == 2
    assert cli.main(["translate", "--to", "xx", "hi"]) == 2
    assert "input error" in capsys.readouterr().err


def test_japanese_single_kanji_word_is_valid():
    from mtverify import languages

    assert languages.detect("水", "ja").ok
    assert languages.detect("東京", "ja").ok


def test_default_local_model_is_business_safe_and_covers_telugu(monkeypatch):
    monkeypatch.delenv("MTVERIFY_LOCAL_MODEL", raising=False)
    assert Settings().local_model == "google/madlad400-3b-mt"
    assert model_family(Settings().local_model) == "madlad"
