"""Offline translation with an open-source model from Hugging Face. No key, no bill, no network after download.

Model families supported (pick with MTVERIFY_LOCAL_MODEL):

  google/madlad400-3b-mt (default) Apache-2.0   business use ok    ~12 GB, all 27 languages
  facebook/m2m100_418M             MIT          business use ok    ~2 GB, fast, weak on Indic, no Telugu
  facebook/m2m100_1.2B             MIT          business use ok    ~5 GB, no Telugu
  facebook/nllb-200-distilled-600M CC-BY-NC     NON-COMMERCIAL     ~2.5 GB

Install: pip install mtverify[local]
"""

from __future__ import annotations

import logging

from mtverify.models import HealthResult
from mtverify.patterns import protect, restore
from mtverify.providers.base import Provider, TranslationError

log = logging.getLogger("mtverify.providers.local")

# NLLB uses FLORES-200 codes instead of ISO 639-1.
NLLB_CODES = {
    "en": "eng_Latn", "ta": "tam_Taml", "te": "tel_Telu", "kn": "kan_Knda", "ml": "mal_Mlym",
    "hi": "hin_Deva", "mr": "mar_Deva", "gu": "guj_Gujr", "bn": "ben_Beng", "pa": "pan_Guru",
    "ur": "urd_Arab", "ar": "arb_Arab", "fa": "pes_Arab", "zh": "zho_Hans", "ja": "jpn_Jpan",
    "ko": "kor_Hang", "th": "tha_Thai", "vi": "vie_Latn", "id": "ind_Latn", "de": "deu_Latn",
    "fr": "fra_Latn", "es": "spa_Latn", "pt": "por_Latn", "it": "ita_Latn", "ru": "rus_Cyrl",
    "tr": "tur_Latn", "nl": "nld_Latn", "pl": "pol_Latn",
}


def model_family(model_id: str) -> str:
    m = model_id.lower()
    if "nllb" in m:
        return "nllb"
    if "madlad" in m:
        return "madlad"
    if "m2m100" in m or "m2m_100" in m:
        return "m2m100"
    raise TranslationError(
        f"local: unsupported model {model_id!r}; use an m2m100, nllb-200 or madlad400 checkpoint"
    )


class LocalProvider(Provider):
    name = "local"
    max_batch = 16

    def __init__(self, settings):
        super().__init__(settings)
        self.model_id = settings.local_model
        self.family = model_family(self.model_id)
        self._tok = None
        self._model = None
        self._device = None
        self._torch = None

    # ---------- loading ----------
    def _pick_device(self, torch) -> str:
        want = (self.settings.device or "auto").lower()
        if want != "auto":
            return want
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as e:
            raise TranslationError("local: transformers/torch missing (pip install mtverify[local])") from e
        self._torch = torch
        log.info("loading %s (%s family); first run downloads it", self.model_id, self.family)
        try:
            self._tok = AutoTokenizer.from_pretrained(self.model_id)
            # 3B-class models load in bfloat16 (half the RAM); small ones stay float32 for stable output
            dtype = torch.bfloat16 if self.family == "madlad" else torch.float32
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id, dtype=dtype)
        except Exception as e:
            raise TranslationError(f"local: cannot load {self.model_id}: {type(e).__name__}: {e}") from e
        self._device = self._pick_device(torch)
        try:
            self._model.to(self._device)
        except Exception as e:  # e.g. an op missing on mps
            log.warning("device %s failed (%s); using cpu", self._device, e)
            self._device = "cpu"
            self._model.to("cpu")
        self._model.eval()
        log.info("model ready on %s", self._device)

    # ---------- language codes ----------
    def _codes(self, source: str, target: str) -> tuple[str, str]:
        if self.family == "nllb":
            try:
                return NLLB_CODES[source], NLLB_CODES[target]
            except KeyError as e:
                raise TranslationError(f"local: no NLLB code for {e.args[0]!r}") from e
        return source, target

    def _forced_bos(self, target_code: str) -> int | None:
        if self.family == "madlad":
            return None  # MADLAD takes the target as a <2xx> prefix instead
        if self.family == "m2m100":
            try:
                return self._tok.get_lang_id(target_code)
            except KeyError as e:
                raise TranslationError(f"local: {self.model_id} has no language {target_code!r}") from e
        tid = self._tok.convert_tokens_to_ids(target_code)
        if tid is None or tid == self._tok.unk_token_id:
            raise TranslationError(f"local: {self.model_id} has no language {target_code!r}")
        return tid

    # ---------- contract ----------
    def health(self) -> HealthResult:
        def probe() -> str:
            out = self.translate(["Hello"], "en", "fr")
            return f"{self.model_id} on {self._device}: 'Hello' -> {out[0]!r}"

        return self._timed(probe)

    def translate(self, texts: list[str], source: str, target: str) -> list[str]:
        self._load()
        src_code, tgt_code = self._codes(source, target)
        masked = [protect(t) for t in texts]
        inputs = [m for m, _ in masked]
        if self.family == "madlad":
            inputs = [f"<2{tgt_code}> {t}" for t in inputs]
        else:
            self._tok.src_lang = src_code
        forced = self._forced_bos(tgt_code)
        enc = self._tok(inputs, return_tensors="pt", padding=True, truncation=True, max_length=512)
        enc = {k: v.to(self._device) for k, v in enc.items()}
        longest = int(enc["input_ids"].shape[1])
        gen_kw = {"num_beams": 4, "max_new_tokens": min(512, max(32, longest * 3))}
        if forced is not None:
            gen_kw["forced_bos_token_id"] = forced
        try:
            with self._torch.inference_mode():
                out = self._model.generate(**enc, **gen_kw)
        except Exception as e:
            raise TranslationError(f"local: generation failed: {type(e).__name__}: {e}") from e
        result = self._tok.batch_decode(out, skip_special_tokens=True)
        if len(result) != len(texts):
            raise TranslationError(f"local: sent {len(texts)} texts, got {len(result)} back")
        return [restore(r.strip(), saved) for r, (_, saved) in zip(result, masked, strict=True)]
