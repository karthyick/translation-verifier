"""Gate 3: does the translation mean the same thing.

Always on:   chrF2 and BLEU via sacrebleu (needs a reference).
Optional:    COMET (pip install mtverify[comet]); reference-free with a CometKiwi model.
Optional:    LaBSE cosine (pip install mtverify[labse]); reference-free, source vs output.
Missing optional packages produce a 'skip', never an error.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import sacrebleu

from mtverify import languages
from mtverify.models import CheckResult, GateResult, TestCase

log = logging.getLogger("mtverify.quality")


@dataclass(frozen=True)
class QualityConfig:
    chrf_min: float = 50.0
    bleu_min: float | None = None  # informational by default
    comet_model: str | None = None  # e.g. Unbabel/wmt22-comet-da or Unbabel/wmt22-cometkiwi-da
    comet_min: float = 0.5
    labse: bool = False
    labse_min: float = 0.75
    batch_size: int = 16
    gpus: int = 0


class QualityScorer:
    def __init__(self, cfg: QualityConfig):
        self.cfg = cfg
        self._comet = None
        self._labse = None

    # ---------- reference metrics ----------
    def _chrf_bleu(self, case: TestCase, hyp: str) -> list[CheckResult]:
        if not case.reference:
            return [CheckResult(name="chrf", status="skip", detail="no reference"),
                    CheckResult(name="bleu", status="skip", detail="no reference")]
        lang = languages.get(case.target_lang)
        chrf = sacrebleu.sentence_chrf(hyp, [case.reference]).score
        bleu = sacrebleu.sentence_bleu(hyp, [case.reference], tokenize=lang.bleu_tokenize).score
        out = [
            CheckResult(
                name="chrf", status="pass" if chrf >= self.cfg.chrf_min else "fail",
                detail=f"chrF2 {chrf:.1f}", value=round(chrf, 2), threshold=self.cfg.chrf_min,
            )
        ]
        if self.cfg.bleu_min is None:
            out.append(CheckResult(name="bleu", status="pass", detail=f"BLEU {bleu:.1f} (info)",
                                   value=round(bleu, 2)))
        else:
            out.append(CheckResult(
                name="bleu", status="pass" if bleu >= self.cfg.bleu_min else "fail",
                detail=f"BLEU {bleu:.1f}", value=round(bleu, 2), threshold=self.cfg.bleu_min,
            ))
        return out

    # ---------- COMET ----------
    def _comet_scores(self, cases: list[TestCase], hyps: list[str]) -> list[CheckResult] | None:
        if not self.cfg.comet_model:
            return None
        try:
            from comet import download_model, load_from_checkpoint
        except ImportError:
            return [CheckResult(name="comet", status="skip",
                                detail="unbabel-comet not installed (pip install mtverify[comet])")] * len(cases)
        try:
            if self._comet is None:
                self._comet = load_from_checkpoint(download_model(self.cfg.comet_model))
            needs_ref = "kiwi" not in self.cfg.comet_model.lower()
            data = []
            for c, h in zip(cases, hyps, strict=True):
                row = {"src": c.source, "mt": h}
                if needs_ref:
                    row["ref"] = c.reference or ""
                data.append(row)
            out = self._comet.predict(data, batch_size=self.cfg.batch_size, gpus=self.cfg.gpus,
                                      progress_bar=False)
            scores = list(out.scores)
        except Exception as e:  # model download, CUDA, gated repo ...
            log.warning("comet failed: %s", e)
            return [CheckResult(name="comet", status="error", detail=f"{type(e).__name__}: {e}")] * len(cases)
        res = []
        for c, s in zip(cases, scores, strict=True):
            if needs_ref and not c.reference:
                res.append(CheckResult(name="comet", status="skip", detail="no reference for ref model"))
                continue
            res.append(CheckResult(
                name="comet", status="pass" if s >= self.cfg.comet_min else "fail",
                detail=f"{self.cfg.comet_model.split('/')[-1]} {s:.3f}", value=round(float(s), 4),
                threshold=self.cfg.comet_min,
            ))
        return res

    # ---------- LaBSE ----------
    def _labse_scores(self, cases: list[TestCase], hyps: list[str]) -> list[CheckResult] | None:
        if not self.cfg.labse:
            return None
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            return [CheckResult(name="labse", status="skip",
                                detail="sentence-transformers missing (pip install mtverify[labse])")] * len(cases)
        try:
            if self._labse is None:
                self._labse = SentenceTransformer("sentence-transformers/LaBSE")
            a = self._labse.encode([c.source for c in cases], normalize_embeddings=True,
                                   batch_size=self.cfg.batch_size)
            b = self._labse.encode(hyps, normalize_embeddings=True, batch_size=self.cfg.batch_size)
            sims = (a * b).sum(axis=1)
        except Exception as e:
            log.warning("labse failed: %s", e)
            return [CheckResult(name="labse", status="error", detail=f"{type(e).__name__}: {e}")] * len(cases)
        return [
            CheckResult(name="labse", status="pass" if s >= self.cfg.labse_min else "fail",
                        detail=f"cosine {s:.3f}", value=round(float(s), 4), threshold=self.cfg.labse_min)
            for s in sims
        ]

    # ---------- entry ----------
    def score(self, cases: list[TestCase], hyps: list[str]) -> list[GateResult]:
        """Score all cases at once so neural models batch properly."""
        per_case = [self._chrf_bleu(c, h) for c, h in zip(cases, hyps, strict=True)]
        for extra in (self._comet_scores(cases, hyps), self._labse_scores(cases, hyps)):
            if extra:
                for checks, e in zip(per_case, extra, strict=True):
                    checks.append(e)
        return [GateResult.from_checks("quality", checks) for checks in per_case]
