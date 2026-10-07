"""mtverify: four-gate verification for machine translation services.

Gate 1  health      is the service alive, is the key valid
Gate 2  functional  did the translation keep placeholders, tags, numbers, language
Gate 3  quality     does the translation mean the same thing (chrF, BLEU, COMET, LaBSE)
Gate 4  monitoring  run gates 2 and 3 on a schedule; this package is the job body
"""

from mtverify.models import CaseReport, CheckResult, GateResult, RunReport, TestCase
from mtverify.runner import RunConfig, run

__all__ = [
    "CaseReport",
    "CheckResult",
    "GateResult",
    "RunConfig",
    "RunReport",
    "TestCase",
    "run",
]
__version__ = "0.1.0"
