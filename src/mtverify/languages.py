"""Language registry and target-language detection.

Detection is two-layered on purpose:
  1. Unicode script check: cheap, offline, covers every language here (lingua lacks kn, ml).
  2. lingua statistical model: separates languages that share a script (hi/mr, ur/ar/fa, zh/ja).
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from functools import lru_cache

from lingua import Language, LanguageDetectorBuilder

from mtverify.patterns import strip_non_language

Range = tuple[int, int]

_LATIN: tuple[Range, ...] = ((0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F), (0x1E00, 0x1EFF))
_ARABIC: tuple[Range, ...] = ((0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF))
_DEVANAGARI: tuple[Range, ...] = ((0x0900, 0x097F),)
_CJK: tuple[Range, ...] = ((0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x3000, 0x303F))
_KANA: tuple[Range, ...] = ((0x3040, 0x309F), (0x30A0, 0x30FF), (0xFF66, 0xFF9F))
_HANGUL: tuple[Range, ...] = ((0xAC00, 0xD7AF), (0x1100, 0x11FF), (0x3130, 0x318F))
_CYRILLIC: tuple[Range, ...] = ((0x0400, 0x04FF), (0x0500, 0x052F))


@dataclass(frozen=True)
class Lang:
    code: str
    name: str
    script: str
    ranges: tuple[Range, ...]
    lingua: str | None  # lingua Language enum name, or None when lingua cannot detect it
    bleu_tokenize: str = "13a"  # sacrebleu tokenizer; "char" for scripts without spaces
    length_ratio: tuple[float, float] = (0.3, 3.5)  # allowed len(hyp)/len(src)


REGISTRY: dict[str, Lang] = {
    lang.code: lang
    for lang in [
        Lang("en", "English", "Latin", _LATIN, "ENGLISH"),
        Lang("ta", "Tamil", "Tamil", ((0x0B80, 0x0BFF),), "TAMIL"),
        Lang("te", "Telugu", "Telugu", ((0x0C00, 0x0C7F),), "TELUGU"),
        Lang("kn", "Kannada", "Kannada", ((0x0C80, 0x0CFF),), None),
        Lang("ml", "Malayalam", "Malayalam", ((0x0D00, 0x0D7F),), None),
        Lang("hi", "Hindi", "Devanagari", _DEVANAGARI, "HINDI"),
        Lang("mr", "Marathi", "Devanagari", _DEVANAGARI, "MARATHI"),
        Lang("gu", "Gujarati", "Gujarati", ((0x0A80, 0x0AFF),), "GUJARATI"),
        Lang("bn", "Bengali", "Bengali", ((0x0980, 0x09FF),), "BENGALI"),
        Lang("pa", "Punjabi", "Gurmukhi", ((0x0A00, 0x0A7F),), "PUNJABI"),
        Lang("ur", "Urdu", "Arabic", _ARABIC, "URDU"),
        Lang("ar", "Arabic", "Arabic", _ARABIC, "ARABIC"),
        Lang("fa", "Persian", "Arabic", _ARABIC, "PERSIAN"),
        Lang("zh", "Chinese", "Han", _CJK, "CHINESE", "zh", (0.15, 2.0)),
        Lang("ja", "Japanese", "Kana+Han", _KANA + _CJK, "JAPANESE", "char", (0.2, 2.5)),
        Lang("ko", "Korean", "Hangul", _HANGUL, "KOREAN", "char", (0.2, 2.5)),
        Lang("th", "Thai", "Thai", ((0x0E00, 0x0E7F),), "THAI", "char", (0.3, 3.5)),
        Lang("vi", "Vietnamese", "Latin", _LATIN, "VIETNAMESE"),
        Lang("id", "Indonesian", "Latin", _LATIN, "INDONESIAN"),
        Lang("de", "German", "Latin", _LATIN, "GERMAN"),
        Lang("fr", "French", "Latin", _LATIN, "FRENCH"),
        Lang("es", "Spanish", "Latin", _LATIN, "SPANISH"),
        Lang("pt", "Portuguese", "Latin", _LATIN, "PORTUGUESE"),
        Lang("it", "Italian", "Latin", _LATIN, "ITALIAN"),
        Lang("ru", "Russian", "Cyrillic", _CYRILLIC, "RUSSIAN"),
        Lang("tr", "Turkish", "Latin", _LATIN, "TURKISH"),
        Lang("nl", "Dutch", "Latin", _LATIN, "DUTCH"),
        Lang("pl", "Polish", "Latin", _LATIN, "POLISH"),
    ]
}

SCRIPT_RATIO_MIN = 0.6  # share of letters that must sit in the expected script
LINGUA_CONFIDENCE_MIN = 0.6  # lingua must be this sure before it overrides the script check


class UnknownLanguage(ValueError):
    pass


def get(code: str) -> Lang:
    try:
        return REGISTRY[code.lower()]
    except KeyError as e:
        raise UnknownLanguage(f"language {code!r} not in registry ({', '.join(REGISTRY)})") from e


@dataclass(frozen=True)
class Detection:
    ok: bool
    expected: str
    detected: str | None
    method: str
    script_ratio: float
    detail: str


def _letters(text: str) -> list[str]:
    return [ch for ch in text if unicodedata.category(ch).startswith("L")]


def _in_ranges(ch: str, ranges: tuple[Range, ...]) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in ranges)


def script_ratio(text: str, lang: Lang) -> float:
    letters = _letters(text)
    if not letters:
        return 0.0
    hits = sum(1 for ch in letters if _in_ranges(ch, lang.ranges))
    return hits / len(letters)


@lru_cache(maxsize=1)
def _detector():
    langs = [getattr(Language, lg.lingua) for lg in REGISTRY.values() if lg.lingua]
    return LanguageDetectorBuilder.from_languages(*langs).with_preloaded_language_models().build()


def _lingua_guess(text: str) -> tuple[str | None, float]:
    values = _detector().compute_language_confidence_values(text)
    if not values:
        return None, 0.0
    top = values[0]
    by_name = {lg.lingua: lg.code for lg in REGISTRY.values() if lg.lingua}
    return by_name.get(top.language.name), float(top.value)


def detect(text: str, expected_code: str) -> Detection:
    """Decide whether `text` is written in `expected_code`.

    Script check first. lingua is consulted only when it knows the expected language and is
    confident, so a same-script neighbour (Marathi for Hindi) is caught without false alarms
    on short strings.
    """
    lang = get(expected_code)
    text = strip_non_language(text)
    ratio = script_ratio(text, lang)
    if not _letters(text):
        return Detection(True, lang.code, None, "no-letters", 0.0, "no letters to judge")
    if ratio < SCRIPT_RATIO_MIN:
        return Detection(
            False, lang.code, None, "script", ratio,
            f"only {ratio:.0%} of letters are {lang.script} script",
        )
    if lang.lingua is None:
        return Detection(True, lang.code, lang.code, "script", ratio, f"{ratio:.0%} {lang.script}")
    guess, conf = _lingua_guess(text)
    if guess and guess != lang.code and conf >= LINGUA_CONFIDENCE_MIN:
        siblings_same_script = get(guess).ranges == lang.ranges
        if siblings_same_script:
            return Detection(
                False, lang.code, guess, "lingua", ratio,
                f"script ok but lingua says {guess} ({conf:.2f})",
            )
    return Detection(True, lang.code, guess or lang.code, "script+lingua", ratio, f"{ratio:.0%} {lang.script}")
