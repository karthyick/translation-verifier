"""Regexes shared by the functional gate and the language detector. No project imports here."""

from __future__ import annotations

import re

# Placeholder styles seen in real products: {name} {{name}} ${name} %s %d %(name)s %1$s :name
PLACEHOLDER = re.compile(
    r"\{\{[^{}]+\}\}|\{[^{}\s]+\}|\$\{[^}]+\}|%\([A-Za-z_][\w]*\)[sdif]|%\d+\$[sd]|%[sdif]|(?<!\w):[A-Za-z_]\w*\b"
)
HTML_TAG = re.compile(r"<(/?)([A-Za-z][\w-]*)(?:\s[^<>]*?)?\s*(/?)>")
URL = re.compile("(?:https?://|www\\.)[^\\s<>\"'\u0e00-\u0e7f\u3000-\u9fff\uff00-\uffef]+")
_URL_TRAIL = ".,;:!?)]}'\""
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# digits with any common group/decimal separator, including Arabic ones and the spaces fr/de use
NUMBER = re.compile("\\d+(?:[.,:\u066b\u066c\u00a0\u202f' ]\\d+)*")
NUMBER_SEPARATORS = re.compile("[.,:\u066b\u066c\u00a0\u202f' ]")
BAD_CHARS = re.compile("[\ufffd\x00-\x08\x0b\x0c\x0e-\x1f]")


def find_urls(text: str) -> list[str]:
    """URLs with trailing punctuation removed, so `see https://x.io.` yields https://x.io."""
    return [m.rstrip(_URL_TRAIL) for m in URL.findall(text)]


def strip_non_language(text: str) -> str:
    """Remove the parts of a string that carry no language: placeholders, tags, URLs, emails."""
    for rx in (URL, EMAIL, HTML_TAG, PLACEHOLDER):
        text = rx.sub(" ", text)
    return text
