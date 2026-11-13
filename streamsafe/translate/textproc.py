"""Plain text helpers: what is worth translating, a cache key, sentence splitting. No models, no I/O."""

from __future__ import annotations

import re
import unicodedata

_SPACES = re.compile(r"\s+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")
MIN_LETTERS = 3


def clean(text: str) -> str:
    """One line of OCR text with the usual noise removed: control characters, runs of spaces."""
    text = "".join(c for c in unicodedata.normalize("NFKC", text) if c == "\n" or unicodedata.category(c)[0] != "C")
    return _SPACES.sub(" ", text).strip()


def cache_key(text: str) -> str:
    """Normalized text used to find a translation again: case, spacing and trailing punctuation do not matter, so
    a line that repeats (games repeat lines) or that OCR reads a little differently each time is one entry."""
    t = clean(text).casefold()
    t = re.sub(r"[^\w\s']", " ", t)
    return _SPACES.sub(" ", t).strip()


def worth_translating(text: str) -> bool:
    """Real English text: at least a few letters, mostly Latin letters / digits / punctuation (a HUD icon read as
    noise, a number, or a name of a place in another script is not sent to the translator)."""
    t = clean(text)
    letters = [c for c in t if c.isalpha()]
    if len(letters) < MIN_LETTERS:
        return False
    latin = sum(1 for c in letters if c.isascii())
    return latin >= 0.9 * len(letters) and len(letters) >= 0.5 * len(t.replace(" ", ""))


def sentences(text: str) -> list[str]:
    """Sentence-sized pieces: the translation model drops words from long inputs, so a paragraph is split."""
    parts = [p.strip() for p in _SENTENCE_END.split(clean(text)) if p.strip()]
    return parts or ([clean(text)] if clean(text) else [])
