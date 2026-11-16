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


# What is copied, never translated: addresses, @handles, #hashtags, player names (a digit or an underscore inside, or
# CamelCase) and bracketed button names. A translator that respells one of them ("dadbodsquad" -> "دادبودسكواد") is wrong.
_KEEP = re.compile(
    r"""(?:https?://|www\.)[^\s<>]+[^\s<>.,!?;:)\]]            # address
      | [\w.+-]+@[\w-]+(?:\.[\w-]+)+                            # e-mail
      | \b[\w-]+\.(?:com|net|org|tv|gg|io|me|co|live|app)\b(?:/[^\s<>]*[^\s<>.,!?;:)\]])?
      | (?<![\w@#])[@#][A-Za-z_][\w.]*\w                         # @handle, #hashtag
      | \[[A-Z0-9][A-Za-z0-9 ]{0,7}\]                             # [RT], [SQUAD]
      | (?<![\w@#])(?=[A-Za-z0-9_-]*[A-Za-z]{2})(?:[A-Za-z]+[\d_][\w-]*|[\d_]+[A-Za-z_]\w*[\d_]\w*|_\w+)(?![\w])   # k1ller_queen, x_y
      | (?<![\w@#])[A-Za-z]+-\d+\b                               # Rook-7
      | (?<![\w@#])(?:[A-Z]+[a-z]+){2,}[A-Za-z]*!?(?![\w])        # NoobMaster, WhatUp!
      | (?<![\w@#])[a-z]+[A-Z][A-Za-z]*(?![\w])                   # xDark
    """,
    re.X,
)
_UNIT = re.compile(r"^\d+(?:st|nd|rd|th|x|k|m|s|ms|kg|km|mph|kph|fps|hp|xp|d|p)$|^x\d+$", re.I)
_VOWEL = re.compile(r"[aeiouy]", re.I)
_SOUND = re.compile(r"^(?:sh+|hm+|mm+|br+|gr+|ps+t|pf+t|tsk|zz+)$", re.I)  # "Shhh!", "Hmm."
_SAME4 = re.compile(r"([A-Za-z])\1{3,}")
_WORD = re.compile(r"[A-Za-z]+")
_OK_SIGNS = frozenset(" .,!?;:'\"-()[]<>/%$#@&+*_=~")


def keep_tokens(text: str) -> list[str]:
    """The pieces of ``text`` that must reach the viewer exactly as written (see ``_KEEP``), in order."""
    return [m.group(0) for m in _KEEP.finditer(text) if not _UNIT.match(m.group(0))]


def sub_keep(text: str, fn) -> str:
    """``text`` with every kept piece replaced by ``fn(piece)``."""
    return _KEEP.sub(lambda m: m.group(0) if _UNIT.match(m.group(0)) else fn(m.group(0)), text)


def only_names(text: str) -> bool:
    """Nothing to translate: the line is handles / hashtags / addresses / player names and signs ("@dadbodsquad")."""
    if not keep_tokens(text):
        return False
    return not _WORD.search(_KEEP.sub(" ", text))


def garbage(text: str) -> bool:
    """What a text reader makes of a picture that is not text: mostly single letters, words without a vowel, one
    letter many times, odd signs. English game text is none of these."""
    t = _KEEP.sub(" ", clean(text))
    words = _WORD.findall(t)
    if not words:
        return not keep_tokens(text)
    if _SAME4.search(t):
        return True
    if len(words) >= 3 and sum(1 for w in words if len(w) == 1) > 0.7 * len(words):  # not "Press W, A, S, D to move"
        return True
    if sum(1 for w in words if len(w) >= 4 and not _VOWEL.search(w) and not _SOUND.match(w)) > 0.5 * len(words):
        return True
    solid = t.replace(" ", "")
    return sum(1 for c in solid if not c.isalnum() and c not in _OK_SIGNS) > 0.25 * len(solid)


def worth_translating(text: str) -> bool:
    """Real English text: at least a few letters, mostly Latin letters / digits / punctuation (a HUD icon read as
    noise, a number, or a name of a place in another script is not sent to the translator), not reading noise
    (``garbage``) and not a bare name or address (``only_names``)."""
    t = clean(text)
    letters = [c for c in t if c.isalpha()]
    if len(letters) < MIN_LETTERS:
        return False
    latin = sum(1 for c in letters if c.isascii())
    if not (latin >= 0.9 * len(letters) and len(letters) >= 0.5 * len(t.replace(" ", ""))):
        return False
    return not garbage(t) and not only_names(t)


def sentences(text: str) -> list[str]:
    """Sentence-sized pieces: the translation model drops words from long inputs, so a paragraph is split."""
    parts = [p.strip() for p in _SENTENCE_END.split(clean(text)) if p.strip()]
    return parts or ([clean(text)] if clean(text) else [])
