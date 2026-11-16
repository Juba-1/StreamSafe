"""The Unicode Bidirectional Algorithm (UAX #9) for one line of text, in plain Python: logical order -> drawing order.

Written here because the usual ``python-bidi`` is LGPL and Pillow's own bidi (libraqm) is not in the Windows wheels.
What is implemented: the implicit algorithm in full: weak types W1-W7 (numbers, their separators and terminators, Arabic
numbers after Arabic letters), paired brackets N0, neutrals N1-N2, levels I1-I2, trailing spaces L1, reordering L2 and
mirroring L4, with the marks LRM / RLM / ALM as strong characters. What is not: explicit embeddings, overrides and
isolates (LRE, RLO, LRI, ...): game text has none; they are dropped like every other format character.

The paragraph direction is the caller's (``base``); ``base_of`` says right-to-left as soon as the text has one
Arabic or Hebrew letter: the overlay is an Arabic line even when it starts with a Latin name.
"""

from __future__ import annotations

import unicodedata

LRM, RLM, ALM = "\u200e", "\u200f", "\u061c"
_MARKS = {LRM, RLM, ALM}
_BRACKETS = {"(": ")", "[": "]", "{": "}", "\u2329": "\u232a", "\u3008": "\u3009", "\u300a": "\u300b", "\u300c": "\u300d"}
_CLOSERS = {v: k for k, v in _BRACKETS.items()}
_MIRROR = str.maketrans(
    "()[]{}<>\u00ab\u00bb\u2039\u203a\u2264\u2265\u2329\u232a\u3008\u3009\u300a\u300b\u300c\u300d",
    ")(][}{><\u00bb\u00ab\u203a\u2039\u2265\u2264\u232a\u2329\u3009\u3008\u300b\u300a\u300d\u300c",
)
_EXPLICIT = {"LRE", "RLE", "LRO", "RLO", "PDF", "LRI", "RLI", "FSI", "PDI", "BN"}
_NI = {"B", "S", "WS", "ON"}


def base_of(text: str) -> str:
    """'R' when the text has a right-to-left letter, else 'L'."""
    return "R" if any(unicodedata.bidirectional(c) in ("R", "AL") for c in text) else "L"


def _resolve(chars: list[str], e: int) -> list[int]:
    """Embedding level of every character of ``chars`` (no format characters among them) at paragraph level ``e``."""
    n = len(chars)
    edge = "R" if e else "L"  # sos and eos: one run at the paragraph level
    t = [unicodedata.bidirectional(c) or "ON" for c in chars]
    nsm = [x == "NSM" for x in t]
    # W1 nonspacing marks take the type before them
    for i in range(n):
        if t[i] == "NSM":
            t[i] = t[i - 1] if i else edge
    # W2 European numbers after an Arabic letter are Arabic numbers; W3 AL -> R
    strong = edge
    for i in range(n):
        if t[i] in ("L", "R", "AL"):
            strong = t[i]
        elif t[i] == "EN" and strong == "AL":
            t[i] = "AN"
    t = ["R" if x == "AL" else x for x in t]
    # W4 one separator between two numbers of one kind
    for i in range(1, n - 1):
        if t[i] == "ES" and t[i - 1] == "EN" and t[i + 1] == "EN":
            t[i] = "EN"
        elif t[i] == "CS" and t[i - 1] == t[i + 1] and t[i - 1] in ("EN", "AN"):
            t[i] = t[i - 1]
    # W5 terminators next to a European number
    i = 0
    while i < n:
        if t[i] == "ET":
            j = i
            while j < n and t[j] == "ET":
                j += 1
            if (i and t[i - 1] == "EN") or (j < n and t[j] == "EN"):
                t[i:j] = ["EN"] * (j - i)
            i = j
        else:
            i += 1
    # W6 the other separators and terminators are neutral
    t = ["ON" if x in ("ES", "ET", "CS") else x for x in t]
    # W7 European numbers after a Latin letter are Latin
    strong = edge
    for i in range(n):
        if t[i] in ("L", "R"):
            strong = t[i]
        elif t[i] == "EN" and strong == "L":
            t[i] = "L"

    def side(x: str) -> str:
        return "R" if x in ("R", "EN", "AN") else "L" if x == "L" else ""

    # N0 paired brackets
    stack: list[tuple[str, int]] = []
    pairs: list[tuple[int, int]] = []
    for i, c in enumerate(chars):
        if t[i] != "ON":
            continue
        if c in _BRACKETS:
            if len(stack) == 63:
                break
            stack.append((_BRACKETS[c], i))
        elif c in _CLOSERS:
            for k in range(len(stack) - 1, -1, -1):
                if stack[k][0] == c:
                    pairs.append((stack[k][1], i))
                    del stack[k:]
                    break
    for a, b in sorted(pairs):
        inside = {side(x) for x in t[a + 1 : b]} - {""}
        if not inside:
            continue
        if edge not in inside and next((side(t[k]) for k in range(a - 1, -1, -1) if side(t[k])), edge) != edge:
            new = "L" if edge == "R" else "R"  # the other direction inside and before the pair
        else:
            new = edge
        for k in (a, b):
            t[k] = new
            m = k + 1
            while m < n and nsm[m]:  # marks on a bracket go with it
                t[m] = new
                m += 1
    # N1 / N2 neutrals between two characters of one direction take it, the others the paragraph's
    i = 0
    while i < n:
        if t[i] in _NI:
            j = i
            while j < n and t[j] in _NI:
                j += 1
            left = side(t[i - 1]) if i else edge
            right = side(t[j]) if j < n else edge
            t[i:j] = [left if left == right else edge] * (j - i)
            i = j
        else:
            i += 1
    # I1 / I2
    levels = []
    for x in t:
        if e % 2 == 0:
            levels.append(e + (1 if x == "R" else 2 if x in ("AN", "EN") else 0))
        else:
            levels.append(e + (1 if x in ("L", "EN", "AN") else 0))
    # L1 spaces at the end of the line
    k = n
    while k and unicodedata.bidirectional(chars[k - 1]) in ("WS", "S", "B"):
        k -= 1
        levels[k] = e
    return levels


def reorder(text: str, base: str | None = None) -> str:
    """``text`` (one line, logical order) as it is drawn from left to right. ``base``: 'R', 'L' or None (``base_of``)."""
    if not text:
        return text
    e = 1 if (base or base_of(text)) == "R" else 0
    chars = [c for c in text if unicodedata.bidirectional(c) not in _EXPLICIT]
    if not chars:
        return ""
    levels = _resolve(chars, e)
    out = [c.translate(_MIRROR) if lv % 2 else c for c, lv in zip(chars, levels, strict=True)]
    top, low = max(levels), min((lv for lv in levels if lv % 2), default=None)
    if low is not None:
        for lv in range(top, low - 1, -1):
            i = 0
            while i < len(out):
                if levels[i] >= lv:
                    j = i
                    while j < len(out) and levels[j] >= lv:
                        j += 1
                    out[i:j] = out[i:j][::-1]
                    i = j
                else:
                    i += 1
    return "".join(c for c in out if c not in _MARKS)


def ltr(token: str) -> str:
    """``token`` held together as one left-to-right piece inside right-to-left text ("@name", "#tag", "twitch.tv/x",
    "C++"): a mark on each side makes its first and last signs part of it."""
    return f"{LRM}{token}{LRM}"
