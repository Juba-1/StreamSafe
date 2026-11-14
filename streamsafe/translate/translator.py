"""English -> Arabic: Helsinki-NLP/opus-mt-en-ar (Marian transformer) run by CTranslate2 (MIT) on the CPU in int8.

``Ct2Translator`` is the model. ``CachedTranslator`` wraps any translator with a cache keyed by normalized text
(``textproc.cache_key``): games repeat their lines, and OCR reads one line a little differently each time.
Anything with ``translate(list[str]) -> list[str]`` can stand in for the model (the tests use stubs).
"""

from __future__ import annotations

import collections
from pathlib import Path
from typing import Protocol

from streamsafe.translate import gamefix, textproc

MODEL_FILE = "model.bin"
TARGET_TOKEN = ">>ara<<"  # opus-mt-en-ar needs a sentence-initial target-language token
EOS = "</s>"


class Translator(Protocol):
    def translate(self, texts: list[str]) -> list[str]: ...


class Ct2Translator:
    def __init__(
        self,
        folder: Path,
        threads: int = 2,
        beam: int = 2,
        repetition_penalty: float = 1.1,
        no_repeat_ngram_size: int = 3,
        length_penalty: float = 1.0,
    ) -> None:
        import ctranslate2
        import sentencepiece as spm

        self.model = ctranslate2.Translator(str(folder), device="cpu", compute_type="int8", inter_threads=1, intra_threads=threads)
        self.source = spm.SentencePieceProcessor(model_file=str(folder / "source.spm"))
        self.target = spm.SentencePieceProcessor(model_file=str(folder / "target.spm"))
        self.beam = beam
        self.decoding = {
            "repetition_penalty": repetition_penalty,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "length_penalty": length_penalty,
        }

    def translate(self, texts: list[str]) -> list[str]:
        """One Arabic string per input. Long texts are split into sentences first (the model drops words from long
        inputs) and the pieces are translated as one batch."""
        pieces: list[list[str]] = [textproc.sentences(t) for t in texts]
        flat = [s for ps in pieces for s in ps]
        if not flat:
            return [""] * len(texts)
        batch = [[TARGET_TOKEN, *self.source.encode(s, out_type=str), EOS] for s in flat]
        results = self.model.translate_batch(
            batch,
            beam_size=self.beam,
            max_decoding_length=max(len(b) for b in batch) * 3 + 8,
            **self.decoding,  # no_repeat_ngram_size: without it this model can loop on a word until the length limit
        )
        done = [self.target.decode(r.hypotheses[0]).strip() for r in results]
        out, i = [], 0
        for ps in pieces:
            out.append(" ".join(d for d in done[i : i + len(ps)] if d))
            i += len(ps)
        return out


class GameTranslator:
    """Game text around the model (``gamefix``): fixed Arabic for stock UI lines, key prompts built from the key and the
    translated action, numbers and place names kept out of the model's hands, ALL CAPS / Title Case read as sentences,
    the Arabic cleaned afterwards. One model call per batch, as without it. ``inner`` is any ``Translator``."""

    def __init__(self, inner: Translator, glossary: bool = True, common=None) -> None:
        self.inner, self.glossary = inner, glossary
        if common is None and hasattr(inner, "source"):
            sp, unk = inner.source, inner.source.unk_id()
            common = lambda w: sp.piece_to_id("\u2581" + w) != unk  # noqa: E731 - is it an everyday word of the model's vocabulary
        self.common = common

    def translate(self, texts: list[str]) -> list[str]:
        jobs = [[gamefix.plan(s, self.common, self.glossary) for s in textproc.sentences(t)] for t in texts]
        flat = [j for js in jobs for j in js]
        need = [j for j in flat if not j.fixed and j.src]
        done = dict(zip(map(id, need), self.inner.translate([j.src for j in need]) if need else [], strict=True))
        result = {id(j): gamefix.finish(j, done.get(id(j), "")) for j in flat}
        redo = [j for j in flat if result[id(j)] is None]  # a stand-in was lost or the output was no translation
        if redo:
            plain = [gamefix.Job(j.text, src=gamefix.sentence_case(j.text, self.common), imperative=j.imperative) for j in redo]
            for j, p, ar in zip(redo, plain, self.inner.translate([p.src for p in plain]), strict=True):
                result[id(j)] = gamefix.finish(p, ar) or ""
        return [" ".join(r for j in js if (r := result[id(j)])) for js in jobs]


class CachedTranslator:
    """``translate`` answers from the cache where it can and asks the model only for the rest."""

    def __init__(self, inner: Translator, size: int = 512) -> None:
        self.inner, self.size = inner, size
        self._cache: collections.OrderedDict[str, str] = collections.OrderedDict()
        self.hits = self.misses = 0

    def translate(self, texts: list[str]) -> list[str]:
        keys = [textproc.cache_key(t) for t in texts]
        todo = {k: t for k, t in zip(keys, texts, strict=True) if k and k not in self._cache}
        if todo:
            for k, ar in zip(todo, self.inner.translate(list(todo.values())), strict=True):
                self._cache[k] = ar
                if len(self._cache) > self.size:
                    self._cache.popitem(last=False)
        out = []
        for k in keys:
            if k in self._cache:
                self._cache.move_to_end(k)
            out.append(self._cache.get(k, ""))
        self.misses += len(todo)
        self.hits += sum(1 for k in keys if k) - len(todo)
        return out
