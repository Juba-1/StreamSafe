"""Arabic translation quality + latency on the GTA-style test set (tests/data/translate_gta.tsv).

    PYTHONPATH=. python -m bench.translate_quality PACK_DIR [--threads 1] [--dump out.tsv]

chrF (sacrebleu) against the written references, per category, and the CPU latency per line (median / p95, one line at
a time = what the live pipeline does). Uses the pack's current translator code (``streamsafe.translate.translator``).
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import time
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "tests" / "data"
DATA = DATA_DIR / "translate_gta.tsv"  # the set the glossary was tuned on
HELDOUT = DATA_DIR / "translate_gta_heldout.tsv"  # written before any rule was, never looked at while tuning


def load(path: Path = DATA) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


_DIAC = re.compile("[\u064b-\u0652\u0640]")


def norm(t: str) -> str:
    """Strip Arabic diacritics and tatweel from both sides: the references are written with some, the model with few."""
    return _DIAC.sub("", t)


def score(rows: list[dict], hyps: list[str]) -> dict[str, float]:
    from sacrebleu.metrics import CHRF

    chrf = CHRF()
    out = {"all": chrf.corpus_score([norm(h) for h in hyps], [[norm(r["ar"]) for r in rows]]).score}
    for cat in sorted({r["category"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["category"] == cat]
        out[cat] = chrf.corpus_score([norm(hyps[i]) for i in idx], [[norm(rows[i]["ar"]) for i in idx]]).score
    return out


def latency(translator, lines: list[str]) -> tuple[float, float]:
    translator.translate(["Press X to end the dance."])  # warm
    ms = []
    for ln in lines:
        t = time.perf_counter()
        translator.translate([ln])
        ms.append(1000 * (time.perf_counter() - t))
    ms.sort()
    return statistics.median(ms), ms[int(0.95 * (len(ms) - 1))]


def run(translator, label: str, dump: str | None = None, lat: bool = True, path: Path = DATA) -> list[str]:
    rows = load(path)
    hyps = translator.translate([r["en"] for r in rows])
    s = score(rows, hyps)
    msg = f"{label}: chrF {s['all']:.1f} | " + " ".join(f"{k} {v:.0f}" for k, v in s.items() if k != "all")
    if lat:
        med, p95 = latency(translator, [r["en"] for r in rows])
        msg += f" | latency median {med:.0f} ms p95 {p95:.0f} ms"
    print(msg)
    if dump:
        with open(dump, "w", encoding="utf-8") as f:
            for r, h in zip(rows, hyps, strict=True):
                f.write(f"{r['category']}\t{r['en']}\t{h}\t{r['ar']}\n")
    return hyps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pack")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--beam", type=int, default=2)
    ap.add_argument("--dump")
    ap.add_argument("--heldout", action="store_true")
    ap.add_argument("--game", action="store_true", help="wrap the model in GameTranslator (the shipped pre/post-processing)")
    ap.add_argument("--no-glossary", action="store_true", help="with --game: no fixed lines / prompt templates / action glossary")
    a = ap.parse_args()
    from streamsafe.translate.translator import Ct2Translator, GameTranslator

    t = Ct2Translator(Path(a.pack), a.threads, a.beam)
    if a.game:
        t = GameTranslator(t, glossary=not a.no_glossary)
    run(t, f"{'game' if a.game else 'raw'} threads={a.threads} beam={a.beam}", a.dump, path=HELDOUT if a.heldout else DATA)


if __name__ == "__main__":
    main()
