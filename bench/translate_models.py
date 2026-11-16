"""Translation models side by side on the multi-game set (tests/data/translate_eval_multigame.tsv): quality, speed, memory.

Two steps, so the GPU machine needs only this file, ``streamsafe/translate/{llm,textproc}.py`` and the TSV:

    # on the GPU (writes one JSON file per model: the raw answers, the time of every line, the video memory used)
    python -m bench.translate_models llm   --gguf M.gguf --label qwen3-4b [--style chatml] [--gpu-layers -1] --out runs/
    python -m bench.translate_models ct2   --dir madlad-ct2 --label madlad-3b --out runs/      # MADLAD-400 (T5) on CTranslate2
    # anywhere (opus-mt from the translation pack, on the CPU: the product's fallback)
    python -m bench.translate_models opus  --pack PACK_DIR --out runs/
    # then: chrF++ per model and kind of text, how many answers the product's check refuses, latency
    python -m bench.translate_models score runs/ [--dump table.tsv]

Latency is one line at a time (what the live pipeline does), after one warm-up line. "short" lines are those of at
most 40 characters. chrF++ is sacrebleu's CHRF(word_order=2) on text without Arabic diacritics.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import subprocess
import time
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "tests" / "data" / "translate_eval_multigame.tsv"
SHORT = 40
_DIAC = re.compile("[\u064b-\u0652\u0640]")


def load(path: Path = DATA) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


def gpu_used_mb() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10
        ).stdout
        return max(int(x) for x in out.split())
    except Exception:
        return None


def timed(rows: list[dict], one) -> tuple[list[str], list[float]]:
    one("Press X to end the dance.")
    outs, ms = [], []
    for r in rows:
        t = time.perf_counter()
        outs.append(one(r["en"]))
        ms.append(round(1000 * (time.perf_counter() - t), 2))
    return outs, ms


def save(out: Path, label: str, rows: list[dict], raw: list[str], ms: list[float], **extra) -> None:
    out.mkdir(parents=True, exist_ok=True)
    doc = dict(label=label, en=[r["en"] for r in rows], raw=raw, ms=ms, **extra)
    (out / f"{label}.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    s = sorted(ms)
    print(f"{label}: {len(raw)} lines, median {statistics.median(s):.0f} ms, p95 {s[int(0.95 * (len(s) - 1))]:.0f} ms, {extra}")


def run_llm(a: argparse.Namespace) -> None:
    from streamsafe.translate.llm import LlamaTranslator

    rows = load(Path(a.data))
    before = gpu_used_mb()
    t = time.perf_counter()
    tr = LlamaTranslator(Path(a.gguf), a.style or None, a.gpu_layers, a.threads)
    load_s = round(time.perf_counter() - t, 1)
    raw, ms = timed(rows, tr.answer)
    after = gpu_used_mb()
    vram = after - before if before is not None and after is not None else None
    save(Path(a.out), a.label, rows, raw, ms, style=tr.style, vram_mb=vram, load_s=load_s, gpu_layers=a.gpu_layers, device=a.device)


def run_ct2(a: argparse.Namespace) -> None:
    """MADLAD-400 (T5, "<2ar> text") converted to CTranslate2."""
    import ctranslate2
    import sentencepiece as spm

    rows = load(Path(a.data))
    before = gpu_used_mb()
    model = ctranslate2.Translator(a.dir, device=a.device, compute_type="int8_float16" if a.device == "cuda" else "int8")
    sp = spm.SentencePieceProcessor(model_file=str(Path(a.dir) / "spiece.model"))

    def one(text: str) -> str:
        res = model.translate_batch([sp.encode("<2ar> " + text, out_type=str)], beam_size=2, max_decoding_length=160)
        return sp.decode(res[0].hypotheses[0])

    raw, ms = timed(rows, one)
    after = gpu_used_mb()
    save(Path(a.out), a.label, rows, raw, ms, style="madlad", vram_mb=after - before if before is not None and after is not None else None)


def run_opus(a: argparse.Namespace) -> None:
    """The pack's opus-mt on the CPU: bare, and inside the game rules (what the product shows without the strong model)."""
    from streamsafe.translate.translator import Ct2Translator, GameTranslator

    rows = load(Path(a.data))
    bare = Ct2Translator(Path(a.pack), a.threads)
    for label, tr in (("opus-mt", bare), ("opus-mt+rules", GameTranslator(bare))):
        raw, ms = timed(rows, lambda s, tr=tr: tr.translate([s])[0])
        save(Path(a.out), label, rows, raw, ms, style="final", vram_mb=0, device="cpu")


def score(a: argparse.Namespace) -> None:
    from sacrebleu.metrics import CHRF

    from streamsafe.translate.llm import clean_answer

    rows = load(Path(a.data))
    refs = [_DIAC.sub("", r["ar"]) for r in rows]
    chrf = CHRF(word_order=2)
    kinds = sorted({r["category"] for r in rows})
    short = [i for i, r in enumerate(rows) if len(r["en"]) <= SHORT]
    print("| model | chrF++ | " + " | ".join(kinds) + " | refused | ms median | ms p95 | ms p95 short | VRAM MB |")
    print("|---|---|" + "---|" * (len(kinds) + 5))
    table: dict[str, list[str]] = {}
    for f in sorted(Path(a.runs).glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        if d["en"] != [r["en"] for r in rows]:
            print(f"{f.name}: not this data set, skipped")
            continue
        final = d.get("style") == "final"  # already the product's output (opus-mt)
        hyps = [h if final else clean_answer(en, h) for en, h in zip(d["en"], d["raw"], strict=True)]
        refused = sum(1 for h in hyps if not h)
        shown = [_DIAC.sub("", h) for h in hyps]
        raw = [_DIAC.sub("", h.strip().split("\n")[0]) for h in d["raw"]]  # scored as the model answered: a refusal is not hidden
        table[d["label"]] = hyps

        def s(idx: list[int], hyp: list[str] = raw) -> str:
            return f"{chrf.corpus_score([hyp[i] for i in idx], [[refs[i] for i in idx]]).score:.1f}"

        per = [s([i for i, r in enumerate(rows) if r["category"] == k]) for k in kinds]
        ms, ms_short = sorted(d["ms"]), sorted(d["ms"][i] for i in short)
        p95 = lambda xs: xs[int(0.95 * (len(xs) - 1))]  # noqa: E731
        print(
            f"| {d['label']} | {s(list(range(len(rows))))} | " + " | ".join(per) + f" | {refused} | {statistics.median(ms):.0f} | "
            f"{p95(ms):.0f} | {p95(ms_short):.0f} | {d.get('vram_mb')} |"
        )
        del shown
    if a.dump:
        with open(a.dump, "w", encoding="utf-8") as f:
            f.write("genre\tcategory\ten\tref\t" + "\t".join(table) + "\n")
            for i, r in enumerate(rows):
                f.write("\t".join([r["genre"], r["category"], r["en"], r["ar"], *(t[i] or "-" for t in table.values())]) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("llm", run_llm), ("ct2", run_ct2), ("opus", run_opus), ("score", score)):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        p.add_argument("--data", default=str(DATA))
        p.add_argument("--threads", type=int, default=2)
        if name != "score":
            p.add_argument("--out", required=True)
            p.add_argument("--label", default=name)
            p.add_argument("--device", default="cuda")
    sub.choices["llm"].add_argument("--gguf", required=True)
    sub.choices["llm"].add_argument("--style", default="")
    sub.choices["llm"].add_argument("--gpu-layers", type=int, default=-1)
    sub.choices["ct2"].add_argument("--dir", required=True)
    sub.choices["opus"].add_argument("--pack", required=True)
    sub.choices["score"].add_argument("runs")
    sub.choices["score"].add_argument("--dump")
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
