"""Builds the translation pack (``streamsafe/translate/pack.py``): downloads the OCR models and the fonts at their
pinned sources, converts Helsinki-NLP/opus-mt-en-ar to CTranslate2 int8, writes the notices and licence texts, and
prints the table to paste into ``streamsafe/app/release.py`` (``TRANSLATE_FILES``).

    pip install ctranslate2 sentencepiece torch transformers   # conversion only; the app needs ctranslate2 + sentencepiece
    python tools/build_translate_pack.py OUT_DIR

Upload the finished folder to ``release.TRANSLATE_BASE_URL``. The converted ``model.bin`` is not byte-for-byte
reproducible across converter versions: the table pins the copy that was uploaded.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path

OPUS_REPO = "Helsinki-NLP/opus-mt-en-ar"
OPUS_REVISION = "03087980e8ce753d64b3248ed0a912444545b840"
RAPID = "https://huggingface.co/SWHL/RapidOCR/resolve/1cfba2e90fc938db55889873735088de210cc173/"
# name in the pack -> (url, sha256 the source must have)
SOURCES = {
    "det.onnx": (RAPID + "PP-OCRv4/ch_PP-OCRv4_det_infer.onnx", "d2a7720d45a54257208b1e13e36a8479894cb74155a5efe29462512d42f49da9"),
    "rec.onnx": (RAPID + "PP-OCRv3/en_PP-OCRv3_rec_infer.onnx", "ef7abd8bd3629ae57ea2c28b425c1bd258a871b93fd2fe7c433946ade9b5d9ea"),
    "NotoSansArabic-Regular.ttf": (
        "https://raw.githubusercontent.com/notofonts/notofonts.github.io/main/fonts/NotoSansArabic/hinted/ttf/NotoSansArabic-Regular.ttf",
        "",
    ),
    "NotoSans-Regular.ttf": (
        "https://raw.githubusercontent.com/notofonts/notofonts.github.io/main/fonts/NotoSans/hinted/ttf/NotoSans-Regular.ttf",
        "",
    ),
}
LICENSES = {
    "LICENSE-Apache-2.0.txt": "https://www.apache.org/licenses/LICENSE-2.0.txt",
    "LICENSE-CC-BY-4.0.txt": "https://creativecommons.org/licenses/by/4.0/legalcode.txt",
    "LICENSE-OFL-NotoSansArabic.txt": "https://raw.githubusercontent.com/notofonts/arabic/main/OFL.txt",
    "LICENSE-OFL-NotoSans.txt": "https://raw.githubusercontent.com/notofonts/latin-greek-cyrillic/main/OFL.txt",
}
NOTICE = """StreamSafe - translation pack: third-party material and its licences

1. Translation model. Helsinki-NLP/opus-mt-en-ar (OPUS-MT, University of Helsinki and the OPUS-MT project),
   revision {rev}. Declared licence: CC-BY 4.0 (OPUS-MT project: github.com/Helsinki-NLP/Opus-MT, "pre-trained
   downloadable translation models, CC-BY 4.0") and Apache-2.0 (the Hugging Face model card). Both allow commercial
   use and redistribution with attribution; both licence texts are in this folder.
   CHANGES: the weights were converted to the CTranslate2 format and quantized to 8-bit integers (model.bin,
   config.json, shared_vocabulary.json); the SentencePiece files (source.spm, target.spm) are unchanged.
   Attribution: "Translation by OPUS-MT (Helsinki-NLP), https://github.com/Helsinki-NLP/Opus-MT, CC-BY 4.0."
   Note: the OPUS-MT / Tatoeba-Challenge README states that the Tatoeba *training data* is CC-BY-NC-SA 4.0. The
   released models themselves are declared CC-BY 4.0 / Apache-2.0 by their authors.

2. Text reader. PaddleOCR PP-OCRv4 text detector (det.onnx) and PP-OCRv3 English recogniser (rec.onnx), Apache-2.0,
   Copyright PaddlePaddle Authors; ONNX conversions published by RapidOCR (github.com/RapidAI/RapidOCR, Apache-2.0,
   huggingface.co/SWHL/RapidOCR). Licence text: LICENSE-Apache-2.0.txt.

3. Fonts. Noto Sans Arabic and Noto Sans, Copyright The Noto Project Authors, SIL Open Font License 1.1
   (LICENSE-OFL-*.txt). The fonts are not modified.

Software (not in this folder): CTranslate2 (MIT, OpenNMT), SentencePiece (Apache-2.0, Google), arabic-reshaper (MIT),
ONNX Runtime (MIT, Microsoft), Pillow (HPND), OpenCV (Apache-2.0).
"""


def get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "StreamSafe-build"}), timeout=120) as r:
        return r.read()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, (url, want) in SOURCES.items():
        data = get(url)
        if want and hashlib.sha256(data).hexdigest() != want:
            raise SystemExit(f"{name}: the source does not have the pinned sha256")
        (out / name).write_bytes(data)
    for name, url in LICENSES.items():
        (out / name).write_bytes(get(url))
    (out / "NOTICE-translate.txt").write_text(NOTICE.format(rev=OPUS_REVISION), encoding="utf-8")
    from ctranslate2.converters import TransformersConverter

    tmp = out / "_ct2"
    shutil.rmtree(tmp, ignore_errors=True)
    TransformersConverter(OPUS_REPO, revision=OPUS_REVISION, copy_files=["source.spm", "target.spm"]).convert(str(tmp), quantization="int8")
    for f in tmp.iterdir():
        shutil.move(str(f), out / f.name)
    tmp.rmdir()
    names = [p for p in sorted(out.iterdir()) if p.is_file() and p.name != "manifest.json"]
    files = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in names}
    (out / "manifest.json").write_text(json.dumps({"files": files}, indent=1), encoding="utf-8")
    print("# paste into streamsafe/app/release.py")
    print("TRANSLATE_FILES: dict[str, tuple[str, int]] = {")
    for n, m in files.items():
        print(f'    "{n}": ("{m["sha256"]}", {m["bytes"]}),')
    print("}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
