# Third-party models and notices

StreamSafe ships or downloads these models. Each keeps its own license; the license texts go into the installer's
`licenses/` folder next to this notice.

| Model | Use in StreamSafe | License | Notice |
|---|---|---|---|
| nvidia/C-RADIOv4-H (pinned commit in `streamsafe/config.py`), and the ONNX export made from it | the eye (who must be covered) | NVIDIA Open Model License | **Licensed by NVIDIA Corporation under the NVIDIA Open Model License.** |
| RF-DETR-Seg (Roboflow, `rfdetr` package) | person detector + masks | Apache-2.0 | Copyright Roboflow, Inc. |
| google/siglip2-base-patch16-224 | scene model | Apache-2.0 | Copyright Google LLC |
| usyd-community/vitpose-base-simple | pose (garment shape) | Apache-2.0 | ViTPose authors |
| RapidOCR (`rapidocr_onnxruntime`), optional | HUD reader | Apache-2.0 | RapidAI |
| ONNX Runtime | model runtime | MIT | Copyright Microsoft Corporation |
| Helsinki-NLP/opus-mt-en-ar (OPUS-MT, pinned revision in `tools/build_translate_pack.py`), converted to CTranslate2 int8; optional, downloaded when the streamer switches on Arabic translation | English -> Arabic translation of on-screen text | CC-BY 4.0 (OPUS-MT project) and Apache-2.0 (Hugging Face model card) | **Translation by OPUS-MT (Helsinki-NLP), https://github.com/Helsinki-NLP/Opus-MT, CC-BY 4.0.** Converted to CTranslate2 and quantized to int8; see the open question below |
| PaddleOCR PP-OCRv4 text detector, PP-OCRv3 English recogniser (ONNX conversions by RapidOCR, `huggingface.co/SWHL/RapidOCR`); same optional download | reads the on-screen text | Apache-2.0 | Copyright PaddlePaddle Authors |
| Noto Sans Arabic, Noto Sans (notofonts); same optional download | fonts of the Arabic box | SIL OFL 1.1 | Copyright The Noto Project Authors |
| CTranslate2 | runs the translator on the CPU | MIT | Copyright SYSTRAN (OpenNMT CTranslate2) |
| SentencePiece | text pieces for the translator | Apache-2.0 | Copyright Google |
| arabic-reshaper | Arabic letter shaping | MIT | Copyright Abdullah Diab |
| NVIDIA TensorRT 10.9 libraries (NVIDIA edition only) | compiles and runs the models on NVIDIA cards (`TRT=1`) | NVIDIA software license with the TensorRT supplement (proprietary; `LICENSE.txt` in the `tensorrt_cu12_libs` wheel) | **Open before shipping:** its distribution clause (12.1) names the libnvinfer and libnvinfer_plugin libraries only; see `docs/NVIDIA_TRT.md` section 1 |

The NVIDIA Open Model License also asks users of the model to follow NVIDIA's Trustworthy AI terms; StreamSafe uses
the eye only to decide which on-screen characters to cover.

Open question for the translation model (to be decided by the owner): the OPUS-MT / Tatoeba-Challenge README states that the
*training data* of the Tatoeba models is CC-BY-NC-SA 4.0 and its test sets CC-BY 2.0 FR, while the released *models* are declared
CC-BY 4.0 (OPUS-MT) and Apache-2.0 (Hugging Face card). We follow the licence the authors declare for the model; whether a
non-commercial training-data licence could reach the weights is a legal judgement we did not make. The translator is a swappable
file set (`streamsafe/translate/translator.py`, `tools/build_translate_pack.py`) if the answer is "no". NLLB and M2M (non-commercial) are not used.
Not allowed in the product: AGPL code or weights (Ultralytics YOLO, NudeNet) and research-only weights.
FFmpeg, when bundled, must be an LGPL build without GPL parts (no libx264).
