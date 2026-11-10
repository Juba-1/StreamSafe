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

The NVIDIA Open Model License also asks users of the model to follow NVIDIA's Trustworthy AI terms; StreamSafe uses
the eye only to decide which on-screen characters to cover.

Not allowed in the product: AGPL code or weights (Ultralytics YOLO, NudeNet) and research-only weights.
FFmpeg, when bundled, must be an LGPL build without GPL parts (no libx264).
