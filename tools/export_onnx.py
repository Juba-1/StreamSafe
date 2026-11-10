"""Export every model the engine uses to ONNX, so it runs on any GPU through ONNX Runtime.

    python -m tools.export_onnx OUT_DIR [detector] [pose] [scene] [eye]   (default: all four)

Writes into OUT_DIR:
  detector.onnx          RF-DETR-Seg (official rfdetr exporter), fixed 1x3xRxR input
  pose.onnx              ViTPose-base, input (n, 3, 256, 192) normalised, output heatmaps (n, 17, 64, 48)
  scene_vision.onnx      SigLIP2 vision tower, input (1, 3, 224, 224) in [-1, 1], output pooled embedding (1, 768)
  scene_text.npy         normalised text embeddings of the scene prompts (fixed, computed once here)
  scene_logit.npy        [logit_scale.exp(), logit_bias]
  eye.onnx (+ eye.onnx.data)  C-RADIOv4-H summary embedding, input (n, 3, H, W) in [0, 1], fp32
  manifest.json          repo + pinned revision + sha256 of every file (the engine checks it before loading)

Model versions come from streamsafe/config.py (the same pins the PyTorch backend uses). The eye runs the
model's own repository code (trust_remote_code) at its pinned commit; only this export step needs that.
"""

from __future__ import annotations

import json
import shutil
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from streamsafe.config import Config
from streamsafe.models.onnx_backend import sha256

OPSET = 17


def export_detector(cfg: Config, out: Path) -> dict:
    import rfdetr

    model = getattr(rfdetr, "RFDETRSeg" + cfg.detector.det_model)()
    tmp = out / "_rfdetr"
    path = Path(model.export(output_dir=str(tmp), fp16=False, verbose=False))
    shutil.move(str(path), out / "detector.onnx")
    shutil.rmtree(tmp, ignore_errors=True)
    return {"detector.onnx": {"model": f"rfdetr RFDETRSeg{cfg.detector.det_model}", "rfdetr": version("rfdetr")}}


class _Pose(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(pixel_values=x).heatmaps


def export_pose(cfg: Config, out: Path) -> dict:
    from transformers import VitPoseForPoseEstimation

    g = cfg.garment
    model = VitPoseForPoseEstimation.from_pretrained(g.pose_repo, revision=g.pose_revision).eval()
    torch.onnx.export(
        _Pose(model), (torch.zeros(2, 3, 256, 192),), str(out / "pose.onnx"), input_names=["pixels"], output_names=["heatmaps"],
        dynamic_axes={"pixels": {0: "n"}, "heatmaps": {0: "n"}}, opset_version=OPSET, dynamo=False,
    )  # fmt: skip
    return {"pose.onnx": {"repo": g.pose_repo, "revision": g.pose_revision}}


class _SceneVision(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.vision = model.vision_model

    def forward(self, x):
        return self.vision(pixel_values=x).pooler_output


def export_scene(cfg: Config, out: Path) -> dict:
    from transformers import AutoModel, AutoProcessor

    from streamsafe.models.scene_common import PROMPTS

    s = cfg.scene
    model = AutoModel.from_pretrained(s.scene_repo, revision=s.scene_revision).eval()
    processor = AutoProcessor.from_pretrained(s.scene_repo, revision=s.scene_revision)
    text = processor(text=PROMPTS, padding="max_length", max_length=64, return_tensors="pt")
    with torch.inference_mode():
        t = model.get_text_features(**text)
        t = torch.nn.functional.normalize(getattr(t, "pooler_output", t).float(), dim=-1)
    np.save(out / "scene_text.npy", t.numpy().astype(np.float32))
    np.save(out / "scene_logit.npy", np.array([model.logit_scale.exp().item(), model.logit_bias.item()], np.float32))
    torch.onnx.export(
        _SceneVision(model), (torch.zeros(1, 3, 224, 224),), str(out / "scene_vision.onnx"), input_names=["pixels"],
        output_names=["embedding"], opset_version=OPSET, dynamo=False,
    )  # fmt: skip
    meta = {"repo": s.scene_repo, "revision": s.scene_revision}
    return {name: dict(meta) for name in ("scene_vision.onnx", "scene_text.npy", "scene_logit.npy")}


class _Eye(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)[0]


def export_eye(cfg: Config, out: Path) -> dict:
    """The exporter writes a >2 GB model as one file per tensor: export into a scratch folder, then save it
    as eye.onnx + one eye.onnx.data, so the manifest covers every byte the engine loads."""
    import onnx
    from transformers import AutoModel

    e = cfg.eye
    model = AutoModel.from_pretrained(e.eye_repo, revision=e.eye_revision, trust_remote_code=True).float().eval()
    scratch = out / "_eye"
    scratch.mkdir(exist_ok=True)
    torch.onnx.export(
        _Eye(model), (torch.rand(2, 3, e.eye_h, e.eye_w),), str(scratch / "eye.onnx"), input_names=["pixels"],
        output_names=["summary"], dynamic_axes={"pixels": {0: "n"}, "summary": {0: "n"}}, opset_version=OPSET, dynamo=False,
        external_data=True,
    )  # fmt: skip
    graph = onnx.load(str(scratch / "eye.onnx"), load_external_data=True)
    onnx.save_model(graph, str(out / "eye.onnx"), save_as_external_data=True, all_tensors_to_one_file=True, location="eye.onnx.data")
    shutil.rmtree(scratch)
    meta = {"repo": e.eye_repo, "revision": e.eye_revision, "input_hw": [e.eye_h, e.eye_w]}
    return {name: dict(meta) for name in ("eye.onnx", "eye.onnx.data")}


EXPORTERS = {"detector": export_detector, "pose": export_pose, "scene": export_scene, "eye": export_eye}


def main(argv: list[str]) -> None:
    if not argv:
        sys.exit(__doc__)
    out = Path(argv[0])
    out.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"files": {}}
    for name in argv[1:] or list(EXPORTERS):
        print("export", name, flush=True)
        manifest["files"].update(EXPORTERS[name](cfg, out))
    for fname, meta in manifest["files"].items():
        meta["sha256"] = sha256(out / fname)
        meta["bytes"] = (out / fname).stat().st_size
    manifest_path.write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
