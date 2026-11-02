"""The eye on 270 real club crops (run24 eyecheck): covered count must match (46 fp32 / 45-47 bf16).
usage: python bench/eye_equiv.py HEAD_DIR [CROP_TAR_DIR]"""

import glob
import io
import json
import sys
import tarfile

import numpy as np
from PIL import Image

from streamsafe.models.torch_backend.eye import RadioEye

head_dir = sys.argv[1]
crop_dir = sys.argv[2] if len(sys.argv) > 2 else "/workspace/eyecrops"
images = []
for path in sorted(glob.glob(f"{crop_dir}/*.tar")):
    with tarfile.open(path) as tf:
        for member in tf.getmembers():
            if member.isfile():
                images.append(np.asarray(Image.open(io.BytesIO(tf.extractfile(member).read())).convert("RGB")))
images = images[:270]
eye = RadioEye("nvidia/C-RADIOv4-H", 224, 384, head_dir)
p = np.concatenate([eye.prisk(eye.embed(images[i : i + 9])) for i in range(0, len(images), 9)])
result = {"n": len(images), "bf16": eye.bf16, "covered": int((p >= eye.thr).sum()), "mean_p": round(float(p.mean()), 4)}
result["expected"] = "46 fp32 / 45-47 bf16"
print("EYE_EQUIV " + json.dumps(result), flush=True)
