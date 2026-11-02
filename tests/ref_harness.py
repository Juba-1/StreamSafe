"""Runs the previous engine (engine2) in offline/step mode with its own stubs and prints, as JSON,
the md5 of every output frame and its diag log. Used by test_equivalence (separate process,
because both engines are packages named ``streamsafe``)."""

import hashlib
import json
import os
import sys
import tempfile

ref_dir, video = sys.argv[1], sys.argv[2]
sys.path.insert(0, ref_dir)
os.environ["SS_STUB"] = "1"
from streamsafe import runtime  # noqa: E402  (the reference engine)
from streamsafe.config import Config  # noqa: E402

hashes = []
original = runtime.Output.render


def render(self, v, f, decs, keys, cut_between):
    out = original(self, v, f, decs, keys, cut_between)
    hashes.append(hashlib.md5(out.tobytes()).hexdigest())
    return out


runtime.Output.render = render
diag = os.path.join(tempfile.mkdtemp(), "diag.json")
runtime.run_step(Config.from_env(src=video, mode="step", diag_out=diag))
with open(diag) as f:
    print("REF_RESULT " + json.dumps({"hashes": hashes, "diag": json.load(f)}))
