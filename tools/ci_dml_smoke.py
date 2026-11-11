"""CI smoke test for the Windows/DirectML edition: the DirectML runtime loads and runs a small model.

GitHub's Windows runners have no GPU. DirectML then runs on Microsoft's software adapter, which is slow but
exercises the same DLLs, provider registration and operators as on an AMD/Intel/NVIDIA card. This catches
packaging breakage (missing DLL, wrong wheel) and operators DirectML cannot run, before a streamer does.
Speed is NOT measured here.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np


def tiny_model(path: Path) -> None:
    """A conv + matmul + softmax graph: the operator families our detector and eye depend on."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(0)
    w = numpy_helper.from_array(rng.standard_normal((8, 3, 3, 3)).astype(np.float32), "w")
    m = numpy_helper.from_array(rng.standard_normal((8 * 30 * 30, 4)).astype(np.float32), "m")
    shape = numpy_helper.from_array(np.array([1, -1], dtype=np.int64), "shape")
    nodes = [
        helper.make_node("Conv", ["x", "w"], ["c"]),
        helper.make_node("Relu", ["c"], ["r"]),
        helper.make_node("Reshape", ["r", "shape"], ["f"]),
        helper.make_node("MatMul", ["f", "m"], ["l"]),
        helper.make_node("Softmax", ["l"], ["y"], axis=1),
    ]
    graph = helper.make_graph(
        nodes,
        "smoke",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 32, 32])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
        [w, m, shape],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), str(path))


def main() -> int:
    import onnxruntime as ort

    providers = ort.get_available_providers()
    print("providers:", providers)
    if "DmlExecutionProvider" not in providers:
        print("FAIL: DirectML provider missing from this onnxruntime build")
        return 1
    with tempfile.TemporaryDirectory() as d:
        model = Path(d) / "smoke.onnx"
        tiny_model(model)
        so = ort.SessionOptions()
        so.enable_mem_pattern = False  # DirectML requirement (onnxruntime docs)
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        x = np.random.default_rng(1).standard_normal((1, 3, 32, 32)).astype(np.float32)
        dml = ort.InferenceSession(str(model), so, providers=["DmlExecutionProvider"]).run(None, {"x": x})[0]
        cpu = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
    diff = float(np.abs(dml - cpu).max())
    print(f"DirectML vs CPU max diff: {diff:.2e}")
    if not np.isfinite(dml).all() or diff > 1e-3:
        print("FAIL: DirectML output differs from CPU")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
