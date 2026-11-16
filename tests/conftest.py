import numpy as np
import pytest

from streamsafe.config import Config
from streamsafe.io.sources import VideoFileSource
from streamsafe.runtime.runner import run
from tests.stubs import stub_analyzer, stub_scene_analyzer


class MemorySink:
    """Keeps every output frame at analysis size (no codec loss)."""

    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []

    def write(self, frame: np.ndarray) -> None:
        import cv2

        self.frames.append(cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA))

    def close(self) -> None:
        pass


def run_stub(path: str, mode: str, scene_model: bool = False, **settings) -> tuple[dict, MemorySink]:
    # the governor follows the real clock: on a busy CI runner a slow stand-in decision would add covers and make
    # these runs depend on the machine. It has its own tests (tests/test_governor.py), also through this runner
    settings.setdefault("eye_budget_adapt", False)
    cfg = Config.from_env(env={}, mode=mode, delay=1.0, **settings)
    source = VideoFileSource(path, cfg.runtime.fps)
    sink = MemorySink()
    try:
        report = run(cfg, source, sink, factory=stub_scene_analyzer if scene_model else stub_analyzer)
    finally:
        source.close()
    return report, sink


@pytest.fixture
def stub_run():
    return run_stub
