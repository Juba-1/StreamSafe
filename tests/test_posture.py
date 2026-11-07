import numpy as np

from streamsafe.analysis.posture import NAMES, features

COL = {n: i for i, n in enumerate(NAMES)}


def _standing() -> tuple[np.ndarray, np.ndarray]:
    kp = np.zeros((17, 2), np.float32)
    kp[0] = (0.5, 0.08)  # nose
    kp[1], kp[2] = (0.47, 0.06), (0.53, 0.06)
    kp[5], kp[6] = (0.35, 0.2), (0.65, 0.2)  # shoulders
    kp[9], kp[10] = (0.3, 0.5), (0.7, 0.5)  # wrists
    kp[11], kp[12] = (0.4, 0.5), (0.6, 0.5)  # hips
    kp[13], kp[14] = (0.4, 0.72), (0.6, 0.72)  # knees
    kp[15], kp[16] = (0.4, 0.95), (0.6, 0.95)  # ankles
    return kp, np.ones(17, np.float32)


def test_standing_is_neutral():
    kp, ks = _standing()
    f = features(kp, ks, 0.4)
    assert f.shape == (len(NAMES),)
    assert f[COL["tilt"]] < 0.1 and f[COL["head_below_hips"]] == 0 and f[COL["knee_bend"]] < 0.05
    assert f[COL["back_view"]] == 0 and f[COL["arms_up"]] == 0


def test_lying_tilts_and_bent_over_puts_head_low():
    kp, ks = _standing()
    lying = kp[:, ::-1].copy()  # swap x / y: the body runs left to right
    assert features(lying, ks, 2.0)[COL["tilt"]] > 0.8
    bent = kp.copy()
    bent[0, 1] = 0.7  # nose below the hips
    assert features(bent, ks, 0.8)[COL["head_below_hips"]] == 1


def test_back_view_and_missing_joints():
    kp, ks = _standing()
    ks[[0, 1, 2]] = 0.05  # no face, shoulders seen
    assert features(kp, ks, 0.4)[COL["back_view"]] == 1
    f = features(kp, np.zeros(17, np.float32), 0.4)  # nothing visible: neutral values, no error
    assert np.isfinite(f).all() and f[COL["tilt"]] == 0
