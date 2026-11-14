"""Fixed rules for women the eye reads as modest: big back view, hips close-up, bare belly (Ali, 14 Nov 2026)."""

import numpy as np

from streamsafe.analysis import body_rule as br

W, H = 960, 540


def kps(front=True, face=0.9, top=0.9, hips=0.9, scale=1.0, cx=480.0, y0=60.0):
    """A standing person: shoulders at y0+60, hips at y0+60+150*scale."""
    kp = np.zeros((17, 2), np.float32)
    sc = np.full(17, 0.9, np.float32)
    sw, hw = 50 * scale, 40 * scale
    sy, hy = y0 + 60 * scale, y0 + (60 + 150) * scale
    sign = 1 if front else -1  # front: her left shoulder on the image's right
    kp[br.NOSE] = (cx, y0 + 20 * scale)
    kp[br.LEYE], kp[br.REYE] = (cx + sign * 8, y0 + 14 * scale), (cx - sign * 8, y0 + 14 * scale)
    kp[br.LSHO], kp[br.RSHO] = (cx + sign * sw, sy), (cx - sign * sw, sy)
    kp[br.LHIP], kp[br.RHIP] = (cx + sign * hw, hy), (cx - sign * hw, hy)
    sc[[br.NOSE, br.LEYE, br.REYE]] = face
    sc[[br.LSHO, br.RSHO]] = top
    sc[[br.LHIP, br.RHIP]] = hips
    return kp, sc


def test_big_back_view_is_covered_small_or_front_is_not():
    kp, sc = kps(front=False, face=0.1, scale=1.6)
    assert br.big_back(kp, sc, (380, 40, 580, 520), (W, H))
    kp, sc = kps(front=False, face=0.1, scale=0.5)  # far away
    assert not br.big_back(kp, sc, (450, 40, 510, 200), (W, H))
    kp, sc = kps(front=True, face=0.9, scale=1.6)  # facing the camera
    assert not br.big_back(kp, sc, (380, 40, 580, 520), (W, H))


def test_turned_body_counts_as_back_even_when_the_face_half_shows():
    kp, sc = kps(front=False, face=0.5, scale=1.6)
    assert br.back_view(kp, sc)


def test_hips_close_up_without_head_or_shoulders():
    kp, sc = kps(face=0.05, top=0.05, scale=3.0)
    assert br.hips_close(kp, sc, (W, H))
    kp, sc = kps(face=0.9, top=0.9, scale=3.0)
    assert not br.hips_close(kp, sc, (W, H))


def _picture(belly_bgr):
    img = np.full((H, W, 3), (40, 120, 40), np.uint8)
    kp, sc = kps(scale=1.5)
    face = (60, 110, 190)  # skin (BGR)
    nx, ny = kp[br.NOSE].astype(int)
    img[ny - 20 : ny + 20, nx - 20 : nx + 20] = face
    sy, hy = int(kp[br.LSHO, 1]), int(kp[br.LHIP, 1])
    img[sy:hy, 400:560] = (150, 40, 40)  # top
    img[int(sy + 0.6 * (hy - sy)) : hy, 400:560] = belly_bgr
    mask = np.zeros((H, W), bool)
    mask[40:520, 380:580] = True
    return img, mask, kp, sc


def test_bare_belly_matches_the_face_skin():
    img, mask, kp, sc = _picture((60, 110, 190))  # skin shows
    assert br.bare_midriff(img, mask, kp, sc)
    img, mask, kp, sc = _picture((150, 40, 40))  # the top reaches the hips
    assert not br.bare_midriff(img, mask, kp, sc)


def test_dark_skin_tone_works_the_same():
    img, mask, kp, sc = _picture((40, 60, 95))
    nx, ny = kp[br.NOSE].astype(int)
    img[ny - 20 : ny + 20, nx - 20 : nx + 20] = (40, 60, 95)
    assert br.bare_midriff(img, mask, kp, sc)


def test_hit_names_the_rule():
    img, mask, kp, sc = _picture((60, 110, 190))
    assert br.hit(img, mask, (380, 40, 580, 520), kp, sc) == "midriff"
    img, mask, kp, sc = _picture((150, 40, 40))
    assert br.hit(img, mask, (380, 40, 580, 520), kp, sc) == ""
