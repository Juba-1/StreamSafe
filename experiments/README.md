# experiments/ - one-off research, kept for the record

Each script answered one question for one run. The engine does not use them. They stay so that the numbers in
the project findings log can be reproduced. Do not build on them. Move anything useful into `training/` or `bench/`.

| Script | Run | Question | Outcome |
|---|---|---|---|
| `sweep_heads.py` | after 30 | How should a hard-mining round be weighted in head training? | Fed into the `compare_heads` recipes |
| `hires_eye.py` | 36 | Does the eye judge small people better from full-resolution crops? | Yes: `full_res_crops=True` (women 63->77%) |
| `pose_eval.py` | 39 | Do Ali's review32 labels and posture features improve the heads? | Posture rule not adopted. Labels kept |
| `pose_speed.py` | 39 | What does posture cost at run time? | Measured, see findings log |
| `scene_probe.py` | 41 | What do the scene model and HUD reader say on real clips? | Zero-shot explicit score too noisy. Trained scene head required |
