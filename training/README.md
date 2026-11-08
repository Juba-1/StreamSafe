# training/ - build the eye's heads and the scene head

The eye (C-RADIOv4-H) turns each person crop into an embedding. Two small logistic heads on that embedding
decide who is covered (`common.py` explains the recipe). The engine reads the heads from `HEAD_DIR`:
`fast_h6.txt`, `male_h6.txt`, `thr_h6.txt` (and `scene_h.txt` for the scene head).

Run from the repository root as modules (`python -m training.<name> ...`); each file's docstring has its usage.

| Step | Script | What it does |
|---|---|---|
| shared | `common.py` | game families, the head recipe, LOFO scoring, writing head files. Every script below uses it |
| 1. data | `build_labeled.py` | embed every labeled game crop of the earlier rounds -> `labeled.npz` |
| 1. data | `mine_hard.py` | find the people the eye is least sure about in new videos -> crops for Ali to label |
| 1. data | `refind.py` | find labeled mined people again, with full-resolution embeddings and pose |
| 1. data | `add_examples.py` | turn a moment of a video into examples with one label (e.g. "Lucia in prison is modest") |
| 1. data | `oi_photos.py` | real photos (Open Images) as extra examples; drawings and images with children excluded |
| 2. train | `compare_heads.py` | the production path: train every recipe on the same data, score them alike, write each head set |
| 2. train | `train_heads.py` | the original single-recipe trainer (runs 16-21) |
| 2. train | `calibrate.py` | fair thresholds for the run 39-40 recipes |
| 2. train | `train_scene.py` | the explicit-scene head on labeled game frames |

Rules:
- A held-out test split is never trained on. Examples from `add_examples.py` (`EXTRA=`) go into training only.
- New heads are adopted only after the disagreement sheets (`bench/disagree.py`) and cover metrics show no new misses.
- No children in labeling batches. No leaked footage. Research-only data never goes into product heads.

Current product heads: run 37 `ali` (`runs/run37_ali_labels_photos/heads/ali_labels`, threshold 0.2104).
