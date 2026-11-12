"""Does this Python's Tk draw Arabic correctly (letters joined, right to left)? ``python tools/ci_tk_arabic.py OUT_DIR``.

The app window is tkinter and Arabic is its default language, so this is checked on the real Windows runner, not
assumed: the labels are drawn in a real window, the screen is captured and three facts are measured.

  * shaping: "محمي" must not look like its four letters kept apart (zero-width non-joiners between them);
  * order:   in "ابببب" the tall alef, the FIRST letter, must be at the RIGHT end;
  * mixed:   a line with Arabic and Latin words is drawn (saved for the eye, not measured).

Exit code 0 = both measured facts hold. The captured pictures are written to OUT_DIR for a human to look at.
"""

from __future__ import annotations

import json
import sys
import tkinter as tk
from pathlib import Path

from PIL import ImageGrab

FONT = ("Segoe UI", 40)
JOINED = "محمي"
APART = "\u200c".join(JOINED)
ORDER = "ابببب"
MIXED = "StreamSafe: محمي / مغطى بالكامل 100%"


def ink_columns(image) -> list[int]:
    """Dark pixels per column, cut to the columns that have ink."""
    grey = image.convert("L")
    w, h = grey.size
    px = grey.load()
    cols = [sum(1 for y in range(h) if px[x, y] < 128) for x in range(w)]
    inked = [i for i, c in enumerate(cols) if c]
    return cols[inked[0] : inked[-1] + 1] if inked else []


def grab(widget: tk.Widget):
    widget.update()
    x, y = widget.winfo_rootx(), widget.winfo_rooty()
    return ImageGrab.grab(bbox=(x, y, x + widget.winfo_width(), y + widget.winfo_height()))


def main(argv: list[str]) -> int:
    out = Path(argv[0] if argv else ".")
    out.mkdir(parents=True, exist_ok=True)
    root = tk.Tk()
    root.title("Arabic check")
    root.configure(bg="white")
    root.attributes("-topmost", True)
    labels = {}
    for name, text in (("joined", JOINED), ("apart", APART), ("order", ORDER), ("mixed", MIXED)):
        labels[name] = tk.Label(root, text=text, font=FONT, bg="white", fg="black", padx=20, pady=10)
        labels[name].pack(anchor="e")
    root.update()
    root.after(700, root.quit)  # let the window manager really put it on screen
    root.mainloop()
    shots = {name: grab(label) for name, label in labels.items()}
    for name, shot in shots.items():
        shot.save(out / f"tk_arabic_{name}.png")
    grab(root).save(out / "tk_arabic_all.png")
    joined, apart, order = (ink_columns(shots[n]) for n in ("joined", "apart", "order"))
    tallest = max(range(len(order)), key=order.__getitem__) / max(1, len(order) - 1) if order else -1.0
    result = dict(
        tk=root.tk.call("info", "patchlevel"),
        platform=sys.platform,
        joined_width=len(joined),
        apart_width=len(apart),
        shaping=bool(joined) and joined != apart,
        alef_position=round(tallest, 2),  # 1.0 = right end
        right_to_left=tallest > 0.7,
    )
    root.destroy()
    (out / "tk_arabic.txt").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result))
    return 0 if result["shaping"] and result["right_to_left"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
