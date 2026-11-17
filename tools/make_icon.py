"""Builds marketing/icon.ico (the program, installer and shortcut icon) from marketing/logo_rgba.png.

Big sizes (64-256) use the whole round logo; small sizes (16-48) use the shield with the S, so it stays readable.
Run: python tools/make_icon.py
"""

from pathlib import Path

from PIL import Image, ImageDraw

M = Path(__file__).resolve().parents[1] / "marketing"


def square(im: Image.Image) -> Image.Image:
    s = max(im.size)
    out = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    out.paste(im, ((s - im.width) // 2, (s - im.height) // 2), im)
    return out


def shield(sq: Image.Image) -> Image.Image:
    k = sq.width / 598  # crop box measured on the 598 px logo
    crop = sq.crop(tuple(round(v * k) for v in (165, 70, 435, 340))).resize((256, 256), Image.LANCZOS)
    mask = Image.new("L", (256, 256), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, 255, 255), fill=255)
    out = Image.new("RGBA", (256, 256), (20, 8, 40, 255))
    out.paste(crop, (0, 0), crop)
    out.putalpha(mask)
    return out


def main() -> None:
    sq = square(Image.open(M / "logo_rgba.png").convert("RGBA"))
    small = shield(sq)
    frames = [small.resize((z, z), Image.LANCZOS) for z in (16, 24, 32, 48)]
    frames += [sq.resize((z, z), Image.LANCZOS) for z in (64, 128, 256)]
    big = frames[-1]
    big.save(M / "icon.ico", format="ICO", sizes=[f.size for f in frames], append_images=frames[:-1])
    small.resize((64, 64), Image.LANCZOS).save(M / "icon_small.png")
    print("icon.ico", (M / "icon.ico").stat().st_size, "bytes")


if __name__ == "__main__":
    main()
