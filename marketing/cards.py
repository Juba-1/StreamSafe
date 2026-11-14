"""Draws the promo intro / outro cards and the corner logo from the brand logo (one-off asset script).

    python marketing/cards.py LOGO.png CAIRO.ttf MONTSERRAT.ttf   -> intro.png, outro.png, wm.png here
Fonts: Cairo and Montserrat (Google Fonts, OFL), variable weight. Arabic needs Pillow with raqm."""

import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1920, 1080


def font(path: str, size: int, weight: int) -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(path, size)
    axes = [weight] if len(f.get_variation_axes()) == 1 else [weight, 0]
    f.set_variation_by_axes(axes)
    return f


def direction(text: str) -> str:
    return "rtl" if any("؀" <= c <= "ۿ" for c in text) else "ltr"


def background() -> Image.Image:
    y, x = np.mgrid[0:H, 0:W]
    r = np.sqrt(((x - W / 2) / W) ** 2 + ((y - H * 0.45) / H) ** 2)
    g = np.clip(1 - r * 1.6, 0, 1) ** 2
    a = np.zeros((H, W, 3), np.float32)
    a[..., 0], a[..., 1], a[..., 2] = 40 * g, 6 * g, 60 * g
    return Image.fromarray(a.astype(np.uint8))


def glow_text(img, xy, text, f, fill, glow) -> None:
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(layer).text(xy, text, font=f, fill=(*glow, 255), anchor="mm", direction=direction(text))
    layer = layer.filter(ImageFilter.GaussianBlur(14))
    img.paste(layer, (0, 0), layer)
    img.paste(layer, (0, 0), layer)
    ImageDraw.Draw(img).text(xy, text, font=f, fill=fill, anchor="mm", direction=direction(text))


def logo_mask(logo: Image.Image) -> Image.Image:
    m = np.asarray(logo).max(axis=2).astype(np.float32)
    return Image.fromarray(np.clip((m - 8) * 4, 0, 255).astype(np.uint8))


def put_logo(img, logo, h, cy) -> None:
    small = logo.resize((int(logo.width * h / logo.height), h), Image.LANCZOS)
    img.paste(small, ((W - small.width) // 2, cy - h // 2), logo_mask(small))


def main(logo_path: str, cairo: str, mont: str) -> None:
    logo = Image.open(logo_path).convert("RGB")
    pink, blue, gold = (255, 40, 200), (40, 160, 255), (255, 120, 0)
    intro = background()
    put_logo(intro, logo, 430, 330)
    glow_text(intro, (W // 2, 690), "نفس التريلر… بعد ستريم سيف", font(cairo, 96, 800), (255, 255, 255), pink)
    glow_text(intro, (W // 2, 800), "THE SAME TRAILER — PROTECTED LIVE", font(mont, 46, 700), (200, 235, 255), blue)
    intro.save("intro.png")
    outro = background()
    put_logo(outro, logo, 560, 360)
    line1 = "يغطي المشاهد غير اللائقة قبل ما توصل لمشاهدينك"
    glow_text(outro, (W // 2, 745), line1, font(cairo, 64, 700), (255, 255, 255), pink)
    glow_text(outro, (W // 2, 840), "Live stream protection for GTA 6", font(mont, 44, 600), (200, 235, 255), blue)
    glow_text(outro, (W // 2, 940), "النسخة التجريبية للستريمرز — قريباً", font(cairo, 50, 700), (255, 210, 90), gold)
    outro.save("outro.png")
    small = logo.resize((int(logo.width * 150 / logo.height), 150), Image.LANCZOS)
    rgba = np.dstack([np.asarray(small), (np.asarray(logo_mask(small), np.float32) * 0.85).astype(np.uint8)])
    Image.fromarray(rgba).save("wm.png")


if __name__ == "__main__":
    main(*sys.argv[1:4])
