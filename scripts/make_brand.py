"""Generate the integration's brand images (custom_components/lmstudio/brand)."""

from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "custom_components" / "lmstudio" / "brand"
S = 1024  # draw large, then downscale for smooth edges


def gradient() -> Image.Image:
    top, bottom = (79, 70, 229), (139, 92, 246)  # indigo -> violet
    img = Image.new("RGB", (S, S))
    px = img.load()
    for y in range(S):
        t = y / (S - 1)
        row = tuple(round(a + (b - a) * t) for a, b in zip(top, bottom, strict=True))
        for x in range(S):
            px[x, y] = row
    return img


def render() -> Image.Image:
    base = gradient().convert("RGBA")
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, S - 1, S - 1), radius=230, fill=255)
    base.putalpha(mask)

    d = ImageDraw.Draw(base)
    white = (255, 255, 255, 255)
    soft = (255, 255, 255, 150)
    # three stacked bars: a context window being filled, with a status dot
    bars = [(200, 250, 824), (200, 450, 700), (200, 650, 540)]
    for i, (x0, y, x1) in enumerate(bars):
        d.rounded_rectangle((x0, y, x1, y + 120), radius=60, fill=white if i == 0 else soft)
    d.ellipse((730, 650, 850, 770), fill=(110, 231, 183, 255))  # green "online" dot
    return base


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    big = render()
    big.resize((256, 256), Image.LANCZOS).save(OUT / "icon.png", optimize=True)
    big.resize((512, 512), Image.LANCZOS).save(OUT / "icon@2x.png", optimize=True)
    big.resize((256, 256), Image.LANCZOS).save(OUT / "logo.png", optimize=True)
    big.resize((512, 512), Image.LANCZOS).save(OUT / "logo@2x.png", optimize=True)
