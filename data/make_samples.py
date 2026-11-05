"""Generate a handful of deterministic SAMPLE drawings for smoke tests and the demo site.

These are NOT the training set (that is Milestone 2's synthetic generator). They
exist to exercise every router path:

* ``jp_finish_schedule_vector.pdf`` -- A3 digital sheet, Japanese 仕上表 + 特記仕様書 (VECTOR)
* ``en_finish_schedule_vector.pdf`` -- A1 digital sheet, English schedule (VECTOR)
* ``jp_scanned.pdf``                -- the JP sheet printed & scanned: image-only PDF (RASTER)
* ``jp_scanned.png``                -- same scan as a PNG with 200-DPI metadata (RASTER, image input)
* ``cad_outlined_text.pdf``         -- CAD-style export where annotation text was
                                       vectorised to strokes; only the sheet no. survives (RASTER)
* ``mixed_set.pdf``                 -- 3 pages: vector JP, scan, outlined-text (per-page routing)

Usage::

    python data/make_samples.py --out data/samples --seed 42
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import cv2
import pymupdf as fitz
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
VOCAB_PATH = ROOT / "configs" / "vocab.yaml"

A3 = (1190.55, 841.89)   # landscape, points
A1 = (2383.94, 1683.78)
INK = (0, 0, 0)
GREY = (0.45, 0.45, 0.45)
LIGHT = (0.75, 0.75, 0.75)


def load_vocab() -> dict[str, Any]:
    """Load the shared vocabulary YAML."""
    return yaml.safe_load(VOCAB_PATH.read_text(encoding="utf-8"))


# Proportional Japanese fonts, first match wins. PyMuPDF's built-in "japan" font is the last
# resort: it sets Latin characters full-width ("t 1 2"), which splits words on extraction.
JP_FONT_CANDIDATES = [
    os.environ.get("PA_JP_FONT", ""),
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",       # Debian/Ubuntu/Colab (fonts-noto-cjk)
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",            # Arch/Fedora
    "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",                  # macOS
    "C:/Windows/Fonts/YuGothR.ttc",
    "C:/Windows/Fonts/msgothic.ttc",
]


def find_jp_font() -> str | None:
    """Return the first existing Japanese font file, or None (fall back to built-in)."""
    return next((p for p in JP_FONT_CANDIDATES if p and Path(p).exists()), None)


class Sheet:
    """Small drawing helper around a PyMuPDF page, with an 'outline text' mode."""

    def __init__(self, page: fitz.Page, lang: str, rng: np.random.Generator, outline_text: bool = False):
        self.page, self.lang, self.rng, self.outline = page, lang, rng, outline_text
        jp_file = find_jp_font() if lang == "ja" else None
        if jp_file:
            page.insert_font(fontname="jpfont", fontfile=jp_file)
            self.font, self._font_obj = "jpfont", fitz.Font(fontfile=jp_file)
        else:
            self.font = "japan" if lang == "ja" else "helv"
            self._font_obj = fitz.Font("japan" if lang == "ja" else "helv")

    def width(self, s: str, size: float) -> float:
        """Advance width of ``s`` in points."""
        return self._font_obj.text_length(s, fontsize=size)

    # -- primitives ---------------------------------------------------------
    def line(self, p0, p1, w: float = 0.5, color=INK, dashes: str | None = None) -> None:
        self.page.draw_line(fitz.Point(*p0), fitz.Point(*p1), color=color, width=w, dashes=dashes)

    def rect(self, r, w: float = 0.6, color=INK, fill=None) -> None:
        self.page.draw_rect(fitz.Rect(*r), color=color, width=w, fill=fill)

    def text(self, pos, s: str, size: float = 7.0, color=INK, rotate: int = 0, force_real: bool = False) -> None:
        """Insert real text, or (outline mode) strokes that look like text but have no text layer."""
        if self.outline and not force_real:
            x, y = pos
            for c in s:
                adv = size if ord(c) > 0x2E80 else 0.55 * size
                if not c.isspace():
                    for _ in range(3):  # 3 random strokes per glyph
                        x0, y0 = x + self.rng.uniform(0, adv * 0.8), y - self.rng.uniform(0, size * 0.8)
                        x1, y1 = x + self.rng.uniform(0, adv * 0.8), y - self.rng.uniform(0, size * 0.8)
                        self.line((x0, y0), (x1, y1), w=0.35, color=color)
                x += adv
            return
        self.page.insert_text(fitz.Point(*pos), s, fontname=self.font, fontsize=size, color=color, rotate=rotate)

    # -- drawing content ----------------------------------------------------
    def border(self) -> None:
        W, H = self.page.rect.width, self.page.rect.height
        self.rect((20, 20, W - 20, H - 20), w=1.2)

    def floor_plan(self, area: tuple[float, float, float, float], rooms: list[str]) -> None:
        """Grid, walls, doors, dimensions, hatching and room-name callouts with leaders."""
        x0, y0, x1, y1 = area
        nx, ny = 5, 3
        xs = np.linspace(x0 + 40, x1 - 20, nx)
        ys = np.linspace(y0 + 40, y1 - 40, ny)
        for i, x in enumerate(xs):  # grid lines + bubbles
            self.line((x, y0 + 10), (x, y1), w=0.3, color=GREY, dashes="[6 2 1 2] 0")
            self.page.draw_circle(fitz.Point(x, y0 + 10), 7, color=INK, width=0.5)
            self.text((x - 3, y0 + 13), f"X{i + 1}", size=6, force_real=True)
        for j, y in enumerate(ys):
            self.line((x0 + 10, y), (x1, y), w=0.3, color=GREY, dashes="[6 2 1 2] 0")
            self.page.draw_circle(fitz.Point(x0 + 10, y), 7, color=INK, width=0.5)
            self.text((x0 + 6, y + 3), f"Y{j + 1}", size=6, force_real=True)
        # walls: double lines around each bay
        k = 0
        for i in range(nx - 1):
            for j in range(ny - 1):
                r = (xs[i] + 3, ys[j] + 3, xs[i + 1] - 3, ys[j + 1] - 3)
                self.rect(r, w=1.6)
                self.rect((r[0] + 4, r[1] + 4, r[2] - 4, r[3] - 4), w=0.4)
                # door swing
                dx, dy = r[0] + 15, r[3] - 4
                self.page.draw_curve(fitz.Point(dx, dy - 22), fitz.Point(dx + 22, dy - 22), fitz.Point(dx + 22, dy),
                                     color=GREY, width=0.4)
                self.line((dx, dy), (dx, dy - 22), w=0.6)
                # room name + leader callout
                cx, cy = (r[0] + r[2]) / 2, (r[1] + r[3]) / 2
                name = rooms[k % len(rooms)]
                self.text((cx - self.width(name, 8) / 2, cy), name, size=8)
                self.text((cx - 10, cy + 11), f"{self.rng.integers(12, 80)}.{self.rng.integers(0, 9)}㎡"
                          if self.lang == "ja" else f"{self.rng.integers(12, 80)} m2", size=6, color=GREY)
                if k % 3 == 1:  # hatched area (e.g. wet room)
                    for t in np.arange(r[0] + 6, r[2] - 6, 7):
                        self.line((t, r[3] - 6), (min(t + 25, r[2] - 6), r[3] - 31), w=0.25, color=LIGHT)
                k += 1
        # dimension line along the bottom
        yd = y1 - 8
        self.line((xs[0], yd), (xs[-1], yd), w=0.4)
        for a, b in zip(xs[:-1], xs[1:]):
            self.line((a - 3, yd + 3), (a + 3, yd - 3), w=0.6)
            self.text(((a + b) / 2 - 10, yd - 3), f"{int(self.rng.choice([6000, 7200, 8000, 9000]))}", size=6)
        self.line((xs[-1] - 3, yd + 3), (xs[-1] + 3, yd - 3), w=0.6)
        # a 90-degree rotated callout
        self.text((x1 - 6, y0 + 160), "EXP.J" if self.lang == "en" else "エキスパンションジョイント", size=6, rotate=90)

    def finish_schedule(self, origin: tuple[float, float], vocab: dict, n_rows: int, size: float = 6.5,
                        max_w: float = 1e9) -> fitz.Rect:
        """Draw a finish schedule (仕上表) table no wider than ``max_w``; returns its rect."""
        L = self.lang
        h = vocab["headers"][L]
        cols = ["room", "floor", "baseboard", "wall", "ceiling"]
        rooms = list(self.rng.permutation(vocab["rooms"][L]))[:n_rows]
        values = {c: rooms if c == "room" else [str(self.rng.choice(vocab["materials"][c][L])) for _ in rooms]
                  for c in cols}
        pad = 4.0
        # size columns to their content; shrink the font if the table would not fit
        natural = [max(self.width(s, size) for s in [h[c], *values[c]]) + 2 * pad for c in cols]
        if sum(natural) > max_w:
            size *= max_w / sum(natural)
            natural = [w * max_w / sum(natural) for w in natural]
        widths = natural
        row_h = size * 2.2
        x0, y0 = origin
        title = "仕上表" if L == "ja" else "FINISH SCHEDULE"
        self.text((x0, y0 - 5), title, size=size * 1.5)
        total_w = sum(widths)
        total_h = row_h * (n_rows + 1)
        self.rect((x0, y0, x0 + total_w, y0 + row_h), w=0.0, color=None, fill=(0.9, 0.9, 0.9))
        for r in range(n_rows + 2):
            self.line((x0, y0 + r * row_h), (x0 + total_w, y0 + r * row_h), w=0.9 if r in (0, 1, n_rows + 1) else 0.4)
        x = x0
        for c, w in enumerate(widths):
            self.line((x, y0), (x, y0 + total_h), w=0.9 if c == 0 else 0.4)
            x += w
        self.line((x, y0), (x, y0 + total_h), w=0.9)
        # header + body
        x = x0
        for c, w in zip(cols, widths):
            self.text((x + pad, y0 + row_h * 0.68), h[c], size=size)
            for r, val in enumerate(values[c]):
                self.text((x + pad, y0 + row_h * (r + 1.68)), val, size=size * 0.92)
            x += w
        return fitz.Rect(x0, y0, x0 + total_w, y0 + total_h)

    def notes(self, origin: tuple[float, float], vocab: dict, size: float = 6.5) -> None:
        x0, y0 = origin
        self.text((x0, y0), "特記仕様書" if self.lang == "ja" else "GENERAL NOTES", size=size * 1.4)
        for i, n in enumerate(vocab["notes"][self.lang]):
            self.text((x0, y0 + (i + 1) * size * 1.9), f"{i + 1}. {n}", size=size)

    def legend(self, origin: tuple[float, float], size: float = 6.5) -> None:
        x0, y0 = origin
        items = [("凡例", None)] if self.lang == "ja" else [("LEGEND", None)]
        labels = (["石綿含有調査箇所", "撤去範囲", "既存壁"] if self.lang == "ja"
                  else ["survey location", "removal area", "existing wall"])
        self.text((x0, y0), items[0][0], size=size * 1.3)
        for i, lab in enumerate(labels):
            y = y0 + (i + 1) * size * 2.2
            if i == 0:
                self.page.draw_circle(fitz.Point(x0 + 5, y - 2), 3.5, color=INK, width=0.6)
            elif i == 1:
                self.rect((x0, y - 6, x0 + 10, y + 1), w=0.4, fill=(0.8, 0.8, 0.8))
            else:
                self.line((x0, y - 2), (x0 + 10, y - 2), w=1.6)
            self.text((x0 + 16, y), lab, size=size)

    def title_block(self, size: float = 7.0) -> None:
        W, H = self.page.rect.width, self.page.rect.height
        bw, bh = 260 * size / 7, 110 * size / 7
        x0, y0 = W - 20 - bw, H - 20 - bh
        self.rect((x0, y0, W - 20, H - 20), w=1.0)
        rows = (["工事名称  ○○ビル改修工事", "図面名称  仕上表・1階平面図", "縮尺  1/100", "日付  2026.11.05",
                 "図面番号  A-101", "設計  △△設計事務所"] if self.lang == "ja"
                else ["PROJECT  Office Refurbishment", "TITLE  Finish Schedule / 1F Plan", "SCALE  1:100",
                      "DATE  2026-11-05", "DWG No.  A-101", "ARCHITECT  Example Studio"])
        rh = bh / len(rows)
        for i, s in enumerate(rows):
            if i:
                self.line((x0, y0 + i * rh), (W - 20, y0 + i * rh), w=0.4)
            # the drawing number stays real text even in outline mode (typical of CAD exports)
            self.text((x0 + 6, y0 + i * rh + rh * 0.68), s, size=size, force_real=s.startswith(("図面番号", "DWG")))


def draw_sheet(doc: fitz.Document, lang: str, size: tuple[float, float], seed: int, outline: bool = False) -> None:
    """Append one complete drawing sheet to ``doc``."""
    vocab = load_vocab()
    rng = np.random.default_rng(seed)
    W, H = size
    page = doc.new_page(width=W, height=H)
    s = Sheet(page, lang, rng, outline_text=outline)
    k = W / A3[0]  # scale content for larger sheets
    s.border()
    s.floor_plan((30 * k, 30 * k, W * 0.58, H - 40 * k), vocab["rooms"][lang])
    s.finish_schedule((W * 0.61, 60 * k), vocab, n_rows=int(rng.integers(6, 10)), size=6.5 * k,
                     max_w=W - 30 * k - W * 0.61)
    s.notes((W * 0.61, H * 0.52), vocab, size=6.2 * k)
    s.legend((W * 0.61, H * 0.80), size=6.2 * k)
    s.title_block(size=7 * k)


def scan_effect(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simulate print+scan: greyscale, slight skew, blur, faded ink, noise, JPEG artefacts."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = g.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), float(rng.uniform(-0.8, 0.8)), 1.0)
    g = cv2.warpAffine(g, M, (w, h), borderValue=255)
    g = cv2.GaussianBlur(g, (3, 3), 0.8)
    g = (g.astype(np.float32) * 0.8 + 255 * 0.2)  # fade ink / grey paper
    g += rng.normal(0, 9, g.shape)
    g = np.clip(g, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", g, [cv2.IMWRITE_JPEG_QUALITY, 55])
    assert ok
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2RGB)


def image_only_pdf(img: np.ndarray, size: tuple[float, float]) -> fitz.Document:
    """Wrap an image as a single-page PDF with no text layer (like a scanner output)."""
    doc = fitz.open()
    page = doc.new_page(width=size[0], height=size[1])
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
    assert ok
    page.insert_image(page.rect, stream=buf.tobytes())
    return doc


def main() -> None:
    """Write all samples to ``--out``."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "samples")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    jp = fitz.open(); draw_sheet(jp, "ja", A3, args.seed); jp.subset_fonts(); jp.save(out / "jp_finish_schedule_vector.pdf")
    en = fitz.open(); draw_sheet(en, "en", A1, args.seed + 1); en.save(out / "en_finish_schedule_vector.pdf")
    cad = fitz.open(); draw_sheet(cad, "ja", A3, args.seed + 2, outline=True); cad.subset_fonts(); cad.save(out / "cad_outlined_text.pdf")

    pix = jp[0].get_pixmap(dpi=200, colorspace=fitz.csRGB, alpha=False)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
    scan = scan_effect(img, rng)
    image_only_pdf(scan, A3).save(out / "jp_scanned.pdf")
    from PIL import Image
    Image.fromarray(scan).save(out / "jp_scanned.png", dpi=(200, 200))

    mixed = fitz.open()
    for src in ("jp_finish_schedule_vector.pdf", "jp_scanned.pdf", "cad_outlined_text.pdf"):
        with fitz.open(out / src) as d:
            mixed.insert_pdf(d)
    mixed.save(out / "mixed_set.pdf")

    # small PNG previews for the README / demo site
    prev = out / "previews"
    prev.mkdir(exist_ok=True)
    for f in sorted(out.glob("*.pdf")):
        with fitz.open(f) as d:
            d[0].get_pixmap(dpi=40 if d[0].rect.width < 2000 else 20).save(prev / f"{f.stem}.png")
    print(f"wrote samples to {out}")


if __name__ == "__main__":
    main()
