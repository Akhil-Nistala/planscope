"""Compose a full sheet from components: border, title block, then one of three layouts.

* ``plan_side``      -- floor plan + a column of schedules / notes / legends (column right, or left when mirrored)
* ``schedule_sheet`` -- no plan; 1-4 columns filled with schedules, notes and legends
* ``plan_only``      -- plan across the sheet, a band of legend / notes along the bottom
"""

from __future__ import annotations

from .components import Component, Ctx, FloorPlan, Legend, Maker, NotesBlock, Region, ScheduleTable, TitleBlock

SHEET_SIZES = {"A3": (1190.55, 841.89), "A2": (1683.78, 1190.55), "A1": (2383.94, 1683.78)}  # landscape, pt
LAYOUTS = ("plan_side", "schedule_sheet", "plan_only")
TITLE_BLOCK_STYLES = ("box", "strip")

Box = tuple[float, float, float, float]


def _makers(ctx: Ctx, kinds: list[str]) -> list[Maker]:
    table = lambda w, h: ScheduleTable(ctx, w, h)  # noqa: E731
    notes = lambda w, h: NotesBlock(ctx, w, h)  # noqa: E731
    legend = lambda w, h: Legend(ctx, w, h)  # noqa: E731
    return [{"table": table, "notes": notes, "legend": legend}[k] for k in kinds]


def _fits(comp: Component, w: float, h: float) -> bool:
    return comp.ok and comp.w <= w + 1e-6 and comp.h <= h + 1e-6


def stack(ctx: Ctx, col: Box, makers: list[Maker], gap: float, max_items: int = 6) -> list[Region]:
    """Place components top to bottom in ``col`` while they fit."""
    x0, y0, x1, y1 = col
    y, placed, regions = y0, 0, []
    for make in makers:
        avail_h, avail_w = y1 - y, x1 - x0
        if placed >= max_items or avail_h < 30 or avail_w < 60:
            break
        comp = make(avail_w, avail_h)
        if not _fits(comp, avail_w, avail_h):
            continue
        x = x0 + (ctx.u(0, 0.3) * (avail_w - comp.w) if ctx.chance(0.3) else 0.0)
        regions += comp.draw(x, y)
        y += comp.h + gap
        placed += 1
    return regions


def hstack(ctx: Ctx, band: Box, makers: list[Maker], gap: float) -> list[Region]:
    """Place components left to right in ``band`` while they fit."""
    x0, y0, x1, y1 = band
    x, regions = x0, []
    for make in makers:
        avail_w, avail_h = x1 - x, y1 - y0
        if avail_w < 60 or avail_h < 30:
            break
        comp = make(avail_w, avail_h)
        if not _fits(comp, avail_w, avail_h):
            continue
        regions += comp.draw(x, y0)
        x += comp.w + gap
    return regions


def _zone_marks(ctx: Ctx, m: float) -> None:
    """Grid-reference letters/numbers in the sheet margin (unlabelled negatives)."""
    cv, s = ctx.cv, min(6.0, m * 0.45)
    nx, ny = int(cv.W // 150), int(cv.H // 150)
    for i in range(nx):
        x = m + (cv.W - 2 * m) * (i + 0.5) / nx
        lab = str(i + 1)
        for y in (m * 0.5 + s * 0.35, cv.H - m * 0.5 + s * 0.35):
            cv.text(x - cv.width(lab, s) / 2, y, lab, s)
    for j in range(ny):
        y = m + (cv.H - 2 * m) * (j + 0.5) / ny + s * 0.35
        lab = chr(ord("A") + j)
        for x in (m * 0.5, cv.W - m * 0.5):
            cv.text(x - cv.width(lab, s) / 2, y, lab, s)


def compose(ctx: Ctx, layout: str, tb_style: str) -> list[Region]:
    """Draw a whole sheet on ``ctx.cv`` and return its labelled regions."""
    cv = ctx.cv
    W, H = cv.W, cv.H
    m = ctx.u(14, 28)
    cv.rect((m, m, W - m, H - m), w=ctx.u(0.8, 1.6))
    if ctx.chance(0.5):
        _zone_marks(ctx, m)
    g = ctx.u(6, 14)
    inner = (m + g, m + g, W - m - g, H - m - g)
    L, T, R, B = inner

    tb = TitleBlock(ctx, tb_style, inner)
    regions = tb.draw()
    tx0, ty0, tx1, _ = tb.rect
    gap = ctx.size * ctx.u(2.5, 5.0)
    right = tx0 - gap if tb_style == "strip" else R

    def bottom_for(a: float, b: float) -> float:
        """Lowest y available for a column spanning [a, b] (stops above a box title block)."""
        return ty0 - gap if tb_style == "box" and a < tx1 and b > tx0 else B

    if layout == "plan_side":
        frac = ctx.u(0.55, 0.7)
        kinds = ["table", "notes"] + (["legend"] if ctx.chance(0.7) else [])
        if not ctx.chance(0.6):
            ctx.rng.shuffle(kinds)
        kinds += [ctx.pick(["table", "notes", "legend"]) for _ in range(4)]
        if ctx.chance(0.25):  # mirrored: column on the left
            cx1 = L + (right - L) * (1 - frac)
            col = (L, T, cx1, bottom_for(L, cx1))
            plan = (cx1 + gap, T, right, bottom_for(cx1 + gap, right))
        else:
            split = L + (right - L) * frac
            plan = (L, T, split, bottom_for(L, split))
            col = (split + gap, T, right, bottom_for(split + gap, right))
        regions += FloorPlan(ctx, plan).draw()
        regions += stack(ctx, col, _makers(ctx, kinds), gap)
    elif layout == "schedule_sheet":
        ncol = max(1, round((right - L) / ctx.u(320, 480)))
        cw = (right - L - gap * (ncol - 1)) / ncol
        for c in range(ncol):
            a = L + c * (cw + gap)
            kinds = ["table"] + [str(k) for k in ctx.rng.choice(["table", "notes", "legend"], 5, p=[0.6, 0.25, 0.15])]
            regions += stack(ctx, (a, T, a + cw, bottom_for(a, a + cw)), _makers(ctx, kinds), gap)
    elif layout == "plan_only":
        band_top = ty0 if tb_style == "box" else B - (B - T) * ctx.u(0.16, 0.24)
        band_right = tx0 - gap if tb_style == "box" else right
        regions += FloorPlan(ctx, (L, T, right, band_top - gap)).draw()
        kinds = ["legend", "notes"] if ctx.chance(0.5) else ["notes", "legend"]
        regions += hstack(ctx, (L, band_top, band_right, B), _makers(ctx, kinds + ["table", "notes"]), gap)
    else:
        raise ValueError(f"unknown layout {layout!r}")
    return regions
