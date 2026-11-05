"""Drawing-sheet components. Each one lays itself out first (``w``, ``h``, ``ok``), then draws at a
position and returns its ground-truth :class:`Region` annotations (PDF points).

Labelling rules (what counts as each class):

* ``schedule_table`` -- the ruled grid of a finish schedule, header row included. The title above
  the grid ("仕上表") is not part of the region.
* ``notes_block``    -- heading + numbered notes (+ frame if drawn).
* ``legend``         -- heading + symbol rows (+ frame if drawn).
* ``title_block``    -- the whole title block frame, including any revision rows.
* ``text_callout``   -- room-name tags and leader callouts on plans: the text only, not the leader
  line. Oriented with the text (vertical / oblique callouts get rotated boxes).

Dimension figures, grid-bubble labels, zone markers, schedule titles and the north arrow are drawn
but deliberately left unlabelled; they are the detector's text-like negatives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .canvas import GREY, INK, LIGHT, Canvas, Poly, aabb, overlaps, rect_poly

Box = tuple[float, float, float, float]


# -- annotations -------------------------------------------------------------
@dataclass
class Cell:
    """Ground-truth table cell (PDF points)."""

    text: str
    bbox: Box
    row: int
    col: int
    row_span: int = 1
    col_span: int = 1
    is_header: bool = False


@dataclass
class Table:
    """Ground-truth table structure. ``rows[0]`` is the header row."""

    headers: list[str]
    canonical_headers: list[str | None]
    rows: list[list[Cell]]


@dataclass
class Region:
    """One labelled layout region (PDF points)."""

    cls: str
    polygon: Poly
    text: str
    table: Table | None = None


# -- shared context ----------------------------------------------------------
@dataclass
class Ctx:
    """Everything a component needs: the canvas, language, vocabulary, RNG and sheet text size."""

    cv: Canvas
    lang: str
    vocab: dict
    rng: np.random.Generator
    size: float
    angle_probs: dict[str, float] = field(default_factory=lambda: {"horizontal": 1.0})
    oblique_range: tuple[float, float] = (10.0, 60.0)

    def pick(self, seq):
        """Uniform choice that keeps Python types (np.random.choice would return numpy strings)."""
        return seq[int(self.rng.integers(len(seq)))]

    def chance(self, p: float) -> bool:
        """Bernoulli draw."""
        return bool(self.rng.random() < p)

    def u(self, a: float, b: float) -> float:
        """Uniform float."""
        return float(self.rng.uniform(a, b))

    def material(self, kind: str) -> str:
        """Random material name for a finish field."""
        return self.pick(self.vocab["materials"][kind][self.lang])

    def fill(self, template: str) -> str:
        """Fill a vocabulary template ({floor} {baseboard} {wall} {ceiling} {room} {n})."""
        return template.format(
            floor=self.material("floor"), baseboard=self.material("baseboard"), wall=self.material("wall"),
            ceiling=self.material("ceiling"), room=self.pick(self.vocab["rooms"][self.lang]),
            n=int(self.rng.integers(1, 40)),
        )

    def callout_angle(self) -> float:
        """Orientation of a leader callout, from the configured mix."""
        kinds = list(self.angle_probs)
        p = np.array([self.angle_probs[k] for k in kinds], dtype=float)
        kind = kinds[int(self.rng.choice(len(kinds), p=p / p.sum()))]
        if kind == "vertical":
            return 90.0
        if kind == "oblique":
            return float(self.rng.choice([-1, 1])) * self.u(*self.oblique_range)
        return 0.0


def wrap(cv: Canvas, text: str, size: float, max_w: float, lang: str) -> list[str]:
    """Greedy line wrap: per character for Japanese, per word for English."""
    if cv.width(text, size) <= max_w:
        return [text]
    units, joiner = (list(text), "") if lang == "ja" else (text.split(" "), " ")
    lines: list[str] = []
    cur = ""
    for unit in units:
        cand = unit if not cur else cur + joiner + unit
        # never start a Japanese line with closing punctuation
        if not cur or cv.width(cand, size) <= max_w or (lang == "ja" and unit in "。、）」・"):
            cur = cand
        else:
            lines.append(cur)
            cur = unit
    if cur:
        lines.append(cur)
    return lines


class Component:
    """Base: lay out in ``__init__`` (set ``ok``, ``w``, ``h``), draw with :meth:`draw`."""

    ok: bool = True
    w: float = 0.0
    h: float = 0.0

    def draw(self, x: float, y: float) -> list[Region]:
        """Draw with the top-left corner at (x, y)."""
        raise NotImplementedError


# -- finish schedule ---------------------------------------------------------
FINISH_FIELDS = ("floor", "baseboard", "wall", "ceiling")


class ScheduleTable(Component):
    """Finish schedule (仕上表): optional merged level column, room, 2-4 finish fields, optional remarks."""

    def __init__(self, ctx: Ctx, max_w: float, max_h: float):
        self.ctx = ctx
        cv, L, V, r = ctx.cv, ctx.lang, ctx.vocab, ctx.rng
        fields_ = [f for f in FINISH_FIELDS if ctx.chance(0.85)]
        if len(fields_) < 2:
            fields_ = list(FINISH_FIELDS)
        self.cols = (["level"] if ctx.chance(0.3) else []) + ["room"] + fields_ + (
            ["remarks"] if ctx.chance(0.4) else [])
        self.headers = [ctx.pick(V["header_variants"][L][c]) for c in self.cols]

        size = ctx.size * ctx.u(0.85, 1.1)
        self.title = ctx.pick(V["schedule_titles"][L]) if ctx.chance(0.85) else None
        self.title_size = size * ctx.u(1.3, 1.8)
        self.center = ctx.chance(0.3)
        self.header_fill = ctx.chance(0.5)
        self.vlines = ctx.chance(0.92)
        self.outer_w, self.inner_w = ctx.u(0.6, 1.4), ctx.u(0.2, 0.5)
        row_f = ctx.u(1.8, 2.6)
        head_f = row_f * ctx.u(1.0, 1.3)
        self.pad_f = ctx.u(0.5, 1.0)

        n = int(r.integers(3, 17))
        rooms = [str(s) for s in r.permutation(V["rooms"][L])]
        names = [rooms[i % len(rooms)] + ("" if i < len(rooms) else str(i // len(rooms) + 1)) for i in range(n)]
        self.values: dict[str, list[str]] = {}
        for c in self.cols:
            if c == "room":
                self.values[c] = names
            elif c == "remarks":
                self.values[c] = [ctx.pick(V["remarks"][L]) for _ in range(n)]
            elif c in FINISH_FIELDS:
                self.values[c] = [ctx.material(c) for _ in range(n)]
        self.groups: list[tuple[int, int, str]] = []
        if "level" in self.cols:
            g = int(r.integers(1, min(3, n) + 1))
            cuts = sorted(int(v) for v in r.choice(np.arange(1, n), g - 1, replace=False)) if g > 1 else []
            bounds = [0, *cuts, n]
            levels = V["levels"][L]
            start = int(r.integers(0, len(levels) - g + 1))
            self.groups = [(bounds[k], bounds[k + 1], levels[start + k]) for k in range(g)]

        def natural(sz: float) -> list[float]:
            out = []
            for c, h in zip(self.cols, self.headers):
                cells = [lab for _, _, lab in self.groups] if c == "level" else self.values[c]
                out.append(max(cv.width(s, sz) for s in [h, *cells]) + 2 * self.pad_f * sz)
            return out

        widths = natural(size)
        if sum(widths) > max_w:
            size *= max_w / sum(widths)
            widths = natural(size)
            if size < 3.5 or sum(widths) > max_w:
                self.ok = False
                return
        if self.title and cv.width(self.title, self.title_size) > max_w:
            self.title_size = max_w / cv.width(self.title, 1.0)
        self.size, self.widths = size, widths
        self.title_h = self.title_size * 1.9 if self.title else 0.0
        self.row_h, self.head_h = size * row_f, size * head_f

        max_rows = int((max_h - self.title_h - self.head_h) // self.row_h)
        if max_rows < 2:
            self.ok = False
            return
        self.n = min(n, max_rows)
        self.groups = [(s, min(e, self.n), lab) for s, e, lab in self.groups if s < self.n]
        self.w = sum(widths)
        self.h = self.title_h + self.head_h + self.n * self.row_h

    def _put(self, s: str, bb: Box, size: float) -> None:
        if not s:
            return
        cv = self.ctx.cv
        x = (bb[0] + bb[2] - cv.width(s, size)) / 2 if self.center else bb[0] + self.pad_f * self.size
        self.ctx.cv.text(x, cv.baseline_for(bb[1], bb[3], size), s, size)

    def draw(self, x: float, y: float) -> list[Region]:
        cv = self.ctx.cv
        if self.title:
            cv.text(x, y + self.title_size * 1.25, self.title, self.title_size)
        gy = y + self.title_h
        xs = [x + sum(self.widths[:i]) for i in range(len(self.widths) + 1)]
        ys = [gy, gy + self.head_h] + [gy + self.head_h + (i + 1) * self.row_h for i in range(self.n)]
        x1, y1 = xs[-1], ys[-1]
        if self.header_fill:
            cv.rect((x, gy, x1, ys[1]), w=0, color=None, fill=(0.88, 0.88, 0.88))

        has_level = "level" in self.cols
        group_starts = {s for s, _, _ in self.groups}
        for i, yy in enumerate(ys):
            outer = i in (0, 1, len(ys) - 1)
            # inside a merged level cell the row line starts after the level column
            x_start = xs[1] if (has_level and 2 <= i <= self.n and (i - 1) not in group_starts) else x
            cv.line((x_start, yy), (x1, yy), w=self.outer_w if outer else self.inner_w)
        for j, xx in enumerate(xs):
            edge = j in (0, len(xs) - 1)
            if edge or self.vlines or j == 1:
                cv.line((xx, gy), (xx, y1), w=self.outer_w if edge else self.inner_w)

        rows: list[list[Cell]] = [[]]
        for c, h in enumerate(self.headers):
            bb = (xs[c], ys[0], xs[c + 1], ys[1])
            self._put(h, bb, self.size)
            rows[0].append(Cell(h, bb, 0, c, is_header=True))
        body = self.size * 0.95
        for rr in range(self.n):
            row: list[Cell] = []
            for c, col in enumerate(self.cols):
                if col == "level":
                    grp = next((g for g in self.groups if g[0] == rr), None)
                    if grp is None:
                        continue  # covered by a merged cell above
                    bb = (xs[c], ys[rr + 1], xs[c + 1], ys[grp[1] + 1])
                    text, span = grp[2], grp[1] - grp[0]
                else:
                    bb = (xs[c], ys[rr + 1], xs[c + 1], ys[rr + 2])
                    text, span = self.values[col][rr], 1
                self._put(text, bb, body)
                row.append(Cell(text, bb, rr + 1, c, row_span=span))
            rows.append(row)

        text = "\n".join(" ".join(cell.text for cell in row if cell.text) for row in rows)
        table = Table(list(self.headers), list(self.cols), rows)
        return [Region("schedule_table", rect_poly(x, gy, x1, y1, pad=1.0), text, table)]


# -- notes -------------------------------------------------------------------
class NotesBlock(Component):
    """Heading + numbered, wrapped notes; optional frame."""

    def __init__(self, ctx: Ctx, max_w: float, max_h: float):
        self.ctx = ctx
        cv, L, V, r = ctx.cv, ctx.lang, ctx.vocab, ctx.rng
        size = ctx.size * ctx.u(0.9, 1.05)
        hsize = size * ctx.u(1.15, 1.5)
        self.bordered = ctx.chance(0.35)
        pad = size * ctx.u(0.7, 1.2) if self.bordered else 0.0
        col_w = min(max_w - 2 * pad, ctx.u(170, 380) * size / 7)
        heading = ctx.pick(V["notes_headings"][L])
        if col_w < 60 or cv.width(heading, hsize) > max_w - 2 * pad:
            self.ok = False
            return
        style = ctx.pick(["{i}.", "({i})", "・"] if L == "ja" else ["{i}.", "{i})", "-"])
        pool = list(V["notes"][L]) + [ctx.fill(t) for t in V["note_templates"][L]]
        items = [pool[int(i)] for i in r.permutation(len(pool))[: int(r.integers(3, 10))]]
        line_h = size * ctx.u(1.45, 1.9)

        y = pad + cv.asc(hsize)
        self.ops: list[tuple[float, float, str, float]] = [(pad, y, heading, hsize)]
        lines = [heading]
        y += hsize * ctx.u(1.4, 1.9)
        n_items = 0
        for i, item in enumerate(items, 1):
            prefix = style.format(i=i)
            indent = cv.width(prefix + " ", size)
            wrapped = wrap(cv, item, size, col_w - indent, L)
            ops = [(pad, y, prefix, size)] + [(pad + indent, y + k * line_h, ln, size) for k, ln in enumerate(wrapped)]
            last_baseline = y + (len(wrapped) - 1) * line_h
            if last_baseline + cv.desc(size) + pad > max_h:
                break
            self.ops += ops
            lines += [f"{prefix} {wrapped[0]}", *wrapped[1:]]
            y = last_baseline + line_h
            n_items += 1
        if n_items < 2:
            self.ok = False
            return
        self.text = "\n".join(lines)
        self.w = max(dx + cv.width(t, s) for dx, _, t, s in self.ops) + pad
        self.h = (y - line_h) + cv.desc(size) + pad

    def draw(self, x: float, y: float) -> list[Region]:
        cv = self.ctx.cv
        for dx, dy, t, s in self.ops:
            cv.text(x + dx, y + dy, t, s)
        if self.bordered:
            cv.rect((x, y, x + self.w, y + self.h), w=0.6)
        return [Region("notes_block", rect_poly(x, y, x + self.w, y + self.h, pad=0.5 if self.bordered else 1.5),
                       self.text)]


# -- legend ------------------------------------------------------------------
SYMBOLS = ("circle", "fill", "hatch", "thick", "dash", "triangle", "cross")


def draw_symbol(cv: Canvas, kind: str, box: Box) -> None:
    """Small legend symbol inside ``box``."""
    x0, y0, x1, y1 = box
    cx, cy, rad = (x0 + x1) / 2, (y0 + y1) / 2, min(x1 - x0, y1 - y0) / 2
    if kind == "circle":
        cv.circle((cx, cy), rad * 0.8, w=0.6)
    elif kind == "fill":
        cv.rect(box, w=0.4, fill=(0.75, 0.75, 0.75))
    elif kind == "hatch":
        cv.rect(box, w=0.4)
        step, hgt = (y1 - y0) / 2.5, y1 - y0
        for t in np.arange(x0 - hgt + step, x1, step):
            a = (max(t, x0), y1 - (max(t, x0) - t))
            b = (min(t + hgt, x1), y1 - (min(t + hgt, x1) - t))
            cv.line(a, b, w=0.3)
    elif kind == "thick":
        cv.line((x0, cy), (x1, cy), w=1.8)
    elif kind == "dash":
        cv.line((x0, cy), (x1, cy), w=0.8, dashes="[3 2] 0")
    elif kind == "triangle":
        cv.polyline([(cx, y0), (cx + rad, y1), (cx - rad, y1)], w=0.6, closed=True)
    else:  # cross
        cv.circle((cx, cy), rad * 0.8, w=0.5)
        cv.line((cx - rad * 0.8, cy), (cx + rad * 0.8, cy), w=0.5)
        cv.line((cx, cy - rad * 0.8), (cx, cy + rad * 0.8), w=0.5)


class Legend(Component):
    """Heading + rows of (symbol, label); optional frame."""

    def __init__(self, ctx: Ctx, max_w: float, max_h: float):
        self.ctx = ctx
        cv, L, V, r = ctx.cv, ctx.lang, ctx.vocab, ctx.rng
        self.size = size = ctx.size * ctx.u(0.9, 1.05)
        self.hsize = size * ctx.u(1.1, 1.4)
        self.bordered = ctx.chance(0.5)
        self.pad = pad = size * ctx.u(0.7, 1.2) if self.bordered else 0.0
        self.heading = ctx.pick(V["legend_headings"][L])
        labels = [str(s) for s in r.permutation(V["legend_labels"][L])[: int(r.integers(3, 8))]]
        self.row_h = size * ctx.u(1.9, 2.6)
        self.sym_w = size * ctx.u(1.8, 2.6)
        self.gap = size * 0.8
        self.rows_top = pad + self.hsize * ctx.u(1.5, 1.9)
        n_fit = int((max_h - self.rows_top - pad) // self.row_h)
        self.labels = labels[: max(0, n_fit)]
        self.kinds = [ctx.pick(SYMBOLS) for _ in self.labels]
        if len(self.labels) < 2:
            self.ok = False
            return
        self.w = max(pad + cv.width(self.heading, self.hsize),
                     pad + self.sym_w + self.gap + max(cv.width(s, size) for s in self.labels)) + pad
        self.h = self.rows_top + len(self.labels) * self.row_h + pad
        if self.w > max_w:
            self.ok = False

    def draw(self, x: float, y: float) -> list[Region]:
        cv = self.ctx.cv
        cv.text(x + self.pad, y + self.pad + cv.asc(self.hsize), self.heading, self.hsize)
        for i, (lab, kind) in enumerate(zip(self.labels, self.kinds)):
            top = y + self.rows_top + i * self.row_h
            sx = x + self.pad
            draw_symbol(cv, kind, (sx, top + self.row_h * 0.25, sx + self.sym_w, top + self.row_h * 0.75))
            cv.text(sx + self.sym_w + self.gap, cv.baseline_for(top, top + self.row_h, self.size), lab, self.size)
        if self.bordered:
            cv.rect((x, y, x + self.w, y + self.h), w=0.6)
        text = "\n".join([self.heading, *self.labels])
        return [Region("legend", rect_poly(x, y, x + self.w, y + self.h, pad=0.5 if self.bordered else 1.5), text)]


# -- title block -------------------------------------------------------------
class TitleBlock(Component):
    """Title block, either a box in the bottom-right corner or a full-height strip on the right edge.

    The drawing number is drawn with ``keep_real=True``: it survives as real text in outlined sheets.
    """

    def __init__(self, ctx: Ctx, style: str, inner: Box):
        self.ctx, self.style = ctx, style
        cv, L, r = ctx.cv, ctx.lang, ctx.rng
        tbv = ctx.vocab["title_block"]
        v = tbv[L]
        self.keys = ["project", "title", "scale", "date", "number", "firm"] + (
            ["drawn", "checked"] if ctx.chance(0.5) else [])
        y, m, d = int(r.integers(2012, 2027)), int(r.integers(1, 13)), int(r.integers(1, 29))
        prefix = ctx.pick(["A", "A", "A", "AS", "S", "M", "E"])
        self.values = {
            "project": ctx.pick(v["projects"]), "title": ctx.pick(v["titles"]), "scale": ctx.pick(tbv["scales"]),
            "date": f"{y}.{m:02d}.{d:02d}" if L == "ja" else f"{y}-{m:02d}-{d:02d}",
            "number": f"{prefix}-{int(r.integers(1, 10))}{int(r.integers(0, 100)):02d}",
            "firm": ctx.pick(v["firms"]), "drawn": ctx.pick(v["people"]), "checked": ctx.pick(v["people"]),
        }
        self.labels = {k: v["labels"][k] for k in self.keys}
        self.size = ctx.size * ctx.u(0.9, 1.15)
        self.vsize = {k: self.size * (1.5 if k == "number" else 1.0) for k in self.keys}
        self.revs: list[str] = []
        if style == "box" and ctx.chance(0.5):
            # base-14 Latin fonts have no "△", so English sheets use "Rev." marks
            words = ["修正", "追記", "変更"] if L == "ja" else ["revised", "issued for review", "updated"]
            mark = "△" if L == "ja" else "Rev."
            self.revs = [f"{mark}{i + 1}  {self.values['date'][:5]}{min(m + i, 12):02d}{self.values['date'][7:]}  "
                         f"{ctx.pick(words)}" for i in range(int(r.integers(1, 4)))]
        (self._plan_box if style == "box" else self._plan_strip)(inner)

    def _plan_box(self, inner: Box) -> None:
        cv, s = self.ctx.cv, self.size
        self.pad = s * 0.6
        self.row_h = s * self.ctx.u(2.0, 2.8)
        self.lab_w = max(cv.width(self.labels[k], s * 0.85) for k in self.keys) + 2 * self.pad
        val_w = max(cv.width(self.values[k], self.vsize[k]) for k in self.keys) + 2 * self.pad
        rev_w = max([cv.width(t, s * 0.85) for t in self.revs], default=0) + 2 * self.pad
        self.w = max(self.lab_w + val_w, rev_w, self.ctx.u(190, 300) * s / 7)
        self.heights = [self.row_h * (1.4 if k == "number" else 1.0) for k in self.keys]
        self.rev_h = self.row_h * 0.8
        self.h = sum(self.heights) + len(self.revs) * self.rev_h
        x1, y1 = inner[2], inner[3]
        self.rect = (x1 - self.w, y1 - self.h, x1, y1)

    def _plan_strip(self, inner: Box) -> None:
        cv, s = self.ctx.cv, self.size
        self.pad = s * 0.5
        self.w = self.ctx.u(80, 120) * s / 7
        avail = self.w - 2 * self.pad
        self.lsize = s * 0.7
        for k in self.keys:  # shrink values that do not fit the strip width
            self.vsize[k] = min(self.vsize[k], avail / max(cv.width(self.values[k], 1.0), 1e-6))
        self.heights = [self.pad + self.lsize * 1.5 + self.vsize[k] * 1.5 for k in self.keys]
        self.h = inner[3] - inner[1]
        self.rect = (inner[2] - self.w, inner[1], inner[2], inner[3])

    def draw(self, x: float | None = None, y: float | None = None) -> list[Region]:
        cv = self.ctx.cv
        x0, y0, x1, y1 = self.rect
        cv.rect(self.rect, w=1.0)
        lines: list[str] = []
        if self.style == "box":
            yy = y0
            for t in self.revs:
                cv.text(x0 + self.pad, cv.baseline_for(yy, yy + self.rev_h, self.size * 0.85), t, self.size * 0.85)
                yy += self.rev_h
                cv.line((x0, yy), (x1, yy), w=0.4)
                lines.append(t)
            cv.line((x0 + self.lab_w, yy), (x0 + self.lab_w, y1), w=0.4)
            for k, hgt in zip(self.keys, self.heights):
                ls, vs = self.size * 0.85, self.vsize[k]
                cv.text(x0 + self.pad, cv.baseline_for(yy, yy + hgt, ls), self.labels[k], ls)
                cv.text(x0 + self.lab_w + self.pad, cv.baseline_for(yy, yy + hgt, vs), self.values[k], vs,
                        keep_real=(k == "number"))
                yy += hgt
                if yy < y1 - 0.5:
                    cv.line((x0, yy), (x1, yy), w=0.4)
                lines.append(f"{self.labels[k]} {self.values[k]}")
        else:
            # fields stacked from the bottom; the space above is left as empty stamp boxes
            yy = y1
            for k, hgt in reversed(list(zip(self.keys, self.heights))):
                top = yy - hgt
                cv.line((x0, top), (x1, top), w=0.4)
                cv.text(x0 + self.pad, top + self.pad + cv.asc(self.lsize), self.labels[k], self.lsize)
                cv.text(x0 + self.pad, top + self.pad + self.lsize * 1.5 + cv.asc(self.vsize[k]), self.values[k],
                        self.vsize[k], keep_real=(k == "number"))
                yy = top
            n_boxes = int(self.ctx.rng.integers(1, 4))
            for b in range(1, n_boxes):
                cv.line((x0, y0 + (yy - y0) * b / n_boxes), (x1, y0 + (yy - y0) * b / n_boxes), w=0.4)
            lines = [f"{self.labels[k]} {self.values[k]}" for k in self.keys]
        return [Region("title_block", rect_poly(*self.rect, pad=1.0), "\n".join(lines))]


# -- floor plan --------------------------------------------------------------
class FloorPlan:
    """Grid, rooms, doors, hatching, dimensions, north arrow; room tags and leader callouts (labelled)."""

    def __init__(self, ctx: Ctx, rect: Box):
        self.ctx, self.rect = ctx, rect

    def draw(self) -> list[Region]:
        ctx, cv, r = self.ctx, self.ctx.cv, self.ctx.rng
        x0, y0, x1, y1 = self.rect
        if x1 - x0 < 220 or y1 - y0 < 160:
            return []
        s = ctx.size
        bub = s * 1.1
        gx0, gy0 = x0 + bub * 2 + s * 3.5, y0 + bub * 2 + 6
        gx1, gy1 = x1 - 10, y1 - s * 4
        occupied: list[Box] = []
        regions: list[Region] = []

        def axis(a: float, b: float, lo: float, hi: float, nmax: int) -> np.ndarray:
            n = int(np.clip((b - a) // ctx.u(lo, hi), 1, nmax))
            pts = np.linspace(a, b, n + 1)
            if n > 1:
                pts[1:-1] += r.uniform(-0.15, 0.15, n - 1) * (b - a) / n
            return pts

        xs, ys = axis(gx0, gx1, 80, 170, 10), axis(gy0, gy1, 70, 150, 6)
        alpha = ctx.chance(0.5)
        for i, x in enumerate(xs):
            cv.line((x, y0 + bub * 2), (x, gy1), w=0.3, color=GREY, dashes="[6 2 1 2] 0")
            cv.circle((x, y0 + bub), bub, w=0.5)
            lab = f"X{i + 1}" if not alpha else str(i + 1)
            cv.text(x - cv.width(lab, s * 0.8) / 2, cv.baseline_for(y0, y0 + 2 * bub, s * 0.8), lab, s * 0.8)
            occupied.append((x - bub, y0, x + bub, y0 + 2 * bub))
        for j, y in enumerate(ys):
            cv.line((x0 + bub * 2, y), (gx1, y), w=0.3, color=GREY, dashes="[6 2 1 2] 0")
            cv.circle((x0 + bub, y), bub, w=0.5)
            lab = f"Y{j + 1}" if not alpha else chr(ord("A") + j)
            cv.text(x0 + bub - cv.width(lab, s * 0.8) / 2, cv.baseline_for(y - bub, y + bub, s * 0.8), lab, s * 0.8)
            occupied.append((x0, y - bub, x0 + 2 * bub, y + bub))

        # rooms (some span two bays)
        rooms: list[Box] = []
        for j in range(len(ys) - 1):
            i = 0
            while i < len(xs) - 1:
                span = 2 if (i + 2 < len(xs) and ctx.chance(0.25)) else 1
                rooms.append((xs[i] + 3, ys[j] + 3, xs[i + span] - 3, ys[j + 1] - 3))
                i += span
        wall_w = ctx.u(1.0, 2.0)
        for rr in rooms:
            cv.rect(rr, w=wall_w)
            cv.rect((rr[0] + 3.5, rr[1] + 3.5, rr[2] - 3.5, rr[3] - 3.5), w=0.35)
            rw = rr[2] - rr[0]
            if ctx.chance(0.8) and rw > 40:
                dl = min(22.0, rw * 0.3)
                dx = ctx.u(rr[0] + 8, rr[2] - dl - 8)
                dy = rr[3] - 3.5
                cv.arc((dx, dy - dl), (dx + dl, dy - dl), (dx + dl, dy))
                cv.line((dx, dy), (dx, dy - dl), w=0.6)
            if ctx.chance(0.25):
                step = ctx.u(5, 9)
                for t in np.arange(rr[0] + 6, rr[2] - 6, step):
                    cv.line((t, rr[3] - 6), (min(t + 25, rr[2] - 6), rr[3] - 6 - (min(t + 25, rr[2] - 6) - t)),
                            w=0.25, color=LIGHT)
        if ctx.chance(0.5):
            for x in xs:
                for y in ys:
                    cv.rect((x - 3, y - 3, x + 3, y + 3), w=0.3, fill=GREY)

        # room tags (labelled text_callout)
        names = [str(n) for n in r.permutation(ctx.vocab["rooms"][ctx.lang])]
        for k, rr in enumerate(rooms):
            name = names[k % len(names)]
            second = ""
            if ctx.chance(0.6):
                area = f"{int(r.integers(8, 120))}.{int(r.integers(0, 10))}"
                second = (f"{area}㎡" if ctx.lang == "ja" else f"{area} m2") if ctx.chance(0.7) else \
                    f"CH={ctx.pick([2400, 2500, 2600, 2700, 3000])}"
            ls, ss = s * ctx.u(1.0, 1.25), s * ctx.u(0.75, 0.9)
            iw, ih = rr[2] - rr[0] - 12, rr[3] - rr[1] - 12
            cx = (rr[0] + rr[2]) / 2 + ctx.u(-0.1, 0.1) * iw
            cy = (rr[1] + rr[3]) / 2 + ctx.u(-0.1, 0.1) * ih
            nw = cv.width(name, ls)
            sw = cv.width(second, ss) if second else 0.0
            line2 = ss * 0.3 + cv.asc(ss) + cv.desc(ss) if second else 0.0
            block_h = cv.asc(ls) + cv.desc(ls) + line2
            if max(nw, sw) <= iw and block_h <= ih:
                b1 = cy - block_h / 2 + cv.asc(ls)
                polys = [cv.text(cx - nw / 2, b1, name, ls)]
                text = name
                if second:
                    polys.append(cv.text(cx - sw / 2, b1 + cv.desc(ls) + ss * 0.3 + cv.asc(ss), second, ss))
                    text += "\n" + second
                bb = aabb([p for poly in polys for p in poly])
                poly = rect_poly(*bb, pad=1.5)
            elif nw <= ih and (cv.asc(ls) + cv.desc(ls)) <= iw:
                # narrow room: name reads bottom-to-top (glyph tops face left)
                bx = cx + (cv.asc(ls) - cv.desc(ls)) / 2
                cv.text(bx, cy + nw / 2, name, ls, angle=90)
                poly, text = cv.text_poly(bx, cy + nw / 2, name, ls, 90, pad=1.5), name
            else:
                continue
            occupied.append(aabb(poly))
            regions.append(Region("text_callout", poly, text))

        # dimensions (unlabelled): bottom string, optional left string read bottom-to-top
        ds = s * 0.8
        yd = gy1 + s * 2.2
        cv.line((xs[0], yd), (xs[-1], yd), w=0.4)
        for a, b in zip(xs[:-1], xs[1:]):
            cv.line((a - 3, yd + 3), (a + 3, yd - 3), w=0.6)
            lab = str(int(round((b - a) * ctx.u(55, 65), -2)))
            p = cv.text((a + b) / 2 - cv.width(lab, ds) / 2, yd - 2, lab, ds)
            occupied.append(aabb(p))
        cv.line((xs[-1] - 3, yd + 3), (xs[-1] + 3, yd - 3), w=0.6)
        if ctx.chance(0.6):
            xd = gx0 - s * 2.2
            cv.line((xd, ys[0]), (xd, ys[-1]), w=0.4)
            for a, b in zip(ys[:-1], ys[1:]):
                cv.line((xd - 3, a + 3), (xd + 3, a - 3), w=0.6)
                lab = str(int(round((b - a) * ctx.u(55, 65), -2)))
                start = (xd - 2, (a + b) / 2 + cv.width(lab, ds) / 2)
                p = cv.text(*start, lab, ds, angle=90)
                occupied.append(aabb(p))
            cv.line((xd - 3, ys[-1] + 3), (xd + 3, ys[-1] - 3), w=0.6)

        # north arrow (unlabelled) in the top-right corner
        if ctx.chance(0.6):
            nc = (x1 - 22, y0 + 34)
            cv.circle(nc, 10, w=0.5)
            cv.polyline([(nc[0], nc[1] - 10), (nc[0] + 4, nc[1] + 6), (nc[0], nc[1] + 3), (nc[0] - 4, nc[1] + 6)],
                        w=0.4, fill=INK, closed=True)
            cv.text(nc[0] - cv.width("N", s) / 2, nc[1] - 12, "N", s)
            occupied.append((nc[0] - 12, nc[1] - 12 - s, nc[0] + 12, nc[1] + 12))

        # leader callouts (labelled text_callout)
        n_call = int(r.integers(1, min(8, len(rooms) // 2 + 2) + 1))
        lim = (x0 + 2, y0 + 2, x1 - 2, y1 - 2)
        for _ in range(n_call):
            for _attempt in range(40):
                rr = rooms[int(r.integers(len(rooms)))]
                if rr[2] - rr[0] < 20 or rr[3] - rr[1] < 20:
                    continue
                target = (ctx.u(rr[0] + 8, rr[2] - 8), ctx.u(rr[1] + 8, rr[3] - 8))
                text = ctx.fill(ctx.pick(ctx.vocab["callout_templates"][ctx.lang]))
                cs = s * ctx.u(0.85, 1.05)
                angle = ctx.callout_angle()
                dist, theta = ctx.u(20, 80), ctx.u(0, 2 * math.pi)
                ax, ay = target[0] + dist * math.cos(theta), target[1] + dist * math.sin(theta)
                v = (math.cos(math.radians(angle)), -math.sin(math.radians(angle)))  # text direction
                tw = cv.width(text, cs)
                behind = math.cos(theta) * v[0] + math.sin(theta) * v[1] < 0
                sx, sy = (ax - tw * v[0], ay - tw * v[1]) if behind else (ax, ay)
                poly = cv.text_poly(sx, sy, text, cs, angle, pad=1.5)
                bb = aabb(poly)
                if not (bb[0] >= lim[0] and bb[1] >= lim[1] and bb[2] <= lim[2] and bb[3] <= lim[3]):
                    continue
                if any(overlaps(bb, o, gap=2.0) for o in occupied):
                    continue
                cv.text(sx, sy, text, cs, angle=angle)
                up = (-math.sin(math.radians(angle)), -math.cos(math.radians(angle)))
                hm = (cv.asc(cs) - cv.desc(cs)) / 2
                cv.line(target, (ax + up[0] * hm, ay + up[1] * hm), w=0.35)
                cv.circle(target, 1.2, w=0.3, fill=INK)
                occupied.append(bb)
                regions.append(Region("text_callout", poly, text))
                break
        return regions


Maker = Callable[[float, float], Component]
