"""
World layouts for the SAR training simulator.

make_v2_layout()      -> exact replica of your Gazebo world sar_building_v2.sdf
make_random_layout()  -> procedural corridor + rooms buildings (training set)

A layout is just geometry in metres:
    boxes   : occupied axis-aligned rectangles (cx, cy, sx, sy)
    victims : true person positions (x, y)
    spawn   : robot start (x, y)
"""
from dataclasses import dataclass, field
import math
import numpy as np

WALL_T = 0.15


@dataclass
class Layout:
    width: float
    height: float
    boxes: list
    victims: list
    spawn: tuple
    rooms: list = field(default_factory=list)
    name: str = "layout"
    entrance: tuple = (0.0, 0.0)   # y-range of the west entrance gap


# ----------------------------------------------------------------------
# rasterisation
# ----------------------------------------------------------------------
def rasterize(boxes, origin, res, shape):
    """Boxes -> boolean occupancy grid. Walls thinner than one cell are
    widened to exactly one cell so they never vanish at coarse resolution."""
    rows, cols = shape
    occ = np.zeros(shape, dtype=bool)
    ox, oy = origin
    for cx, cy, sx, sy in boxes:
        sx = max(sx, res * 1.01)
        sy = max(sy, res * 1.01)
        c0 = int(math.ceil((cx - sx / 2 - ox) / res - 0.5))
        c1 = int(math.floor((cx + sx / 2 - ox) / res - 0.5))
        r0 = int(math.ceil((cy - sy / 2 - oy) / res - 0.5))
        r1 = int(math.floor((cy + sy / 2 - oy) / res - 0.5))
        c0, r0 = max(c0, 0), max(r0, 0)
        c1, r1 = min(c1, cols - 1), min(r1, rows - 1)
        if c1 >= c0 and r1 >= r0:
            occ[r0:r1 + 1, c0:c1 + 1] = True
    return occ


# ----------------------------------------------------------------------
# exact copy of sar_building_v2.sdf
# ----------------------------------------------------------------------
def make_v2_layout():
    t = WALL_T
    boxes = [
        # corridor south wall (doors x=4 1.8m, x=12 1.3m PINCH, x=20 1.8m)
        (1.55, 4.8, 3.1, t), (8.125, 4.8, 6.45, t),
        (15.875, 4.8, 6.45, t), (22.45, 4.8, 3.1, t),
        # corridor north wall (doors x=4, 12, 20 all 1.8m)
        (1.55, 7.2, 3.1, t), (8.0, 7.2, 6.2, t),
        (16.0, 7.2, 6.2, t), (22.45, 7.2, 3.1, t),
        # south dividers: x=8 solid, x=16 loop door (gap y 1.6-3.2)
        (8, 2.4, t, 4.8), (16, 0.8, t, 1.6), (16, 4.0, t, 1.6),
        # north dividers: x=8 loop door (gap y 8.8-10.4), x=16 solid
        (8, 8.0, t, 1.6), (8, 11.2, t, 1.6), (16, 9.6, t, 4.8),
        # exterior
        (12, 0, 24, t), (12, 12, 24, t),
        (0, 2.4, t, 4.8), (0, 9.6, t, 4.8),      # west wall, entrance y 4.8-7.2
        (24, 6, t, 12),
        # clutter
        (3.4, 3.3, 0.6, 0.6), (13.6, 3.6, 0.6, 0.6), (19.2, 2.9, 0.6, 0.6),
        (5.1, 8.6, 0.6, 0.6), (12.2, 8.9, 0.6, 0.6),
    ]
    victims = [(3.0, 2.0), (14.5, 2.6), (20.5, 2.0),
               (4.0, 9.5), (10.2, 9.6), (19.5, 9.8)]
    rooms = [(0.1, 0.1, 7.9, 4.7), (8.1, 0.1, 15.9, 4.7), (16.1, 0.1, 23.9, 4.7),
             (0.1, 7.3, 7.9, 11.9), (8.1, 7.3, 15.9, 11.9), (16.1, 7.3, 23.9, 11.9)]
    return Layout(24.0, 12.0, boxes, victims, (0.5, 6.0), rooms, "sar_building_v2", (4.8, 7.2))


# ----------------------------------------------------------------------
# procedural layouts
# ----------------------------------------------------------------------
def _hwall(y, x0, x1, gaps=()):
    """Horizontal wall at height y from x0..x1 with gaps [(g0,g1),...]."""
    out, cur = [], x0
    for g0, g1 in sorted(gaps):
        if g0 > cur:
            out.append(((cur + g0) / 2, y, g0 - cur, WALL_T))
        cur = max(cur, g1)
    if x1 > cur:
        out.append(((cur + x1) / 2, y, x1 - cur, WALL_T))
    return out


def _vwall(x, y0, y1, gaps=()):
    out, cur = [], y0
    for g0, g1 in sorted(gaps):
        if g0 > cur:
            out.append((x, (cur + g0) / 2, WALL_T, g0 - cur))
        cur = max(cur, g1)
    if y1 > cur:
        out.append((x, (cur + y1) / 2, WALL_T, y1 - cur))
    return out


def _split_widths(rng, total, n, min_w):
    for _ in range(100):
        w = rng.dirichlet(np.ones(n) * 4.0) * total
        if w.min() >= min_w:
            return w
    return np.full(n, total / n)


def make_random_layout(rng):
    W = rng.uniform(20.0, 34.0)
    H = rng.uniform(11.0, 16.0)
    cw = rng.uniform(2.0, 2.6)
    cy = H / 2 + rng.uniform(-1.0, 1.0)
    ys, yn = cy - cw / 2, cy + cw / 2
    door_widths = [1.3, 1.6, 1.8, 1.8, 2.0]

    boxes, rooms, room_list = [], [], []
    boxes += _hwall(0, 0, W) + _hwall(H, 0, W)
    boxes += _vwall(0, 0, ys) + _vwall(0, yn, H) + _vwall(W, 0, H)

    corr_gaps_s, corr_gaps_n = [], []
    for side in ("south", "north"):
        y0, y1 = (0.0, ys) if side == "south" else (yn, H)
        depth = y1 - y0
        n = int(rng.integers(2, 5))
        widths = _split_widths(rng, W, n, 4.6)
        n = len(widths)
        edges = np.concatenate([[0.0], np.cumsum(widths)])
        door_info = []
        for i in range(n):
            xa, xb = edges[i], edges[i + 1]
            ndoors = 2 if (xb - xa >= 7.5 and rng.random() < 0.2) else 1
            spans = [(xa, xb)] if ndoors == 1 else [(xa, (xa + xb) / 2), ((xa + xb) / 2, xb)]
            for (sa, sb) in spans:
                dw = float(rng.choice(door_widths))
                lo, hi = sa + dw / 2 + 0.5, sb - dw / 2 - 0.5
                dc = rng.uniform(lo, hi) if hi > lo else (sa + sb) / 2
                (corr_gaps_s if side == "south" else corr_gaps_n).append((dc - dw / 2, dc + dw / 2))
                door_info.append((i, dc, dw))
            rooms.append((xa + 0.1, y0 + 0.1, xb - 0.1, y1 - 0.1))
            room_list.append((side, i, xa, xb, y0, y1))
        loop_info = []
        for i in range(1, n):
            bp = edges[i]
            gaps = []
            if rng.random() < 0.35:
                lw = float(rng.choice([1.4, 1.6, 1.8]))
                lo, hi = y0 + 1.0 + lw / 2, y1 - 1.0 - lw / 2
                if hi > lo:                      # room deep enough for a loop door
                    gc = rng.uniform(lo, hi)
                    gaps = [(gc - lw / 2, gc + lw / 2)]
                    loop_info.append((bp, gc, lw))
            boxes += _vwall(bp, y0, y1, gaps)
        # clutter
        for i in range(n):
            xa, xb = edges[i], edges[i + 1]
            for _ in range(int(rng.choice([0, 1, 1, 2]))):
                for _try in range(12):
                    s = rng.uniform(0.5, 0.9)
                    x = rng.uniform(xa + 0.9, xb - 0.9)
                    y = rng.uniform(y0 + 0.9, y1 - 0.9)
                    near_door = any(
                        (di == i and abs(x - dc) < dw / 2 + 0.9 and
                         abs(y - (ys if side == "south" else yn)) < 1.8)
                        for (di, dc, dw) in door_info)
                    near_loop = any(
                        (abs(x - bp) < 1.6 and abs(y - gc) < lw / 2 + 0.9)
                        for (bp, gc, lw) in loop_info)
                    if not near_door and not near_loop:
                        boxes.append((x, y, s, s))
                        break

    boxes += _hwall(ys, 0, W, corr_gaps_s) + _hwall(yn, 0, W, corr_gaps_n)

    # victims
    n_v = int(rng.integers(3, 11))
    victims, spawn = [], (0.6, cy)
    areas = np.array([(r[3] - r[2]) * (r[5] - r[4]) for r in room_list])
    probs = areas / areas.sum()
    tries = 0
    while len(victims) < n_v and tries < 400:
        tries += 1
        if rng.random() < 0.15:
            x, y = rng.uniform(3.0, W - 1.0), rng.uniform(ys + 0.6, yn - 0.6)
        else:
            _, _, xa, xb, y0, y1 = room_list[int(rng.choice(len(room_list), p=probs))]
            x, y = rng.uniform(xa + 0.7, xb - 0.7), rng.uniform(y0 + 0.7, y1 - 0.7)
        if math.hypot(x - spawn[0], y - spawn[1]) < 3.0:
            continue
        if any(math.hypot(x - vx, y - vy) < 1.8 for vx, vy in victims):
            continue
        if any(abs(x - bx) < sx / 2 + 0.5 and abs(y - by) < sy / 2 + 0.5
               for bx, by, sx, sy in boxes):
            continue
        victims.append((x, y))
    return Layout(W, H, boxes, victims, spawn, rooms, "random", (ys, yn))
