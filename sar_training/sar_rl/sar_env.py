"""
SarExplorationEnv  -  fast 2D training simulator for the high-level
EXPLORE <-> INVESTIGATE decision, mirroring the Mission 2 explorer pipeline.

ONE STEP = ONE DECISION. The agent picks a goal from a variable-size candidate
set (frontier clusters + known target entries). The simulator then does what
Nav2 + the stuck watchdog + YOLO + the semantic DB would do: plan on the known
map, drive there (sensing as it goes), maybe get stuck, inspect, update the
map and target database, and returns the next decision.

What is IDENTICAL to Mission 2 (copied logic):
  frontier detection / clustering, visible-IG ray casting, goal-cell safety
  filter, visited-frontier + blacklist filtering, target availability
  (noise floor, inspected, abandoned), max_investigate_attempts, standoff /
  visited_radius inspection rule, completion after N empty cycles.

What is APPROXIMATED (randomised per episode = domain randomisation):
  Nav2 (Dijkstra on inflated known map), stuck events (probability grows with
  how narrow the path/goal is), YOLO (range/FOV/LOS dependent detection,
  localisation noise, duplicates, false positives), driving speed, timing.

No Mission 2 utility formula appears anywhere in this file.
"""
import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .layouts import Layout, make_random_layout, make_v2_layout, rasterize

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # tiny shim so the simulator runs without gymnasium
    class _Box:
        def __init__(self, low, high, shape=None, dtype=np.float32):
            self.low, self.high, self.shape, self.dtype = low, high, shape, dtype

    class _Discrete:
        def __init__(self, n):
            self.n = n

    class _Dict(dict):
        def __init__(self, d):
            super().__init__(d)

    class spaces:  # noqa: N801
        Box, Discrete, Dict = _Box, _Discrete, _Dict

    class gym:  # noqa: N801
        class Env:
            pass

SQ2 = math.sqrt(2.0)
NEIGH = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
         (-1, -1, SQ2), (-1, 1, SQ2), (1, -1, SQ2), (1, 1, SQ2)]

N_CAND_FEATURES = 13
N_GLOBAL_FEATURES = 10


@dataclass
class SimConfig:
    # grid / sensing
    res: float = 0.2
    lidar_range: float = 8.0
    lidar_rays: int = 360
    ig_range: float = 10.0
    ig_rays: int = 180
    # copied from Mission 2 defaults
    min_cluster_size: int = 4
    min_frontier_goal_dist: float = 0.75
    visited_radius: float = 1.1
    occupied_threshold: int = 65
    robot_clearance: float = 0.30       # traversable if >= this from known obstacle (planner)
    obstacle_margin_cells: int = 2      # Mission 2 pick_goal_cell margin: 8 cells x 0.05 m = 0.4 m = 2 cells here
    planner_tolerance_m: float = 0.5    # ASSUMPTION: Nav2 planner goal tolerance - check nav2_params.yaml
    max_investigate_attempts: int = 2
    min_target_conf: float = 0.10
    blacklist_timeout: float = 30.0
    empty_cycles_before_done: int = 3
    battery_drain_pct_per_m: float = 0.5
    # candidate slots
    frontier_pool: int = 15            # Mission 2: largest 15 clusters get evaluated
    max_frontier_slots: int = 8        # Mission 2: top 8 by information gain go to Nav2
    max_target_slots: int = 8
    # timing
    explore_period_s: float = 3.0
    decision_overhead_s: float = 4.0
    inspect_time_s: float = 10.0
    sense_every_m: float = 0.8
    true_inspect_radius: float = 2.5    # a snapshot of a real person is possible within this range (with line of sight)
    max_time_s: float = 2400.0
    max_decisions: int = 300
    # reward (task outcomes only)
    r_inspect: float = 10.0
    r_inspect_speed: float = 5.0        # extra for inspecting early
    r_coverage: float = 5.0             # x coverage fraction gained
    c_time: float = 0.002               # per second
    c_dist: float = 0.005               # per metre
    c_stuck: float = 1.0
    c_abandon_real: float = 2.0
    c_abandon_fake: float = 0.3
    r_complete: float = 5.0             # x fraction of real targets inspected
    c_invalid: float = 0.5
    # domain randomisation
    domain_randomize: bool = True
    physics_overrides: dict = field(default_factory=dict)
    physics_center: dict = None          # if set: randomise AROUND these calibrated values
    randomize_width: float = 0.35        # +/- fraction around the centre


# Calibrated to ONE real Gazebo Mission 2 run on sar_building_v2 (about 2000 s,
# 13 stuck events, 15 database entries for 6 people, 11 inspected / 4 abandoned).
# Pass as SimConfig(**CALIBRATED); parameters are randomised +/-35% around it.
CALIBRATED = dict(
    physics_center=dict(speed=0.16, stuck_base=0.06, stuck_gain=1.1, det_base=0.85,
                        det_range=6.0, fov=110.0, fp_rate=0.04, dup_prob=0.12,
                        loc_noise=0.35, recovery_s=42.0, snap_fail=0.08),
    decision_overhead_s=10.0, inspect_time_s=20.0, max_time_s=4800.0)


class SarExplorationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg=None, layout="random", seed=None):
        self.cfg = cfg or SimConfig()
        self.layout_mode = layout
        self.rng = np.random.default_rng(seed)
        c = self.cfg
        self.K = c.max_frontier_slots + c.max_target_slots
        self.action_space = spaces.Discrete(self.K)
        self.observation_space = spaces.Dict({
            "cand": spaces.Box(-2.0, 2.0, (self.K, N_CAND_FEATURES), np.float32),
            "mask": spaces.Box(0.0, 1.0, (self.K,), np.float32),
            "glob": spaces.Box(-2.0, 2.0, (N_GLOBAL_FEATURES,), np.float32),
        })
        # precomputed ray tables
        self._lidar_angles = np.linspace(0, 2 * np.pi, c.lidar_rays, endpoint=False)
        self._lidar_steps = np.arange(1, int(c.lidar_range / (c.res * 0.5)) + 1) * (c.res * 0.5)
        ig_ang = np.linspace(0, 2 * np.pi, c.ig_rays, endpoint=False)
        self._ig_dx, self._ig_dy = np.cos(ig_ang), np.sin(ig_ang)
        self._ig_steps = np.arange(1, max(1, int(c.ig_range / c.res)) + 1, dtype=np.float64)

    # ------------------------------------------------------------------
    # gym API
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mode = (options or {}).get("layout", self.layout_mode)
        self._build_world(mode)
        self.phys = self._sample_physics()

        self.known = np.full(self.occ.shape, -1, dtype=np.int8)
        self.t = 0.0
        self.dist = 0.0
        self.battery = 100.0
        self.decisions = 0
        self.stuck_events = 0
        self.recoveries = 0
        self.empty_cycles = 0
        self.rec_in_row = 0
        self.done = False
        self.completed = False
        self.end_reason = ""
        self.visited_goals = []
        self.blacklist = {}
        self.entries = []
        self.true_entry = {}
        self.dup_counts = {}
        self.next_entry_id = 0
        self.detected_true = set()
        self.inspected_true = {}
        self.first_detect_t = None
        self.first_inspect_t = None
        self.last_inspect_t = None
        self.fp_entries = 0
        self.trace = [self.pos]
        self._r = 0.0
        self.last_cov = 0.0

        self._lidar_update()
        self._detect()
        self.last_cov = self._coverage()
        self._advance()
        return self._obs(), self._info()

    def action_masks(self):
        return self._mask.copy()

    def step(self, action):
        c = self.cfg
        self._r = 0.0
        t0 = self.t
        cov0 = self._coverage()
        slot = self._slots[action] if 0 <= int(action) < self.K else None
        if self.done or slot is None:
            self._r -= c.c_invalid
            self.t += 5.0
        else:
            self.decisions += 1
            self.t += c.decision_overhead_s
            self._execute(slot)
        self._advance()
        cov1 = self._coverage()
        self._r += c.r_coverage * (cov1 - cov0)
        self._r -= c.c_time * (self.t - t0)
        terminated = self.done and self.completed
        truncated = False
        if not terminated and (self.t >= c.max_time_s or self.decisions >= c.max_decisions or
                               (self.done and not self.completed)):
            truncated = True
            self.done = True
            if not self.end_reason:
                self.end_reason = "time" if self.t >= c.max_time_s else "decisions"
        if terminated:
            n = max(len(self.victims), 1)
            self._r += c.r_complete * len(self.inspected_true) / n
        return self._obs(), float(self._r), terminated, truncated, self._info()

    # ------------------------------------------------------------------
    # world construction
    # ------------------------------------------------------------------
    def _sample_physics(self):
        p = dict(speed=0.35, stuck_base=0.02, stuck_gain=0.5, det_base=0.85, det_range=6.0,
                 fov=110.0, fp_rate=0.02, dup_prob=0.05, loc_noise=0.2,
                 recovery_s=25.0, snap_fail=0.08)
        if self.cfg.physics_center is not None:
            p.update(self.cfg.physics_center)
            if self.cfg.domain_randomize:
                u, w = self.rng.uniform, self.cfg.randomize_width
                for k in ("speed", "stuck_base", "stuck_gain", "det_range", "fp_rate",
                          "dup_prob", "loc_noise", "recovery_s", "snap_fail"):
                    p[k] = p[k] * u(1 - w, 1 + w)
                p["det_base"] = float(np.clip(p["det_base"] * u(1 - w / 2, 1 + w / 2), 0.3, 0.97))
                p["fov"] = p["fov"] * u(1 - w / 3, 1 + w / 3)
            p.update(self.cfg.physics_overrides)
            return p
        if self.cfg.domain_randomize:
            u = self.rng.uniform
            p.update(speed=u(0.25, 0.5), stuck_base=u(0.0, 0.05), stuck_gain=u(0.15, 0.9),
                     det_base=u(0.5, 0.95), det_range=u(4.5, 7.5), fov=u(90, 130),
                     fp_rate=u(0.0, 0.05), dup_prob=u(0.0, 0.15), loc_noise=u(0.1, 0.45),
                     recovery_s=u(15.0, 40.0), snap_fail=u(0.05, 0.25))
        p.update(self.cfg.physics_overrides)
        return p

    def _build_world(self, mode):
        res = self.cfg.res
        attempts = 60 if mode == "random" else 1
        for _ in range(attempts):
            if mode == "v2":
                lay = make_v2_layout()
            elif isinstance(mode, Layout):
                lay = mode
            else:
                lay = make_random_layout(self.rng)
            self.ox, self.oy = -1.0, -1.0
            R = int(math.ceil((lay.height + 2.0) / res))
            C = int(math.ceil((lay.width + 2.0) / res))
            occ = rasterize(lay.boxes, (self.ox, self.oy), res, (R, C))
            occ[0, :] = occ[-1, :] = True
            occ[:, 0] = occ[:, -1] = True
            # Everything outside the building is solid, except a 1 m deep
            # vestibule behind the west entrance (otherwise the robot could
            # drive around the outside of the building through the margin).
            cxs = (np.arange(C) + 0.5) * res + self.ox
            cys = (np.arange(R) + 0.5) * res + self.oy
            outside = ((cxs[None, :] < 0.0) | (cxs[None, :] > lay.width) |
                       (cys[:, None] < 0.0) | (cys[:, None] > lay.height))
            vest = (cxs[None, :] < 0.0) & (cys[:, None] >= lay.entrance[0]) & \
                   (cys[:, None] <= lay.entrance[1])
            occ |= outside & ~vest
            clear = ndimage.distance_transform_edt(~occ) * res
            trav = (~occ) & (clear >= self.cfg.robot_clearance)
            sr, sc = self._to_cell(*lay.spawn, shape=(R, C))
            if not trav[sr, sc]:
                continue
            lab, _ = ndimage.label(trav, structure=np.ones((3, 3)))
            reach_trav = lab == lab[sr, sc]
            ok = True
            for vx, vy in lay.victims:
                r, c = self._to_cell(vx, vy, shape=(R, C))
                r0, r1, c0, c1 = max(r - 5, 0), min(r + 6, R), max(c - 5, 0), min(c + 6, C)
                if not reach_trav[r0:r1, c0:c1].any():
                    ok = False
                    break
            if ok or mode != "random":
                break
        self.layout = lay
        self.occ = occ
        self.clear_true = clear
        self.victims = list(lay.victims)
        self.pos = (float(lay.spawn[0]), float(lay.spawn[1]))
        self.heading = 0.0
        lab2, _ = ndimage.label(~occ, structure=np.ones((3, 3)))
        self.reach_free = lab2 == lab2[sr, sc]
        self.reach_total = int(self.reach_free.sum())
        self.shape = (R, C)

    def _to_cell(self, x, y, shape=None):
        res = self.cfg.res
        r = int((y - self.oy) / res)
        c = int((x - self.ox) / res)
        if shape is not None:
            r = min(max(r, 0), shape[0] - 1)
            c = min(max(c, 0), shape[1] - 1)
        return r, c

    def _cell_xy(self, r, c):
        res = self.cfg.res
        return (c + 0.5) * res + self.ox, (r + 0.5) * res + self.oy

    # ------------------------------------------------------------------
    # sensing
    # ------------------------------------------------------------------
    def _lidar_update(self):
        c = self.cfg
        x, y = self.pos
        R, C = self.shape
        ang = self._lidar_angles
        steps = self._lidar_steps
        px = x + np.cos(ang)[:, None] * steps[None, :]
        py = y + np.sin(ang)[:, None] * steps[None, :]
        cc = np.floor((px - self.ox) / c.res).astype(np.int32)
        rr = np.floor((py - self.oy) / c.res).astype(np.int32)
        inb = (rr >= 0) & (rr < R) & (cc >= 0) & (cc < C)
        rr = np.clip(rr, 0, R - 1)
        cc = np.clip(cc, 0, C - 1)
        hit = self.occ[rr, cc] | ~inb
        n = steps.size
        first = np.where(hit.any(1), hit.argmax(1), n)
        idx = np.arange(n)[None, :]
        free_mask = idx < first[:, None]
        hit_mask = (idx == first[:, None]) & inb
        fr, fc = rr[free_mask], cc[free_mask]
        sel = self.known[fr, fc] == -1
        self.known[fr[sel], fc[sel]] = 0
        self.known[rr[hit_mask], cc[hit_mask]] = 100

    def _los(self, x0, y0, x1, y1):
        d = math.hypot(x1 - x0, y1 - y0)
        n = max(2, int(d / (self.cfg.res * 0.5)))
        t = np.linspace(0, 1, n)[1:-1]
        px, py = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
        cc = ((px - self.ox) / self.cfg.res).astype(int)
        rr = ((py - self.oy) / self.cfg.res).astype(int)
        return not self.occ[rr, cc].any()

    def _new_entry(self, kind, true_id, x, y, conf):
        e = dict(id=self.next_entry_id, kind=kind, true_id=true_id, x=x, y=y, conf=conf,
                 n=1, inspected=False, abandoned=False, attempts=0)
        self.next_entry_id += 1
        self.entries.append(e)
        if self.first_detect_t is None:
            self.first_detect_t = self.t
        return e

    def _detect(self):
        p, rng = self.phys, self.rng
        x, y = self.pos
        half_fov = math.radians(p["fov"]) / 2
        for i, (vx, vy) in enumerate(self.victims):
            d = math.hypot(vx - x, vy - y)
            if d > p["det_range"] or d < 0.3:
                continue
            da = abs(math.atan2(math.sin(math.atan2(vy - y, vx - x) - self.heading),
                                math.cos(math.atan2(vy - y, vx - x) - self.heading)))
            if da > half_fov or not self._los(x, y, vx, vy):
                continue
            if rng.random() > p["det_base"] * (1.0 - 0.6 * d / p["det_range"]):
                continue
            conf = float(np.clip(0.95 - 0.1 * d + rng.normal(0, 0.1), 0.12, 0.98))
            noise = p["loc_noise"] * (0.5 + d / p["det_range"])
            ex, ey = vx + rng.normal(0, noise), vy + rng.normal(0, noise)
            e = self.true_entry.get(i)
            if e is None:
                e = self._new_entry("real", i, ex, ey, conf)
                self.true_entry[i] = e
                self.detected_true.add(i)
            else:
                n = e["n"]
                e["x"], e["y"] = (e["x"] * n + ex) / (n + 1), (e["y"] * n + ey) / (n + 1)
                e["n"] = n + 1
                e["conf"] = max(e["conf"], conf)
            if rng.random() < p["dup_prob"] and self.dup_counts.get(i, 0) < 2:
                a, off = rng.uniform(0, 2 * math.pi), rng.uniform(0.7, 1.5)
                self._new_entry("dup", i, vx + off * math.cos(a), vy + off * math.sin(a),
                                float(rng.uniform(0.15, 0.5)))
                self.dup_counts[i] = self.dup_counts.get(i, 0) + 1
        if rng.random() < p["fp_rate"] and self.fp_entries < 10:
            a = self.heading + rng.uniform(-half_fov, half_fov)
            d = rng.uniform(1.5, p["det_range"])
            fx, fy = x + d * math.cos(a), y + d * math.sin(a)
            if not any(math.hypot(fx - e["x"], fy - e["y"]) < 0.5 for e in self.entries):
                self._new_entry("fake", None, fx, fy, float(rng.uniform(0.10, 0.45)))
                self.fp_entries += 1

    # ------------------------------------------------------------------
    # map helpers (copied logic from the Mission 2 node)
    # ------------------------------------------------------------------
    def _coverage(self):
        if self.reach_total == 0:
            return 1.0
        return float(((self.known == 0) & self.reach_free).sum()) / self.reach_total

    def _info_gain(self, r, c):
        R, C = self.shape
        gx = c + 0.5 + self._ig_dx[:, None] * self._ig_steps[None, :]
        gy = r + 0.5 + self._ig_dy[:, None] * self._ig_steps[None, :]
        cc = np.floor(gx).astype(np.int32)
        rr = np.floor(gy).astype(np.int32)
        inb = (rr >= 0) & (rr < R) & (cc >= 0) & (cc < C)
        rr = np.clip(rr, 0, R - 1)
        cc = np.clip(cc, 0, C - 1)
        vals = self.known[rr, cc]
        occ = (vals >= self.cfg.occupied_threshold) & inb
        unk = (vals == -1) & inb
        stop = occ | unk | ~inb
        has = stop.any(1)
        first = stop.argmax(1)
        rays = np.where(has)[0]
        if rays.size == 0:
            return 0
        su = unk[rays, first[rays]]
        rays = rays[su]
        if rays.size == 0:
            return 0
        keys = rr[rays, first[rays]].astype(np.int64) * C + cc[rays, first[rays]]
        return int(np.unique(keys).size)

    def _find_clusters(self):
        known = self.known
        free = known == 0
        unknown = known == -1
        has_unk = ndimage.binary_dilation(unknown, structure=np.ones((3, 3)))
        fr = free & has_unk
        fr[0, :] = fr[-1, :] = False
        fr[:, 0] = fr[:, -1] = False
        labels, n = ndimage.label(fr, structure=np.ones((3, 3)))
        if n == 0:
            return []
        sizes = np.bincount(labels.ravel())[1:]
        objs = ndimage.find_objects(labels)
        out = []
        for l in np.where(sizes >= self.cfg.min_cluster_size)[0] + 1:
            sl = objs[l - 1]
            rs, cs = np.nonzero(labels[sl] == l)
            rs = rs + sl[0].start
            cs = cs + sl[1].start
            mr, mc = rs.mean(), cs.mean()
            m = int(np.argmin((rs - mr) ** 2 + (cs - mc) ** 2))
            out.append(dict(rs=rs, cs=cs, size=int(rs.size), medoid=(int(rs[m]), int(cs[m]))))
        return out

    def _is_visited(self, x, y):
        vr = self.cfg.visited_radius
        return any(math.hypot(x - vx, y - vy) < vr for vx, vy in self.visited_goals)

    def _is_blacklisted(self, x, y):
        self.blacklist = {k: v for k, v in self.blacklist.items() if v > self.t}
        vr = self.cfg.visited_radius
        return any(math.hypot(x - bx, y - by) < vr for bx, by in self.blacklist)

    def _plan(self, trav, src_rc):
        R, C = trav.shape
        idx = np.arange(R * C).reshape(R, C)
        rows, cols, w = [], [], []
        for dr, dc, wt in NEIGH:
            rs = slice(max(0, -dr), R - max(0, dr))
            cs = slice(max(0, -dc), C - max(0, dc))
            rd = slice(max(0, dr), R - max(0, -dr))
            cd = slice(max(0, dc), C - max(0, -dc))
            valid = trav[rs, cs] & trav[rd, cd]
            if dr != 0 and dc != 0:
                valid &= trav[rs, cd] & trav[rd, cs]
            a, b = idx[rs, cs][valid], idx[rd, cd][valid]
            rows.append(a)
            cols.append(b)
            w.append(np.full(a.size, wt * self.cfg.res))
        g = csr_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(R * C, R * C))
        src = src_rc[0] * C + src_rc[1]
        dist, pred = dijkstra(g, directed=True, indices=src, return_predecessors=True)
        return dist, pred, src

    @staticmethod
    def _trace(pred, src, goal):
        path, cur = [], goal
        while cur != src and cur >= 0 and len(path) < 6000:
            path.append(cur)
            cur = pred[cur]
        path.append(src)
        path.reverse()
        return np.array(path, dtype=np.int64)

    # ------------------------------------------------------------------
    # candidate construction (the agent's action set)
    # ------------------------------------------------------------------
    def _target_available(self, e):
        return (not e["inspected"] and not e["abandoned"]
                and e["conf"] >= self.cfg.min_target_conf)

    def _unsafe_mask(self):
        """Port of compute_unsafe_mask(): known-occupied cells dilated by a box of
        obstacle_margin_cells (identical semantics to the Mission 2 node)."""
        occ = self.known >= self.cfg.occupied_threshold
        m = self.cfg.obstacle_margin_cells
        return ndimage.binary_dilation(occ, structure=np.ones((2 * m + 1, 2 * m + 1), dtype=bool))

    def _pick_goal_cell(self, cl, rf, cf, unsafe):
        """Port of pick_goal_cell(): among cluster cells that are safe and at least
        min_frontier_goal_dist from the robot, take the one nearest the cluster medoid."""
        rs, cs = cl["rs"], cl["cs"]
        min_cells = self.cfg.min_frontier_goal_dist / self.cfg.res
        valid = (np.hypot(rs - rf, cs - cf) >= min_cells) & ~unsafe[rs, cs]
        if not valid.any():
            return None
        vr, vc = rs[valid], cs[valid]
        mr, mc = cl["medoid"]
        k = int(np.argmin((vr - mr) ** 2 + (vc - mc) ** 2))
        return int(vr[k]), int(vc[k])

    def _free_offset(self, e, trav, radius_m=1.5):
        """Distance from a target entry to the nearest known, obstacle-clear cell
        (same quantity the ROS node computes in _target_free_offset)."""
        res = self.cfg.res
        R, C = self.shape
        er, ec = self._to_cell(e["x"], e["y"], self.shape)
        w = int(math.ceil(radius_m / res))
        r0, r1, c0, c1 = max(er - w, 0), min(er + w + 1, R), max(ec - w, 0), min(ec + w + 1, C)
        rr, cc = np.nonzero(trav[r0:r1, c0:c1])
        if rr.size == 0:
            return radius_m
        x = (cc + c0 + 0.5) * res + self.ox
        y = (rr + r0 + 0.5) * res + self.oy
        return float(min(np.hypot(x - e["x"], y - e["y"]).min(), radius_m))

    def _refresh(self):
        c = self.cfg
        R, C = self.shape
        res = c.res
        rx, ry = self.pos
        rr0, rc0 = self._to_cell(rx, ry, self.shape)

        clear_known = ndimage.distance_transform_edt(self.known != 100) * res
        trav = (self.known == 0) & (clear_known >= c.robot_clearance)
        trav[rr0, rc0] = True
        dist, pred, src = self._plan(trav, (rr0, rc0))
        self._dist_field, self._clear_known = dist, clear_known

        # ---- frontier candidates: port of Mission 2 choose_frontier() ----------
        #  clusters -> drop visited/blacklisted -> 15 largest -> goal cell (unsafe-mask rule)
        #  -> too-close / visited / blacklist filters -> IG -> top 8 by IG
        #  -> only THEN Nav2 planning (unreachable goals drop out after the cut)
        clusters = self._find_clusters()
        avail = []
        for cl in clusters:
            x, y = self._cell_xy(*cl["medoid"])
            if self._is_visited(x, y) or self._is_blacklisted(x, y):
                continue
            avail.append(cl)
        self._n_avail_clusters = len(avail)
        avail.sort(key=lambda d: d["size"], reverse=True)
        avail = avail[:c.frontier_pool]

        unsafe = self._unsafe_mask()
        rf, cf = (ry - self.oy) / res, (rx - self.ox) / res      # float grid coords like world_to_grid()
        self._unusable = []
        pre = []
        for cl in avail:
            goal = self._pick_goal_cell(cl, rf, cf, unsafe)
            if goal is None:
                self._unusable.append(self._cell_xy(*cl["medoid"]))
                continue
            gr, gc = goal
            x, y = self._cell_xy(gr, gc)
            if math.hypot(x - rx, y - ry) < c.min_frontier_goal_dist:
                continue
            if self._is_visited(x, y) or self._is_blacklisted(x, y):
                continue
            pre.append(dict(gr=gr, gc=gc, x=x, y=y, size=cl["size"], ig=self._info_gain(gr, gc),
                            medoid=cl["medoid"]))
        pre.sort(key=lambda d: d["ig"], reverse=True)
        pre = pre[:c.max_frontier_slots]

        frontier_cands = []
        for d in pre:
            gr, gc = d["gr"], d["gc"]
            if not np.isfinite(dist[gr * C + gc]) or dist[gr * C + gc] <= 0.0:
                self._unusable.append(self._cell_xy(*d["medoid"]))
                continue
            frontier_cands.append(dict(
                kind="frontier", x=d["x"], y=d["y"], r=gr, c=gc, path_len=float(dist[gr * C + gc]),
                path=self._trace(pred, src, gr * C + gc), ig=d["ig"], size=d["size"], conf=0.0,
                attempts=0, entry=None, offset=0.0, nominal=(d["x"], d["y"])))

        # ---- target candidates: Mission 2 standoff rule + Nav2 planner tolerance -----
        target_cands = []
        self._unreachable_entries = []
        Rin = c.visited_radius
        tol_cells = int(math.ceil(c.planner_tolerance_m / res))
        for e in self.entries:
            if not self._target_available(e):
                continue
            d = math.hypot(e["x"] - rx, e["y"] - ry)
            if d <= Rin:
                # compute_standoff_point() returns the robot's own position -> Nav2 returns no
                # usable path (<2 poses) -> Mission 2 treats the target as unreachable
                self._unreachable_entries.append(e)
                continue
            f = (d - Rin * 0.65) / d
            sx, sy = rx + (e["x"] - rx) * f, ry + (e["y"] - ry) * f
            er, ec = self._to_cell(sx, sy, self.shape)
            r0, r1 = max(er - tol_cells, 0), min(er + tol_cells + 1, R)
            c0, c1 = max(ec - tol_cells, 0), min(ec + tol_cells + 1, C)
            wr, wc = np.mgrid[r0:r1, c0:c1]
            wr, wc = wr.ravel(), wc.ravel()
            okc = trav[wr, wc] & np.isfinite(dist[wr * C + wc])
            if not okc.any():
                self._unreachable_entries.append(e)
                continue
            wr, wc = wr[okc], wc[okc]
            wx = (wc + 0.5) * res + self.ox
            wy = (wr + 0.5) * res + self.oy
            dd = np.hypot(wx - sx, wy - sy)
            k = int(np.argmin(dd))
            if dd[k] > c.planner_tolerance_m or dist[wr[k] * C + wc[k]] <= 0.0:
                self._unreachable_entries.append(e)
                continue
            gr, gc = int(wr[k]), int(wc[k])
            target_cands.append(dict(
                kind="target", x=float(wx[k]), y=float(wy[k]), r=gr, c=gc,
                path_len=float(dist[gr * C + gc]), path=self._trace(pred, src, gr * C + gc),
                ig=0, size=0, conf=e["conf"], attempts=e["attempts"], entry=e,
                offset=self._free_offset(e, trav), nominal=(sx, sy)))
        target_cands.sort(key=lambda d: d["path_len"])
        target_cands = target_cands[:c.max_target_slots]

        self._slots = [None] * self.K
        for i, cd in enumerate(frontier_cands):
            self._slots[i] = cd
        for i, cd in enumerate(target_cands):
            self._slots[c.max_frontier_slots + i] = cd
        self._mask = np.array([s is not None for s in self._slots], dtype=np.float32)
        self._n_avail_targets = sum(1 for e in self.entries if self._target_available(e))

    def _advance(self):
        """Refresh candidates; handle empty cycles / recovery until a decision is possible."""
        c = self.cfg
        while True:
            self._refresh()
            if self._mask.any():
                self.empty_cycles = 0
                self.rec_in_row = 0
                return
            if self.t >= c.max_time_s:
                self.done = True
                self.end_reason = "time"
                return
            if self._n_avail_clusters == 0 and self._n_avail_targets == 0:
                self.empty_cycles += 1
                self.t += c.explore_period_s
                if self.empty_cycles >= c.empty_cycles_before_done:
                    self.done = True
                    self.completed = True
                    self.end_reason = "complete"
                    return
            else:
                self.recoveries += 1
                self.rec_in_row += 1
                self.t += self.phys["recovery_s"] * 0.5
                self.blacklist.clear()
                if not self.frontier_available_flag():
                    # nothing usable: give up on unreachable frontier regions (so the
                    # mission can finish) and fail unreachable targets, as Mission 2 does
                    self.visited_goals.extend(self._unusable)
                    for e in self._unreachable_entries:
                        self._register_attempt_failure(e)
                if self.rec_in_row >= 4:
                    self.done = True
                    self.end_reason = "stalled"
                    return

    def frontier_available_flag(self):
        return any(s is not None and s["kind"] == "frontier" for s in self._slots)

    # ------------------------------------------------------------------
    # executing a decision
    # ------------------------------------------------------------------
    def _register_attempt_failure(self, e):
        e["attempts"] += 1
        if e["attempts"] >= self.cfg.max_investigate_attempts:
            e["abandoned"] = True
            real_missed = e["kind"] == "real" and e["true_id"] not in self.inspected_true
            self._r -= self.cfg.c_abandon_real if real_missed else self.cfg.c_abandon_fake

    def _execute(self, cd):
        c, p, rng = self.cfg, self.phys, self.rng
        C = self.shape[1]
        path = cd["path"]
        pr, pc = path // C, path % C
        pxy = np.stack([(pc + 0.5) * c.res + self.ox, (pr + 0.5) * c.res + self.oy], axis=1)
        seg = np.hypot(*np.diff(pxy, axis=0).T) if len(pxy) > 1 else np.zeros(0)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(cum[-1])

        if cd["kind"] == "frontier":
            self.visited_goals.append((cd["x"], cd["y"]))
        else:
            self.visited_goals.append((cd["x"], cd["y"]))

        # stuck model: narrow true clearance along path/goal raises the chance
        true_clear = float(self.clear_true[pr[3:], pc[3:]].min()) if len(pr) > 4 else 1.0
        narrow = max(0.0, 1.0 - true_clear / 0.8)
        p_stuck = float(np.clip(p["stuck_base"] + p["stuck_gain"] * narrow, 0.0, 0.9))
        stuck = rng.random() < p_stuck and total > 1.0
        end_d = total * rng.uniform(0.2, 0.9) if stuck else total

        sense_d = list(np.arange(c.sense_every_m, end_d, c.sense_every_m)) + [end_d]
        prev_d = 0.0
        for d in sense_d:
            i = int(np.searchsorted(cum, d, side="left"))
            i = min(i, len(pxy) - 1)
            if i > 0:
                dx, dy = pxy[i] - pxy[max(i - 3, 0)]
                if abs(dx) + abs(dy) > 1e-6:
                    self.heading = math.atan2(dy, dx)
            self.pos = (float(pxy[i][0]), float(pxy[i][1]))
            self.t += (d - prev_d) / p["speed"]
            self.dist += d - prev_d
            self.battery = max(0.0, self.battery - (d - prev_d) * c.battery_drain_pct_per_m)
            self._r -= c.c_dist * (d - prev_d)
            prev_d = d
            self._lidar_update()
            self._detect()
            self.trace.append(self.pos)

        if stuck:
            self.stuck_events += 1
            self._r -= c.c_stuck
            self.t += p["recovery_s"]
            if cd["kind"] == "frontier":
                self.blacklist[(cd["x"], cd["y"])] = self.t + c.blacklist_timeout
            else:
                self._register_attempt_failure(cd["entry"])
            return

        if cd["kind"] == "target":
            self.t += c.inspect_time_s
            e = cd["entry"]
            px, py = self.pos
            # a real person must actually be close and visible for a snapshot to be confirmed
            near = [i for i, (vx, vy) in enumerate(self.victims)
                    if math.hypot(vx - px, vy - py) <= c.true_inspect_radius and self._los(px, py, vx, vy)]
            snap_ok = bool(near) and rng.random() >= p["snap_fail"]
            if snap_ok:
                for i in near:
                    if i not in self.inspected_true:
                        self.inspected_true[i] = self.t
                        if self.first_inspect_t is None:
                            self.first_inspect_t = self.t
                        self.last_inspect_t = self.t
                        frac_left = max(0.0, 1.0 - self.t / c.max_time_s)
                        self._r += c.r_inspect + c.r_inspect_speed * frac_left
            # Mission 2 rule: distance from the NOMINAL navigation goal to the entry
            nx, ny = cd["nominal"]
            if snap_ok and math.hypot(nx - e["x"], ny - e["y"]) <= c.visited_radius:
                e["inspected"] = True
            else:
                self._register_attempt_failure(e)

    # ------------------------------------------------------------------
    # observation
    # ------------------------------------------------------------------
    def _obs(self):
        c = self.cfg
        rx, ry = self.pos
        F = np.zeros((self.K, N_CAND_FEATURES), dtype=np.float32)
        paths = [s["path_len"] for s in self._slots if s is not None]
        pmax = max(paths) if paths else 1.0
        for i, s in enumerate(self._slots):
            if s is None:
                continue
            is_t = s["kind"] == "target"
            path_clear = float(self._clear_known[s["path"] // self.shape[1],
                                                 s["path"] % self.shape[1]].min())
            nv = min([math.hypot(s["x"] - vx, s["y"] - vy) for vx, vy in self.visited_goals],
                     default=10.0)
            F[i] = [
                1.0 if is_t else 0.0,
                min(s["path_len"] / 30.0, 2.0),
                s["path_len"] / max(pmax, 1e-6),
                s["ig"] / c.ig_rays,
                min(s["size"] / 60.0, 1.0),
                np.clip((s["x"] - rx) / 30.0, -1, 1),
                np.clip((s["y"] - ry) / 30.0, -1, 1),
                s["conf"],
                s["attempts"] / c.max_investigate_attempts,
                min(self._clear_known[s["r"], s["c"]] / 1.5, 1.0),
                min(path_clear / 1.5, 1.0),
                min(s["offset"] / c.visited_radius, 2.0) if is_t else 0.0,
                min(nv / 10.0, 1.0),
            ]
        known_free = float((self.known == 0).sum()) * c.res * c.res
        G = np.array([
            min(known_free / 400.0, 2.0),
            sum(1 for s_ in self._slots[:c.max_frontier_slots] if s_ is not None) / c.max_frontier_slots,
            self.battery / 100.0,
            self.t / c.max_time_s,
            self._n_avail_targets / c.max_target_slots,
            sum(1 for e in self.entries if e["inspected"]) / 10.0,
            sum(1 for e in self.entries if e["abandoned"]) / 10.0,
            self.stuck_events / 10.0,
            self.decisions / c.max_decisions,
            self.empty_cycles / c.empty_cycles_before_done,
        ], dtype=np.float32)
        return {"cand": F, "mask": self._mask.copy(), "glob": np.clip(G, -2, 2)}

    # ------------------------------------------------------------------
    # metrics (same names as your Mission 2 summary where possible)
    # ------------------------------------------------------------------
    def metrics(self):
        n_true = len(self.victims)
        db_detected = len(self.entries)
        db_inspected = sum(1 for e in self.entries if e["inspected"])
        db_abandoned = sum(1 for e in self.entries if e["abandoned"])
        return dict(
            layout=self.layout.name, completed=self.completed, end_reason=self.end_reason,
            duration_sec=round(self.t, 1), distance_traveled_m=round(self.dist, 2),
            final_coverage_pct=round(100 * self._coverage(), 2),
            decisions=self.decisions, stuck_events=self.stuck_events,
            true_targets=n_true, true_detected=len(self.detected_true),
            true_inspected=len(self.inspected_true),
            true_inspected_frac=len(self.inspected_true) / max(n_true, 1),
            db_entries_detected=db_detected, db_entries_inspected=db_inspected,
            db_entries_abandoned=db_abandoned,
            abandonment_rate=db_abandoned / max(db_detected, 1),
            first_detection_sec=self.first_detect_t,
            first_inspection_sec=self.first_inspect_t,
            last_inspection_sec=self.last_inspect_t,
        )

    def _info(self):
        return {"action_mask": self._mask.copy(), "metrics": self.metrics()}

    def save_png(self, path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        R, C = self.shape
        img = np.zeros((R, C, 3), dtype=np.float32)
        img[self.known == -1] = (0.55, 0.55, 0.55)
        img[self.known == 0] = (1, 1, 1)
        img[self.known == 100] = (0, 0, 0)
        img[(self.occ) & (self.known == -1)] = (0.35, 0.35, 0.40)
        ext = [self.ox, self.ox + C * self.cfg.res, self.oy, self.oy + R * self.cfg.res]
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.imshow(img, origin="lower", extent=ext)
        tr = np.array(self.trace)
        ax.plot(tr[:, 0], tr[:, 1], "b-", lw=0.8)
        for i, (vx, vy) in enumerate(self.victims):
            ax.plot(vx, vy, "g*" if i in self.inspected_true else "r*", ms=11)
        for e in self.entries:
            ax.plot(e["x"], e["y"], {"real": "go", "dup": "yo", "fake": "mo"}[e["kind"]],
                    ms=4, mfc="none")
        ax.set_title(f"{self.layout.name}  t={self.t:.0f}s  cov={100*self._coverage():.0f}%  "
                     f"inspected={len(self.inspected_true)}/{len(self.victims)}")
        fig.savefig(path, dpi=90, bbox_inches="tight")
        plt.close(fig)
