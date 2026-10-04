#!/usr/bin/env python3
"""
ExplorerRLNode - runs your EXISTING Mission 2 explorer node, but with a switch
for who picks the next goal:

    policy_mode:=mission2   -> your ORIGINAL Mission 2 node, untouched (reference)
    policy_mode:=mission2c  -> Mission 2's rule (targets first by Nav2 path,
                               else argmax 3*IG - beta*cost) applied to the SAME
                               candidate set the RL policy sees   <- fair baseline
    policy_mode:=rl         -> the trained RL policy chooses among that same
                               candidate set

Everything else (Nav2, stuck watchdog, LiDAR recovery, YOLO / semantic DB,
inspection, CSV result files) is the original code, untouched, because this
class INHERITS from your node. Put this file, rl_policy_np.py and policy.npz
next to your explorer node file, then run (ROS sourced):

    python3 explorer_rl_node.py --ros-args \
        -p policy_mode:=rl -p rl_model_path:=/full/path/policy.npz \
        -p experiment_name:=rl_run01

(use policy_mode:=mission2c for the fair baseline).
If your original node lives in a differently named file, set
EXPLORER_BASE_MODULE=<that module name> before running.
"""
import csv
import importlib
import math
import os
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathToPose

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_policy_np import NumpyCandidatePolicy  # noqa: E402


def _load_base():
    names = [os.environ.get("EXPLORER_BASE_MODULE"), "explorer_node", "explorer",
             "mission2_explorer", "explorer_mission2"]
    for n in [x for x in names if x]:
        try:
            return importlib.import_module(n).ExplorerNode
        except (ImportError, AttributeError):
            continue
    raise ImportError(
        "Could not import your ExplorerNode. Put this file next to your explorer node "
        "file and set EXPLORER_BASE_MODULE=<file name without .py>.")


ExplorerNode = _load_base()

N_FRONTIER_SLOTS = 8     # = Mission 2's top-8-by-IG frontier set
N_TARGET_SLOTS = 8
K_SLOTS = N_FRONTIER_SLOTS + N_TARGET_SLOTS
SIM_RES = 0.2          # grid resolution the policy was trained at
SIM_MAX_DECISIONS = 300.0


class ExplorerRLNode(ExplorerNode):
    def __init__(self):
        super().__init__()
        self.declare_parameter('policy_mode', 'rl')
        self.declare_parameter('rl_model_path', 'policy.npz')
        self.declare_parameter('rl_time_norm_s', 0.0)   # 0 = use value stored in policy.npz

        self.policy_mode = self.get_parameter('policy_mode').value
        self.rl_time_norm_s = float(self.get_parameter('rl_time_norm_s').value)
        self.rl_policy = None
        if self.policy_mode == 'rl':
            path = self.get_parameter('rl_model_path').value
            try:
                self.rl_policy = NumpyCandidatePolicy(path)
                self.get_logger().info(f"[RL] loaded policy from {path}")
            except Exception as exc:
                self.get_logger().error(
                    f"[RL] could not load {path}: {exc} -> falling back to mission2")
                self.policy_mode = 'mission2'
        if self.rl_time_norm_s <= 0.0:
            self.rl_time_norm_s = float(getattr(self.rl_policy, 'time_norm_s', 2400.0))
        self.rl_max_decisions = float(getattr(self.rl_policy, 'max_decisions', SIM_MAX_DECISIONS))
        self.get_logger().info(
            f"[RL] POLICY MODE = {self.policy_mode} (time norm {self.rl_time_norm_s:.0f} s)")

        self.stuck_events_count = 0
        self.rl_decision_count = 0
        self.rl_visited_goals = []

        self.rl_cand_log_path = os.path.join(self.run_dir, 'rl_candidates.csv')
        self.rl_glob_log_path = os.path.join(self.run_dir, 'rl_globals.csv')
        with open(self.rl_cand_log_path, 'w', newline='') as f:
            csv.writer(f).writerow(['decision', 'time_sec', 'slot', 'kind', 'x', 'y', 'path_len',
                                    'chosen'] + [f'f{i}' for i in range(13)])
        with open(self.rl_glob_log_path, 'w', newline='') as f:
            csv.writer(f).writerow(['decision', 'time_sec'] + [f'g{i}' for i in range(10)])
        self.rl_log_path = os.path.join(self.run_dir, 'rl_decisions.csv')
        with open(self.rl_log_path, 'w', newline='') as f:
            csv.writer(f).writerow([
                'policy_mode', 'time_sec', 'decision', 'n_candidates', 'chosen_kind',
                'goal_x', 'goal_y', 'path_len_m', 'prob_of_choice', 'target_id'])

    # ------------------------------------------------------------------
    def _handle_stuck_navigation(self):
        if not self.recovery_in_progress:
            self.stuck_events_count += 1
        super()._handle_stuck_navigation()

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    def explore(self):
        if self.policy_mode == 'mission2':
            return super().explore()

        if self.exploration_done:
            return
        if self.map_data is None:
            self.get_logger().info("Waiting for map...")
            return
        if (self.is_navigating or self.is_evaluating or self.is_direct_targeting
                or self.recovery_in_progress):
            return
        robot_pose = self.get_robot_pose()
        if robot_pose is None:
            return
        robot_x, robot_y = robot_pose

        width = self.map_data.info.width
        height = self.map_data.info.height
        map_array = np.array(self.map_data.data, dtype=np.int8).reshape((height, width))
        self.final_coverage_pct = self.calculate_map_coverage(map_array)

        clusters = self.cluster_frontiers(self.find_frontier_cells(map_array))
        targets_pending = any(self._target_is_available(t) for t in self.semantic_targets)

        # same completion rule as Mission 2
        if targets_pending:
            self.empty_cycle_count = 0
        elif not clusters or not self.has_unvisited_frontier(clusters):
            self.empty_cycle_count += 1
        else:
            self.empty_cycle_count = 0

        if self.empty_cycle_count >= self.empty_cycles_before_done:
            self.get_logger().info("==================================================")
            self.get_logger().info(f"MISSION COMPLETE ({self.policy_mode} mode)")
            self.get_logger().info("==================================================")
            self.exploration_done = True
            self.save_summary_results()
            return

        self.rl_decide(clusters, map_array, robot_x, robot_y)

    # ------------------------------------------------------------------
    # geometry helpers (all in metres, so independent of map resolution)
    # ------------------------------------------------------------------
    def _rc(self, x, y):
        r, c = self.world_to_grid(x, y)
        return int(round(r)), int(round(c))

    def _clearance_at(self, map_array, x, y, max_m=1.5):
        res = float(self.map_data.info.resolution)
        r, c = self._rc(x, y)
        R = int(math.ceil(max_m / res))
        rows, cols = map_array.shape
        r0, r1, c0, c1 = max(0, r - R), min(rows, r + R + 1), max(0, c - R), min(cols, c + R + 1)
        if r0 >= r1 or c0 >= c1:
            return max_m
        rr, cc = np.nonzero(map_array[r0:r1, c0:c1] >= self.occupied_threshold)
        if rr.size == 0:
            return max_m
        return min(float(np.hypot(rr + r0 - r, cc + c0 - c).min()) * res, max_m)

    def _target_free_offset(self, map_array, tx, ty, radius_m=1.5, clr_m=0.3):
        """Distance from a target entry to the nearest known-free cell that is
        at least clr_m from obstacles (mirrors the training simulator)."""
        res = float(self.map_data.info.resolution)
        r, c = self._rc(tx, ty)
        R = int(math.ceil(radius_m / res))
        m = int(math.ceil(clr_m / res))
        rows, cols = map_array.shape
        r0, r1 = max(0, r - R - m), min(rows, r + R + m + 1)
        c0, c1 = max(0, c - R - m), min(cols, c + R + m + 1)
        if r0 >= r1 or c0 >= c1:
            return radius_m
        win = map_array[r0:r1, c0:c1]
        occ = win >= self.occupied_threshold
        row_d = occ.copy()
        for d in range(1, m + 1):
            up = np.zeros_like(occ)
            up[:-d, :] = occ[d:, :]
            dn = np.zeros_like(occ)
            dn[d:, :] = occ[:-d, :]
            row_d |= up | dn
        full = row_d.copy()
        for d in range(1, m + 1):
            lf = np.zeros_like(row_d)
            lf[:, :-d] = row_d[:, d:]
            rt = np.zeros_like(row_d)
            rt[:, d:] = row_d[:, :-d]
            full |= lf | rt
        valid = (win == 0) & ~full
        if not valid.any():
            return radius_m
        rr, cc = np.nonzero(valid)
        d = np.hypot(rr + r0 - r, cc + c0 - c) * res
        return min(float(d.min()), radius_m)

    # ------------------------------------------------------------------
    # async Nav2 path request that also returns the path poses
    # ------------------------------------------------------------------
    def _get_path_async(self, rx, ry, gx, gy, callback):
        """callback(length_or_None, [(x, y), ...] or None)"""
        if not self.compute_path_client.wait_for_server(timeout_sec=0.5):
            callback(None, None)
            return
        start, goal = PoseStamped(), PoseStamped()
        for ps, (x, y) in ((start, (rx, ry)), (goal, (gx, gy))):
            ps.header.frame_id = self.map_frame
            ps.header.stamp = self.get_clock().now().to_msg()
            ps.pose.position.x, ps.pose.position.y = x, y
            ps.pose.orientation.w = 1.0
        req = ComputePathToPose.Goal()
        req.start, req.goal, req.use_start = start, goal, True

        self._path_token += 1
        token = self._path_token
        self._pending_path_requests[token] = {
            "token": token, "start_time": time.time(), "goal_handle": None,
            "callback": (lambda length, cb=callback: cb(length, None)),
            "goal_x": gx, "goal_y": gy}

        def finish(length, poses):
            if token not in self._pending_path_requests:
                return
            del self._pending_path_requests[token]
            callback(length, poses)

        def on_result(future):
            result = future.result()
            if result is None:
                finish(None, None)
                return
            path = result.result.path
            if not path.poses or len(path.poses) < 2:
                finish(None, None)
                return
            pts = [(p.pose.position.x, p.pose.position.y) for p in path.poses]
            length = sum(math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
                         for i in range(1, len(pts)))
            finish(length, pts)

        def on_goal_response(future):
            gh = future.result()
            if gh is None or not gh.accepted:
                finish(None, None)
                return
            if token in self._pending_path_requests:
                self._pending_path_requests[token]["goal_handle"] = gh
            gh.get_result_async().add_done_callback(on_result)

        self.compute_path_client.send_goal_async(req).add_done_callback(on_goal_response)

    # ------------------------------------------------------------------
    # RL decision
    # ------------------------------------------------------------------
    def rl_decide(self, clusters, map_array, rx, ry):
        res = float(self.map_data.info.resolution)
        robot_r, robot_c = self.world_to_grid(rx, ry)
        unsafe_mask = self.compute_unsafe_mask(map_array)

        # ---- frontier candidates: IDENTICAL to Mission 2's choose_frontier() ----
        #  clusters -> drop visited/blacklisted -> 15 largest -> goal cell + IG
        #  -> keep the 8 with the highest information gain
        avail = []
        for cl in clusters:
            x, y = self.grid_to_world(*cl["goal_cell"])
            if self.is_already_visited(x, y) or self.is_blacklisted(x, y):
                continue
            avail.append(cl)
        n_avail_clusters = len(avail)
        avail.sort(key=lambda c: c["size"], reverse=True)
        cands = []
        for cl in avail[:15]:
            goal = self.pick_goal_cell(cl, robot_r, robot_c, unsafe_mask)
            if goal is None:
                continue
            r, c = goal
            x, y = self.grid_to_world(r, c)
            if math.hypot(x - rx, y - ry) < self.min_frontier_goal_distance_m:
                continue
            if self.is_already_visited(x, y) or self.is_blacklisted(x, y):
                continue
            ig = self.estimate_information_gain(map_array, r, c)
            cands.append(dict(kind='frontier', x=x, y=y, size=cl["size"] * res / SIM_RES,
                              ig=ig, conf=0.0, attempts=0, target_id=None, offset=0.0))
        cands.sort(key=lambda c: c["ig"], reverse=True)
        cands = cands[:N_FRONTIER_SLOTS]

        # ---- target candidates ----------------------------------------
        avail_t = [t for t in self.semantic_targets if self._target_is_available(t)]
        n_avail_targets = len(avail_t)
        for t in avail_t:
            sx, sy = self.compute_standoff_point(rx, ry, t.x, t.y)
            cands.append(dict(kind='target', x=sx, y=sy, size=0, ig=0, conf=float(t.confidence),
                              attempts=self.target_attempt_counts.get(t.id, 0), target_id=t.id,
                              offset=self._target_free_offset(map_array, t.x, t.y)))

        if not cands:
            self.get_logger().warning("[RL] no candidates this cycle - recovery")
            self.trigger_recovery_spin()
            return

        # ---- Nav2 path length (+ poses) for every candidate -----------
        self.is_evaluating = True
        done = [0]
        n = len(cands)

        def on_path(c, length, poses):
            c['path_len'] = length if (length is not None and length > 0.0) else None
            c['poses'] = poses
            done[0] += 1
            if done[0] == n:
                try:
                    self._rl_finish(cands, map_array, rx, ry, n_avail_clusters, n_avail_targets)
                except Exception as exc:
                    self.get_logger().error(f"[RL] decision failed: {exc}")
                    self.is_evaluating = False

        for c in cands:
            self._get_path_async(rx, ry, c['x'], c['y'],
                                 lambda length, poses, c=c: on_path(c, length, poses))

    def _rl_features(self, cands, map_array, rx, ry, n_avail_clusters, n_avail_targets):
        fr = [c for c in cands if c['kind'] == 'frontier']
        tg = sorted([c for c in cands if c['kind'] == 'target'], key=lambda c: c['path_len'])
        slots = [None] * K_SLOTS
        for i, c in enumerate(fr[:N_FRONTIER_SLOTS]):
            slots[i] = c
        for i, c in enumerate(tg[:N_TARGET_SLOTS]):
            slots[N_FRONTIER_SLOTS + i] = c

        pmax = max([s['path_len'] for s in slots if s is not None], default=1.0)
        F = np.zeros((K_SLOTS, 13), dtype=np.float64)
        for i, s in enumerate(slots):
            if s is None:
                continue
            is_t = s['kind'] == 'target'
            goal_clear = self._clearance_at(map_array, s['x'], s['y'])
            poses = s.get('poses') or []
            stride = max(1, len(poses) // 60)
            path_clear = min([self._clearance_at(map_array, px, py) for px, py in poses[::stride]],
                             default=goal_clear)
            nv = min([math.hypot(s['x'] - vx, s['y'] - vy) for vx, vy in self.rl_visited_goals],
                     default=10.0)
            F[i] = [
                1.0 if is_t else 0.0,
                min(s['path_len'] / 30.0, 2.0),
                s['path_len'] / max(pmax, 1e-6),
                s['ig'] / float(max(int(self.info_gain_num_rays), 1)),
                min(s['size'] / 60.0, 1.0),
                float(np.clip((s['x'] - rx) / 30.0, -1, 1)),
                float(np.clip((s['y'] - ry) / 30.0, -1, 1)),
                s['conf'],
                s['attempts'] / max(self.max_investigate_attempts, 1),
                min(goal_clear / 1.5, 1.0),
                min(path_clear / 1.5, 1.0),
                min(s['offset'] / self.visited_radius_m, 2.0) if is_t else 0.0,
                min(nv / 10.0, 1.0),
            ]
        res = float(self.map_data.info.resolution)
        known_free = float(np.count_nonzero(map_array == 0)) * res * res
        elapsed = time.time() - self.experiment_start_time
        G = np.clip(np.array([
            min(known_free / 400.0, 2.0),
            sum(1 for s_ in slots[:N_FRONTIER_SLOTS] if s_ is not None) / float(N_FRONTIER_SLOTS),
            self.battery_pct / 100.0,
            elapsed / self.rl_time_norm_s,
            n_avail_targets / float(N_TARGET_SLOTS),
            len(self.inspected_target_ids) / 10.0,
            len(self.abandoned_target_ids) / 10.0,
            self.stuck_events_count / 10.0,
            self.rl_decision_count / self.rl_max_decisions,
            self.empty_cycle_count / float(max(self.empty_cycles_before_done, 1)),
        ], dtype=np.float64), -2.0, 2.0)
        mask = np.array([s is not None for s in slots], dtype=bool)
        return slots, F, mask, G

    def _mission2_choice(self, slots):
        """Mission 2's hand-designed rule on the canonical candidate set."""
        tg = [(i, s) for i, s in enumerate(slots) if s is not None and s['kind'] == 'target']
        if tg:                                   # targets first: shortest Nav2 path
            return min(tg, key=lambda t: t[1]['path_len'])[0]
        fr = [(i, s) for i, s in enumerate(slots) if s is not None]
        norm = max(max(s['path_len'] for _, s in fr), 1e-5)
        beta_eff = self.beta * (1.0 + (1.0 - self.battery_pct / 100.0))
        denom = max(int(self.info_gain_num_rays), 1)
        return max(fr, key=lambda t: self.alpha * min(t[1]['ig'] / denom, 1.0)
                   - beta_eff * min(t[1]['path_len'] / norm, 1.0))[0]

    def _rl_finish(self, cands, map_array, rx, ry, n_avail_clusters, n_avail_targets):
        self.is_evaluating = False
        valid = [c for c in cands if c['path_len'] is not None]

        # Same fallbacks as Mission 2 when nothing is reachable
        if not valid:
            self.get_logger().warning("[RL] no reachable candidates - recovery")
            for c in cands:
                if c['kind'] == 'target':
                    self._register_target_attempt_failure(c['target_id'])
            self.trigger_recovery_spin()
            return

        slots, F, mask, G = self._rl_features(valid, map_array, rx, ry,
                                              n_avail_clusters, n_avail_targets)
        if self.policy_mode == 'rl':
            idx, probs = self.rl_policy.act(F, mask, G)
        else:
            idx = self._mission2_choice(slots)
            probs = np.zeros(K_SLOTS)
            probs[idx] = 1.0
        ch = slots[idx]
        t_now = round(time.time() - self.experiment_start_time, 2)
        with open(self.rl_cand_log_path, 'a', newline='') as f:
            w = csv.writer(f)
            for i, sl in enumerate(slots):
                if sl is not None:
                    w.writerow([self.rl_decision_count + 1, t_now, i, sl['kind'], round(sl['x'], 3),
                                round(sl['y'], 3), round(sl['path_len'], 2), int(i == idx)]
                               + [round(float(v), 4) for v in F[i]])
        with open(self.rl_glob_log_path, 'a', newline='') as f:
            csv.writer(f).writerow([self.rl_decision_count + 1, t_now] + [round(float(v), 4) for v in G])
        self.rl_decision_count += 1
        self.rl_visited_goals.append((ch['x'], ch['y']))

        with open(self.rl_log_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                self.policy_mode, round(time.time() - self.experiment_start_time, 2),
                self.rl_decision_count, int(mask.sum()), ch['kind'], round(ch['x'], 3),
                round(ch['y'], 3), round(ch['path_len'], 2), round(float(probs[idx]), 3),
                ch['target_id'] if ch['target_id'] is not None else ''])
        self.get_logger().info(
            f"[RL] decision {self.rl_decision_count}: {ch['kind']} "
            f"({ch['x']:.2f}, {ch['y']:.2f}) path={ch['path_len']:.1f} m "
            f"p={probs[idx]:.2f} of {int(mask.sum())} candidates")

        if ch['kind'] == 'target':
            self.current_target_id = ch['target_id']
            self.navigate_to(ch['x'], ch['y'], ch['path_len'],
                             goal_type='target', target_id=ch['target_id'])
        else:
            self.navigate_to(ch['x'], ch['y'], ch['path_len'], goal_type='frontier')


def main(args=None):
    rclpy.init(args=args)
    node = ExplorerRLNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_summary_results()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
