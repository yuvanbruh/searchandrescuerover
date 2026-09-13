

import csv
import math
import os
import re
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration

from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, Point, PointStamped
from nav2_msgs.action import NavigateToPose, ComputePathToPose, Spin
from visualization_msgs.msg import Marker, MarkerArray
from std_srvs.srv import Empty

from semantic_interfaces.msg import SemanticTargetArray

import tf2_ros
from tf2_ros import TransformException


class ExplorerNode(Node):
    """
    Mission 2: Search & Inspect Specific Targets

    High-level policy:

        1. If there are known, uninspected, non-abandoned targets
           above the noise floor:
               -> navigate directly to whichever has the shortest
                  actual Nav2 path (no confidence weighting - see
                  min_target_confidence below for the one place
                  confidence is used)
               -> inspect it
               -> resume exploration

        2. If there are no such targets:
               -> perform normal frontier exploration using

                   U = alpha * InformationGain_norm
                       - effective_beta * TravelCost_norm

        3. During frontier exploration, YOLO may detect new targets.
           Their coordinates are stored by the semantic database.
           On the next decision cycle, the target becomes the
           navigation objective.

    Target selection is deterministic and confidence-free:

        T* = argmin_i D_Nav2(i)

    over all known, uninspected, non-abandoned candidates whose
    confidence is at least min_target_confidence. That parameter is
    a NOISE FLOOR only (filters obvious garbage detections, e.g. a
    single-frame misclassification) - it is not a decision gate on
    whether to bother investigating a real-but-uncertain detection.
    Once a candidate clears the floor, confidence plays no further
    role in whether or when it gets investigated.

    Completion guarantee: a target that is repeatedly approached but
    never resolves (never comes within visited_radius_m - e.g. a
    false positive with nothing physically there, or bad
    localization) does not block the mission forever. After
    max_investigate_attempts failed approaches it is marked
    ABANDONED (tracked separately from successfully inspected
    targets) and excluded from further consideration, so exploration
    can still reach completion.

    IMPORTANT:
        There is NO semantic frontier score.
        There is NO lambda_semantic.
        Semantic perception is used for target discovery/localization
        and (via the noise floor only) filtering, not for modifying
        frontier utility or for weighting target priority.

    Research comparison:

        Hand-designed policy:
            EXPLORE <-> INVESTIGATE

        vs

        Future RL policy:
            EXPLORE <-> INVESTIGATE

    The current file implements only the hand-designed Mission-2
    baseline.
    """

    def __init__(self):
        super().__init__('explorer')

        # ============================================================
        # ROS2 PARAMETERS
        # ============================================================

        self.declare_parameter('experiment_name', 'exp_default')

        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')

        self.declare_parameter('explore_period_sec', 5.0)

        self.declare_parameter('min_cluster_size', 4)
        self.declare_parameter('min_frontier_goal_distance_m', 0.75)

        # Distance at which a target is considered physically reached
        # and can be marked as inspected.
        self.declare_parameter('visited_radius_m', 0.8)

        self.declare_parameter('obstacle_margin_cells', 3)
        self.declare_parameter('occupied_threshold', 65)

        self.declare_parameter('blacklist_timeout_sec', 30.0)

        self.declare_parameter('empty_cycles_before_done', 3)

        self.declare_parameter('path_plan_timeout_sec', 3.0)
        self.declare_parameter('path_plan_watchdog_period_sec', 0.5)

        # ============================================================
        # MISSION 2
        # ============================================================

        self.declare_parameter('target_class', 'sports ball')

        # Noise floor only - NOT a "should we investigate" gate.
        # Filters obvious garbage (e.g. a single-frame hallucinated
        # misclassification) before it ever becomes a target
        # candidate. Deliberately low: real-but-uncertain detections
        # (occlusion, distance, bad lighting) should still be
        # investigated, since missing a real target is worse than an
        # extra trip to a false one.
        self.declare_parameter('min_target_confidence', 0.10)

        # Completion guarantee: a target repeatedly approached but
        # never resolved (never within visited_radius_m) is marked
        # ABANDONED after this many failed attempts, so it stops
        # blocking mission completion.
        self.declare_parameter('max_investigate_attempts', 2)

        # ============================================================
        # FRONTIER UTILITY
        #
        # No semantic term.
        #
        # U = alpha * IG_norm - beta * Cost_norm
        # ============================================================

        self.declare_parameter('alpha_info_gain', 3.0)
        self.declare_parameter('beta_travel_cost', 0.25)

        # Information gain is estimated from the actual map geometry
        # along LiDAR-like rays from the candidate viewpoint. We count
        # unknown cells that are directly visible through currently-known
        # free space, rather than all unknown cells inside a square.
        self.declare_parameter('info_gain_max_range_m', 10.0)
        self.declare_parameter('info_gain_num_rays', 180)

        # Stuck/progress recovery
        self.declare_parameter('progress_check_period_sec', 0.5)
        self.declare_parameter('stuck_timeout_sec', 6.0)
        self.declare_parameter('progress_min_distance_m', 0.025)
        self.declare_parameter('progress_start_grace_sec', 3.0)

        # LiDAR-directed recovery
        self.declare_parameter('recovery_scan_topic', '/scan')
        self.declare_parameter('recovery_scan_sectors', 8)
        self.declare_parameter('recovery_sector_min_range_m', 0.12)
        self.declare_parameter('recovery_sector_max_range_m', 3.0)
        self.declare_parameter('recovery_clearance_margin_m', 0.20)

        self.declare_parameter('initial_battery_pct', 100.0)
        self.declare_parameter('battery_drain_pct_per_meter', 0.5)

        # ============================================================
        # SEMANTIC DATABASE
        # ============================================================

        self.declare_parameter('semantic_targets_topic', '/semantic_database')
        self.declare_parameter('mark_visited_topic', '/semantic_database/mark_visited')

        # ============================================================
        # READ PARAMETERS
        # ============================================================

        self.experiment_name = self.get_parameter('experiment_name').value
        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.explore_period_sec = self.get_parameter('explore_period_sec').value
        self.min_cluster_size = self.get_parameter('min_cluster_size').value
        self.min_frontier_goal_distance_m = self.get_parameter(
            'min_frontier_goal_distance_m').value
        self.visited_radius_m = self.get_parameter('visited_radius_m').value
        self.obstacle_margin_cells = self.get_parameter('obstacle_margin_cells').value
        self.occupied_threshold = self.get_parameter('occupied_threshold').value
        self.blacklist_timeout_sec = self.get_parameter('blacklist_timeout_sec').value
        self.empty_cycles_before_done = self.get_parameter('empty_cycles_before_done').value
        self.path_plan_timeout_sec = self.get_parameter('path_plan_timeout_sec').value
        self.path_plan_watchdog_period_sec = self.get_parameter(
            'path_plan_watchdog_period_sec').value

        self.target_class = self.get_parameter('target_class').value
        self.min_target_confidence = self.get_parameter('min_target_confidence').value
        self.max_investigate_attempts = self.get_parameter('max_investigate_attempts').value

        self.alpha = self.get_parameter('alpha_info_gain').value
        self.beta = self.get_parameter('beta_travel_cost').value
        self.info_gain_max_range_m = self.get_parameter('info_gain_max_range_m').value
        self.info_gain_num_rays = int(self.get_parameter('info_gain_num_rays').value)

        self.progress_check_period_sec = self.get_parameter(
            'progress_check_period_sec').value
        self.stuck_timeout_sec = self.get_parameter('stuck_timeout_sec').value
        self.progress_min_distance_m = self.get_parameter(
            'progress_min_distance_m').value
        self.progress_start_grace_sec = self.get_parameter(
            'progress_start_grace_sec').value

        self.recovery_scan_topic = self.get_parameter(
            'recovery_scan_topic').value
        self.recovery_scan_sectors = int(self.get_parameter(
            'recovery_scan_sectors').value)
        self.recovery_sector_min_range_m = self.get_parameter(
            'recovery_sector_min_range_m').value
        self.recovery_sector_max_range_m = self.get_parameter(
            'recovery_sector_max_range_m').value
        self.recovery_clearance_margin_m = self.get_parameter(
            'recovery_clearance_margin_m').value

        self.battery_pct = self.get_parameter('initial_battery_pct').value
        self.battery_drain_pct_per_meter = self.get_parameter('battery_drain_pct_per_meter').value

        self.semantic_targets_topic = self.get_parameter('semantic_targets_topic').value
        self.mark_visited_topic = self.get_parameter('mark_visited_topic').value

        self.get_logger().info("==================================================")
        self.get_logger().info("ARES Mission 2: SEARCH & INSPECT")
        self.get_logger().info(f"Target class = {self.target_class}")
        self.get_logger().info(f"Frontier utility = {self.alpha}*VISIBLE_IG - {self.beta}*TravelCost")
        self.get_logger().info(
            f"Visible IG = {self.info_gain_num_rays} LiDAR rays, "
            f"max range={self.info_gain_max_range_m:.1f}m"
        )
        self.get_logger().info(
            f"Recovery = LiDAR direction selection, "
            f"stuck timeout={self.stuck_timeout_sec:.1f}s"
        )
        self.get_logger().info(f"Inspection radius = {self.visited_radius_m:.2f} m")
        self.get_logger().info(f"Noise floor (min_target_confidence) = {self.min_target_confidence}")
        self.get_logger().info(f"Max investigate attempts before abandon = {self.max_investigate_attempts}")
        self.get_logger().info("Target selection = argmin Nav2 path length (no confidence weighting)")
        self.get_logger().info("Semantic frontier score = DISABLED")
        self.get_logger().info("Lambda semantic = REMOVED")
        self.get_logger().info("==================================================")

        # ============================================================
        # STATE
        # ============================================================

        self.map_data = None

        self.is_navigating = False
        self.is_evaluating = False
        self.is_direct_targeting = False

        self.visited_frontiers_world = []
        self.blacklist = {}

        self.current_goal_handle = None
        self.current_goal_x = None
        self.current_goal_y = None
        self.current_goal_type = None
        self.current_target_id = None

        # Navigation progress watchdog state
        self.nav_goal_started_time = None
        self.last_progress_pose = None
        self.last_progress_time = None
        self.recovery_in_progress = False
        self.goal_cancelled_for_recovery = False

        # Latest 360-degree LiDAR scan used only for emergency recovery
        self.latest_recovery_scan = None

        self.empty_cycle_count = 0
        self.exploration_done = False

        # ============================================================
        # TARGET STATE
        # ============================================================

        self.semantic_targets = []

        self.all_detected_target_ids = set()
        self.inspected_target_ids = set()

        # Targets that repeatedly failed to resolve (never came
        # within visited_radius_m after max_investigate_attempts
        # approaches). Tracked separately from inspected_target_ids
        # so they don't count as successful finds in metrics, but
        # also don't block mission completion.
        self.abandoned_target_ids = set()
        self.target_attempt_counts = {}  # target_id -> failed attempt count

        # ============================================================
        # PATH PLANNING
        # ============================================================

        self._path_token = 0
        self._pending_path_requests = {}

        self._eval_results = []

        # ============================================================
        # EXPERIMENT
        # ============================================================

        self.experiment_start_time = time.time()
        self.results_saved = False
        self.final_coverage_pct = 0.0

        # ============================================================
        # OUTPUT FILES
        # ============================================================

        # ============================================================
        # PER-RUN RESULTS
        # ============================================================
        # Every launch gets its own run_001, run_002, ... directory.
        # Existing runs are never overwritten or mixed together.
        results_root = os.path.expanduser('~/ares_results')
        os.makedirs(results_root, exist_ok=True)

        existing_run_numbers = []
        for entry in os.listdir(results_root):
            match = re.fullmatch(r'run_(\d+)', entry)
            if match and os.path.isdir(os.path.join(results_root, entry)):
                existing_run_numbers.append(int(match.group(1)))

        self.run_number = (
            max(existing_run_numbers) + 1
            if existing_run_numbers else 1
        )

        self.run_dir = os.path.join(
            results_root,
            f'run_{self.run_number:03d}'
        )
        os.makedirs(self.run_dir, exist_ok=True)

        self.get_logger().info(
            f'Experiment results directory: {self.run_dir}'
        )

        self.decision_results_path = os.path.join(
            self.run_dir, 'frontier_decisions.csv'
        )
        self.summary_results_path = os.path.join(
            self.run_dir, 'experiment_summary.csv'
        )
        self.milestone_results_path = os.path.join(
            self.run_dir, 'target_milestones.csv'
        )

        # Create all three CSV files immediately at launch.
        # This means even a Ctrl+C before the first decision still
        # leaves a complete run directory with valid CSV headers.
        self._initialize_result_files()

        # ============================================================
        # FRONTIER METRICS
        # ============================================================

        self.frontier_decisions = 0
        self.total_distance_traveled = 0.0

        # ============================================================
        # TARGET METRICS
        # ============================================================

        self.targets_found_count = 0

        self.targets_detected_count = 0
        self.targets_inspected_count = 0
        self.targets_abandoned_count = 0

        self.first_target_detection_time_sec = None
        self.first_target_inspection_time_sec = None
        self.all_targets_inspection_time_sec = None

        self.detection_milestone_times = {}
        self.inspection_milestone_times = {}

        # Persist each milestone immediately so stopping the mission
        # cannot lose detection/inspection data.
        self.saved_detection_milestones = set()
        self.saved_inspection_milestones = set()

        # ============================================================
        # UTILITY METRICS
        # ============================================================

        self.utility_sum = 0.0
        self.info_gain_sum = 0.0
        self.travel_cost_sum = 0.0

        self.utility_samples = 0

        # Timing metrics for frontier exploration decision-making.
        self.frontier_timing_sum_ms = 0.0
        self.info_gain_timing_sum_ms = 0.0
        self.async_path_timing_sum_ms = 0.0
        self.decision_latency_sum_ms = 0.0
        self.timing_samples = 0
        self.decision_timing_samples = 0
        self.last_explore_sync_ms = 0.0

        # ============================================================
        # TF
        # ============================================================

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ============================================================
        # SUBSCRIPTIONS
        # ============================================================

        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)

        self.recovery_scan_sub = self.create_subscription(
            LaserScan,
            self.recovery_scan_topic,
            self.recovery_scan_callback,
            10)

        self.semantic_sub = self.create_subscription(
            SemanticTargetArray, self.semantic_targets_topic,
            self.semantic_callback, 10)

        # ============================================================
        # ACTION CLIENTS
        # ============================================================

        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.compute_path_client = ActionClient(self, ComputePathToPose, 'compute_path_to_pose')
        self.spin_client = ActionClient(self, Spin, 'spin')

        self.global_costmap_clear_client = self.create_client(
            Empty, '/global_costmap/clear_entirely_global_costmap')
        self.local_costmap_clear_client = self.create_client(
            Empty, '/local_costmap/clear_entirely_local_costmap')

        # ============================================================
        # PUBLISHERS
        # ============================================================

        self.marker_pub = self.create_publisher(MarkerArray, '/explorer/frontier_markers', 10)
        self.mark_visited_pub = self.create_publisher(PointStamped, self.mark_visited_topic, 10)

        # ============================================================
        # TIMERS
        # ============================================================

        self.timer = self.create_timer(self.explore_period_sec, self.explore)
        self.path_plan_watchdog_timer = self.create_timer(
            self.path_plan_watchdog_period_sec, self._check_path_plan_timeout)

        self.progress_watchdog_timer = self.create_timer(
            self.progress_check_period_sec, self._check_navigation_progress)

    # ================================================================
    # CALLBACKS
    # ================================================================

    def map_callback(self, msg):
        self.map_data = msg

    def recovery_scan_callback(self, msg):
        self.latest_recovery_scan = msg

    def semantic_callback(self, msg):
        """
        Receives target detections/localizations from YOLO +
        semantic database.

        Semantic information is NOT used to score frontiers, and
        confidence is NOT used to weight target priority. It is only
        used (a) to maintain the list of known targets and (b) as a
        noise floor - see min_target_confidence.
        """
        self.semantic_targets = list(msg.targets)

        matching_targets = [t for t in msg.targets if t.class_name == self.target_class]
        if not matching_targets:
            return

        previous_count = len(self.all_detected_target_ids)

        for target in matching_targets:
            self.all_detected_target_ids.add(target.id)

        new_count = len(self.all_detected_target_ids)
        self.targets_detected_count = new_count

        if previous_count == 0 and new_count > 0 and self.first_target_detection_time_sec is None:
            self.first_target_detection_time_sec = time.time() - self.experiment_start_time

        if new_count > previous_count:
            detection_time = time.time() - self.experiment_start_time
            for count in range(previous_count + 1, new_count + 1):
                if count not in self.detection_milestone_times:
                    self.detection_milestone_times[count] = detection_time
                    self.save_detection_milestone(count, detection_time)

            self.get_logger().info(f"Target detected | new total = {new_count}")

    # ================================================================
    # ROBOT POSE
    # ================================================================

    def get_robot_pose(self):
        try:
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, now,
                timeout=Duration(seconds=0.5))
            return trans.transform.translation.x, trans.transform.translation.y
        except TransformException as ex:
            self.get_logger().warning(
                f"Could not get robot pose ({self.map_frame} -> {self.base_frame}): {ex}")
            return None

    # ================================================================
    # FRONTIER DETECTION
    # ================================================================

    def find_frontier_cells(self, map_array):
        """
        Vectorized frontier detection: a frontier cell is known-free (0)
        with at least one unknown (-1) neighbor in its 3x3 window.
        Equivalent to the original nested-loop version, but computed via
        NumPy array shifts instead of per-cell Python iteration.
        """
        free_mask = (map_array == 0)
        unknown_mask = (map_array == -1)

        has_unknown_neighbor = np.zeros_like(unknown_mask, dtype=bool)

        shifts = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]

        for dr, dc in shifts:
            shifted = np.zeros_like(unknown_mask)
            src_r0, src_r1 = max(0, -dr), unknown_mask.shape[0] - max(0, dr)
            src_c0, src_c1 = max(0, -dc), unknown_mask.shape[1] - max(0, dc)
            dst_r0, dst_r1 = max(0, dr), unknown_mask.shape[0] - max(0, -dr)
            dst_c0, dst_c1 = max(0, dc), unknown_mask.shape[1] - max(0, -dc)
            shifted[dst_r0:dst_r1, dst_c0:dst_c1] = unknown_mask[
                src_r0:src_r1, src_c0:src_c1
            ]
            has_unknown_neighbor |= shifted

        frontier_mask = free_mask & has_unknown_neighbor

        # Exclude the 1-cell border, matching the original range(1, rows-1)
        frontier_mask[0, :] = False
        frontier_mask[-1, :] = False
        frontier_mask[:, 0] = False
        frontier_mask[:, -1] = False

        rows_idx, cols_idx = np.where(frontier_mask)
        return list(zip(rows_idx.tolist(), cols_idx.tolist()))

    def cluster_frontiers(self, frontier_cells):
        if not frontier_cells:
            return []

        cell_set = set(frontier_cells)
        visited = set()
        clusters = []

        for cell in frontier_cells:
            if cell in visited:
                continue

            stack = [cell]
            cluster = []
            visited.add(cell)

            while stack:
                r, c = stack.pop()
                cluster.append((r, c))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        neighbor = (r + dr, c + dc)
                        if neighbor in cell_set and neighbor not in visited:
                            visited.add(neighbor)
                            stack.append(neighbor)

            if len(cluster) >= self.min_cluster_size:
                mean_r = sum(p[0] for p in cluster) / len(cluster)
                mean_c = sum(p[1] for p in cluster) / len(cluster)
                medoid = min(cluster, key=lambda p: (p[0] - mean_r) ** 2 + (p[1] - mean_c) ** 2)

                clusters.append({"goal_cell": medoid, "size": len(cluster), "cells": cluster})

        return clusters

    # ================================================================
    # MAP UTILITIES
    # ================================================================

    def is_cell_safe(self, map_array, r, c):
        rows, cols = map_array.shape
        m = self.obstacle_margin_cells
        r0, r1 = max(0, r - m), min(rows, r + m + 1)
        c0, c1 = max(0, c - m), min(cols, c + m + 1)
        window = map_array[r0:r1, c0:c1]
        return not np.any(window >= self.occupied_threshold)

    def grid_to_world(self, r, c):
        res = self.map_data.info.resolution
        origin_x = self.map_data.info.origin.position.x
        origin_y = self.map_data.info.origin.position.y
        return c * res + origin_x, r * res + origin_y

    def world_to_grid(self, x, y):
        res = self.map_data.info.resolution
        origin_x = self.map_data.info.origin.position.x
        origin_y = self.map_data.info.origin.position.y
        return (y - origin_y) / res, (x - origin_x) / res

    # ================================================================
    # FRONTIER STATE
    # ================================================================

    def has_unvisited_frontier(self, clusters):
        """
        Determine whether at least one detected frontier cluster
        still represents a new, non-blacklisted exploration region.

        This is used only for mission-completion logic. It checks
        all detected clusters and does not run pick_goal_cell(),
        information-gain ray casting, or Nav2 path evaluation.
        """
        for cluster in clusters:
            r, c = cluster["goal_cell"]
            x, y = self.grid_to_world(r, c)

            if self.is_already_visited(x, y):
                continue

            if self.is_blacklisted(x, y):
                continue

            return True

        return False

    def is_already_visited(self, x, y):
        for vx, vy in self.visited_frontiers_world:
            if math.hypot(x - vx, y - vy) < self.visited_radius_m:
                return True
        return False

    def is_blacklisted(self, x, y):
        now = time.time()
        expired = [k for k, exp in self.blacklist.items() if exp < now]
        for k in expired:
            del self.blacklist[k]

        for bx, by in self.blacklist:
            if math.hypot(x - bx, y - by) < self.visited_radius_m:
                return True
        return False

    def blacklist_point(self, x, y):
        self.blacklist[(x, y)] = time.time() + self.blacklist_timeout_sec
        self.get_logger().warning(
            f"Blacklisting frontier ({x:.2f}, {y:.2f}) for {self.blacklist_timeout_sec:.0f}s")

    def compute_unsafe_mask(self, map_array):
        """
        Precompute, once per cycle, which cells are unsafe because they
        are within obstacle_margin_cells of an occupied cell.

        This preserves the original is_cell_safe() box-window semantics
        while avoiding repeated per-cell NumPy slicing inside
        pick_goal_cell().
        """
        occupied = map_array >= self.occupied_threshold
        m = self.obstacle_margin_cells

        # Dilate along rows first.
        row_dilated = occupied.copy()

        for d in range(1, m + 1):
            shifted_up = np.zeros_like(occupied)
            shifted_up[:-d, :] = occupied[d:, :]

            shifted_down = np.zeros_like(occupied)
            shifted_down[d:, :] = occupied[:-d, :]

            row_dilated |= shifted_up | shifted_down

        # Dilate along columns.
        full_dilated = row_dilated.copy()

        for d in range(1, m + 1):
            shifted_left = np.zeros_like(row_dilated)
            shifted_left[:, :-d] = row_dilated[:, d:]

            shifted_right = np.zeros_like(row_dilated)
            shifted_right[:, d:] = row_dilated[:, :-d]

            full_dilated |= shifted_left | shifted_right

        return full_dilated

    def pick_goal_cell(self, cluster, robot_r, robot_c, unsafe_mask):
        cells = cluster["cells"]

        resolution = float(self.map_data.info.resolution)

        if resolution <= 0.0:
            return None

        min_dist_cells = (
            self.min_frontier_goal_distance_m / resolution
        )

        if not cells:
            return None

        cells_arr = np.asarray(cells, dtype=np.int32)

        rs = cells_arr[:, 0]
        cs = cells_arr[:, 1]

        dist = np.hypot(
            rs - robot_r,
            cs - robot_c
        )

        safe = ~unsafe_mask[rs, cs]

        valid_mask = (
            (dist >= min_dist_cells) &
            safe
        )

        if not np.any(valid_mask):
            return None

        valid_rs = rs[valid_mask]
        valid_cs = cs[valid_mask]

        original_r, original_c = cluster["goal_cell"]

        d2 = (
            (valid_rs - original_r) ** 2 +
            (valid_cs - original_c) ** 2
        )

        best_idx = np.argmin(d2)

        return (
            int(valid_rs[best_idx]),
            int(valid_cs[best_idx])
        )


    def estimate_information_gain(self, map_array, r, c):
        """
        Vectorized ray-cast information gain.

        Preserves the original semantics:
          - Cast num_rays rays from the candidate cell.
          - Continue through known-free cells.
          - Stop at the first occupied cell.
          - Stop and count the first unknown cell.
          - Stop when leaving the map.
          - Count each visible unknown cell only once.
        """

        rows, cols = map_array.shape
        resolution = float(self.map_data.info.resolution)

        if resolution <= 0.0:
            return 0.0

        max_steps = max(
            1,
            int(self.info_gain_max_range_m / resolution)
        )

        num_rays = max(
            36,
            int(self.info_gain_num_rays)
        )

        # ---------------------------------------------------------
        # Ray directions
        # ---------------------------------------------------------

        angles = (
            2.0 * np.pi *
            np.arange(num_rays, dtype=np.float64) /
            num_rays
        )

        dx = np.cos(angles)
        dy = np.sin(angles)

        # ---------------------------------------------------------
        # Sample distances along every ray
        # ---------------------------------------------------------

        steps = np.arange(
            1,
            max_steps + 1,
            dtype=np.float64
        )

        # Candidate cell center
        x0 = c + 0.5
        y0 = r + 0.5

        # Shape:
        #   (num_rays, max_steps)
        gx = x0 + dx[:, None] * steps[None, :]
        gy = y0 + dy[:, None] * steps[None, :]

        # Convert grid coordinates to map indices
        cc = np.floor(gx).astype(np.int32)
        rr = np.floor(gy).astype(np.int32)

        # ---------------------------------------------------------
        # Bounds
        # ---------------------------------------------------------

        in_bounds = (
            (rr >= 0) &
            (rr < rows) &
            (cc >= 0) &
            (cc < cols)
        )

        # Safe indices for NumPy indexing
        rr_safe = np.clip(rr, 0, rows - 1)
        cc_safe = np.clip(cc, 0, cols - 1)

        values = map_array[rr_safe, cc_safe]

        # ---------------------------------------------------------
        # Cell classification
        # ---------------------------------------------------------

        occupied = (
            (values >= self.occupied_threshold) &
            in_bounds
        )

        unknown = (
            (values == -1) &
            in_bounds
        )

        out_of_bounds = ~in_bounds

        # Anything below stops the ray
        stop_mask = (
            occupied |
            unknown |
            out_of_bounds
        )

        # ---------------------------------------------------------
        # Find first stopping cell for every ray
        # ---------------------------------------------------------

        has_stop = stop_mask.any(axis=1)

        first_stop = np.argmax(
            stop_mask,
            axis=1
        )

        # ---------------------------------------------------------
        # Determine which rays stopped on UNKNOWN
        # ---------------------------------------------------------

        valid_rays = np.where(has_stop)[0]

        if valid_rays.size == 0:
            return 0

        stop_indices = first_stop[valid_rays]

        stopped_unknown = unknown[
            valid_rays,
            stop_indices
        ]

        unknown_rays = valid_rays[stopped_unknown]

        if unknown_rays.size == 0:
            return 0

        unknown_steps = first_stop[unknown_rays]

        unknown_rows = rr_safe[
            unknown_rays,
            unknown_steps
        ]

        unknown_cols = cc_safe[
            unknown_rays,
            unknown_steps
        ]

        # ---------------------------------------------------------
        # Distinct unknown cells
        # ---------------------------------------------------------

        visible_unknown = np.unique(
            np.column_stack(
                (unknown_rows, unknown_cols)
            ),
            axis=0
        )

        return int(len(visible_unknown))

    # ================================================================
    # MAP COVERAGE
    # ================================================================

    def calculate_map_coverage(self, map_array):
        known = np.count_nonzero(map_array != -1)
        unknown = np.count_nonzero(map_array == -1)

        total = known + unknown
        if total == 0:
            return 0.0

        return 100.0 * known / total

    # ================================================================
    # FRONTIER UTILITY
    #
    # ONLY: U = alpha * IG_norm - beta_eff * TravelCost_norm
    # ================================================================

    def compute_utility(self, info_gain_norm, travel_cost_norm):
        battery_urgency = 1.0 - (self.battery_pct / 100.0)
        effective_beta = self.beta * (1.0 + battery_urgency)

        utility = self.alpha * info_gain_norm - effective_beta * travel_cost_norm

        self.get_logger().info(
            f"Battery={self.battery_pct:.1f}% | "
            f"Effective_beta={effective_beta:.2f} | "
            f"IG_norm={info_gain_norm:.3f} | "
            f"Cost_norm={travel_cost_norm:.3f} | "
            f"Utility={utility:.3f}"
        )

        return utility

    # ================================================================
    # PATH LENGTH
    # ================================================================

    def get_path_length_async(self, robot_x, robot_y, goal_x, goal_y, callback):
        if not self.compute_path_client.wait_for_server(timeout_sec=0.5):
            self.get_logger().warning("compute_path_to_pose server not available")
            callback(None)
            return

        start = PoseStamped()
        start.header.frame_id = self.map_frame
        start.header.stamp = self.get_clock().now().to_msg()
        start.pose.position.x = robot_x
        start.pose.position.y = robot_y
        start.pose.orientation.w = 1.0

        goal = PoseStamped()
        goal.header.frame_id = self.map_frame
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = goal_x
        goal.pose.position.y = goal_y
        goal.pose.orientation.w = 1.0

        req = ComputePathToPose.Goal()
        req.start = start
        req.goal = goal
        req.use_start = True

        self._path_token += 1
        token = self._path_token

        self._pending_path_requests[token] = {
            "token": token,
            "start_time": time.time(),
            "goal_handle": None,
            "callback": callback,
            "goal_x": goal_x,
            "goal_y": goal_y,
        }

        def finish(length):
            if token not in self._pending_path_requests:
                return
            del self._pending_path_requests[token]
            callback(length)

        def on_result(future):
            result = future.result()
            if result is None:
                finish(None)
                return

            path = result.result.path
            if not path.poses or len(path.poses) < 2:
                finish(None)
                return

            length = 0.0
            for i in range(1, len(path.poses)):
                p0 = path.poses[i - 1].pose.position
                p1 = path.poses[i].pose.position
                length += math.hypot(p1.x - p0.x, p1.y - p0.y)
            finish(length)

        def on_goal_response(future):
            goal_handle = future.result()
            if goal_handle is None or not goal_handle.accepted:
                finish(None)
                return
            if token in self._pending_path_requests:
                self._pending_path_requests[token]["goal_handle"] = goal_handle
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(on_result)

        send_goal_future = self.compute_path_client.send_goal_async(req)
        send_goal_future.add_done_callback(on_goal_response)

    # ================================================================
    # PATH WATCHDOG
    # ================================================================

    def _check_path_plan_timeout(self):
        now = time.time()
        expired_tokens = []

        for token, req_entry in list(self._pending_path_requests.items()):
            elapsed = now - req_entry["start_time"]
            if elapsed >= self.path_plan_timeout_sec:
                expired_tokens.append((token, req_entry))

        for token, req_entry in expired_tokens:
            if token in self._pending_path_requests:
                del self._pending_path_requests[token]

            self.get_logger().warning(
                f"compute_path_to_pose timed out for candidate "
                f"({req_entry['goal_x']:.2f}, {req_entry['goal_y']:.2f})")

            if req_entry["goal_handle"] is not None:
                req_entry["goal_handle"].cancel_goal_async()

            req_entry["callback"](None)

    # ================================================================
    # FRONTIER SELECTION
    # ================================================================

    def choose_frontier(self, clusters, map_array, robot_x, robot_y):
        robot_r, robot_c = self.world_to_grid(robot_x, robot_y)

        # Compute obstacle safety once for the entire map/cycle.
        unsafe_mask = self.compute_unsafe_mask(map_array)

        # Remove already explored / blacklisted frontier regions BEFORE
        # applying the top-15 cap. This prevents a large but unavailable
        # cluster from consuming one of the 15 evaluation slots.
        available_clusters = []

        for cluster in clusters:
            r, c = cluster["goal_cell"]
            x, y = self.grid_to_world(r, c)

            if self.is_already_visited(x, y):
                continue

            if self.is_blacklisted(x, y):
                continue

            available_clusters.append(cluster)

        # Only evaluate the largest available clusters in full detail.
        max_clusters_to_evaluate = 15
        clusters = sorted(
            available_clusters,
            key=lambda c: c["size"],
            reverse=True
        )[:max_clusters_to_evaluate]

        ig_start = time.time()
        candidates = []

        # Separate timing for goal-cell selection and vectorized IG.
        pick_goal_time_ms = 0.0
        ig_only_time_ms = 0.0

        # Debug counters ONLY — no behavior change
        rejected_no_goal = 0
        rejected_too_close = 0
        rejected_visited = 0
        rejected_blacklisted = 0

        for i, cluster in enumerate(clusters):

            pick_goal_start = time.time()

            goal_cell = self.pick_goal_cell(
                cluster,
                robot_r,
                robot_c,
                unsafe_mask
            )

            pick_goal_time_ms += (
                time.time() - pick_goal_start
            ) * 1000.0

            if goal_cell is None:
                rejected_no_goal += 1

                self.get_logger().warning(
                    f"[FRONTIER DEBUG] Cluster {i} REJECTED: "
                    f"pick_goal_cell() returned None "
                    f"| cluster_size={cluster['size']} "
                    f"| original_goal_cell={cluster['goal_cell']}"
                )
                continue

            r, c = goal_cell

            x, y = self.grid_to_world(r, c)

            distance_from_robot = math.hypot(
                x - robot_x,
                y - robot_y
            )

            if distance_from_robot < self.min_frontier_goal_distance_m:
                rejected_too_close += 1

                self.get_logger().warning(
                    f"[FRONTIER DEBUG] Cluster {i} REJECTED: "
                    f"too close "
                    f"| distance={distance_from_robot:.2f}m "
                    f"| minimum={self.min_frontier_goal_distance_m:.2f}m"
                )
                continue

            ig_only_start = time.time()

            info_gain = self.estimate_information_gain(
                map_array,
                r,
                c
            )

            ig_only_time_ms += (
                time.time() - ig_only_start
            ) * 1000.0

            already_visited = self.is_already_visited(x, y)
            blacklisted = self.is_blacklisted(x, y)

            if already_visited:
                rejected_visited += 1

                self.get_logger().warning(
                    f"[FRONTIER DEBUG] Cluster {i} REJECTED: "
                    f"goal already visited "
                    f"| goal=({x:.2f}, {y:.2f})"
                )
                continue

            if blacklisted:
                rejected_blacklisted += 1

                self.get_logger().warning(
                    f"[FRONTIER DEBUG] Cluster {i} REJECTED: "
                    f"goal is blacklisted "
                    f"| goal=({x:.2f}, {y:.2f})"
                )
                continue

            # Candidate survives all current filters.
            candidates.append({
                "x": x,
                "y": y,
                "size": cluster["size"],
                "info_gain": info_gain
            })

        ig_elapsed = time.time() - ig_start
        ig_elapsed_ms = ig_elapsed * 1000.0

        self.info_gain_timing_sum_ms += ig_elapsed_ms

        self.get_logger().info(
            f"[TIMING] info-gain eval for {len(clusters)} clusters: "
            f"{ig_elapsed_ms:.1f} ms"
        )

        self.get_logger().info(
            f"[TIMING] pick_goal_cell total: "
            f"{pick_goal_time_ms:.1f} ms | "
            f"estimate_information_gain total: "
            f"{ig_only_time_ms:.1f} ms"
        )

        # ------------------------------------------------------------
        # DEBUG SUMMARY
        # ------------------------------------------------------------

        self.get_logger().info(
            f"[FRONTIER DEBUG] FILTER SUMMARY: "
            f"clusters={len(clusters)} | "
            f"accepted={len(candidates)} | "
            f"no_goal={rejected_no_goal} | "
            f"too_close={rejected_too_close} | "
            f"visited={rejected_visited} | "
            f"blacklisted={rejected_blacklisted}"
        )

        # Keep existing behavior unchanged.
        MAX_PATH_CANDIDATES = 8

        candidates.sort(
            key=lambda c: c["info_gain"],
            reverse=True
        )

        candidates = candidates[:MAX_PATH_CANDIDATES]

        self.get_logger().info(
            f"[FRONTIER DEBUG] Candidates sent to Nav2 path evaluation: "
            f"{len(candidates)}"
        )

        if not candidates:
            self.publish_markers([], None)
            self.get_logger().warning(
                "No valid frontier found this cycle"
            )
            return

        self.is_evaluating = True

        self.evaluate_candidates_batch(
            candidates,
            robot_x,
            robot_y
        )

    def evaluate_candidates_batch(self, candidates, robot_x, robot_y):
        batch_start = time.time()
        path_results = []
        completed_count = 0
        total_candidates = len(candidates)

        def on_candidate_path_length(candidate, length):
            nonlocal completed_count
            path_results.append((candidate, length))
            completed_count += 1

            if completed_count == total_candidates:
                batch_elapsed = time.time() - batch_start
                batch_elapsed_ms = batch_elapsed * 1000.0
                self.async_path_timing_sum_ms += batch_elapsed_ms

                # Approximate end-to-end decision latency for this cycle:
                # synchronous explore() work + asynchronous Nav2 path-planning batch.
                decision_latency_ms = self.last_explore_sync_ms + batch_elapsed_ms
                self.decision_latency_sum_ms += decision_latency_ms
                self.decision_timing_samples += 1

                self.get_logger().info(
                    f"[TIMING] async path-planning batch ({total_candidates} candidates): "
                    f"{batch_elapsed_ms:.1f} ms"
                )
                self.get_logger().info(
                    f"[TIMING] decision latency approx: "
                    f"{decision_latency_ms:.1f} ms "
                    f"(sync={self.last_explore_sync_ms:.1f} ms + "
                    f"async={batch_elapsed_ms:.1f} ms)"
                )

                self.process_batch_evaluations(path_results)

        for candidate in candidates:
            self.get_path_length_async(
                robot_x, robot_y, candidate["x"], candidate["y"],
                lambda length, c=candidate: on_candidate_path_length(c, length)
            )

    def process_batch_evaluations(self, path_results):
        valid_candidates = [
            (c, length) for c, length in path_results
            if length is not None and length > 0.0
        ]

        if not valid_candidates:
            self.is_evaluating = False
            self.publish_markers([], None)
            self.get_logger().warning(
                "No reachable frontiers found - triggering recovery")
            self.trigger_recovery_spin()
            return

        max_travel_cost = max(length for c, length in valid_candidates)
        cost_normalizer = max(max_travel_cost, 1e-5)

        # Maximum possible visible-IG is bounded by the number of LiDAR
        # rays, so the normalization has a physical/sensor meaning.
        denom_ig = max(int(self.info_gain_num_rays), 1)

        self._eval_results = []

        for candidate, travel_cost in valid_candidates:
            info_gain_norm = min(candidate["info_gain"] / denom_ig, 1.0)
            travel_cost_norm = min(travel_cost / cost_normalizer, 1.0)

            score = self.compute_utility(info_gain_norm, travel_cost_norm)

            self._eval_results.append(
                (candidate["x"], candidate["y"], score, travel_cost, info_gain_norm, travel_cost_norm)
            )

        self.log_frontier_decisions()
        self.finish_evaluation()

    # ================================================================
    # FRONTIER LOGGING
    # ================================================================

    def log_frontier_decisions(self):
        if not self._eval_results:
            return

        os.makedirs(os.path.dirname(self.decision_results_path), exist_ok=True)
        file_exists = os.path.isfile(self.decision_results_path)

        best_score = max(result[2] for result in self._eval_results)

        with open(self.decision_results_path, mode='a', newline='') as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow([
                    'experiment_name', 'time_sec', 'frontier_x', 'frontier_y',
                    'info_gain_norm', 'travel_cost_norm', 'utility', 'chosen'
                ])

            for (x, y, score, length, info_gain_norm, travel_cost_norm) in self._eval_results:
                chosen = int(abs(score - best_score) < 1e-9)

                writer.writerow([
                    self.experiment_name,
                    round(time.time() - self.experiment_start_time, 2),
                    round(x, 3), round(y, 3),
                    round(info_gain_norm, 4), round(travel_cost_norm, 4),
                    round(score, 4), chosen
                ])

                self.frontier_decisions += 1
                self.utility_sum += score
                self.info_gain_sum += info_gain_norm
                self.travel_cost_sum += travel_cost_norm
                self.utility_samples += 1

    # ================================================================
    # FINISH FRONTIER EVALUATION
    # ================================================================

    def finish_evaluation(self):
        self.is_evaluating = False

        best = None
        best_length = None
        best_score = float('-inf')

        for (x, y, score, length, info_gain_norm, travel_cost_norm) in self._eval_results:
            if score > best_score:
                best_score = score
                best = (x, y)
                best_length = length

        markers_data = [
            (x, y, score) for (x, y, score, length, ign, tcn) in self._eval_results
        ]
        self.publish_markers(markers_data, best)

        if best is None:
            self.get_logger().warning("No valid frontier chosen")
            return

        goal_x, goal_y = best
        self.navigate_to(goal_x, goal_y, best_length, goal_type='frontier')

    # ================================================================
    # VISUALIZATION
    # ================================================================

    def publish_markers(self, candidates, chosen):
        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        for i, (x, y, score) in enumerate(candidates):
            m = Marker()
            m.header.frame_id = self.map_frame
            m.header.stamp = stamp
            m.ns = 'frontier_candidates'
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position = Point(x=x, y=y, z=0.1)
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.15
            is_chosen = chosen is not None and (x, y) == chosen
            m.color.r = 0.1 if is_chosen else 1.0
            m.color.g = 1.0 if is_chosen else 0.4
            m.color.b = 0.1
            m.color.a = 1.0
            m.lifetime = Duration(seconds=self.explore_period_sec + 1.0).to_msg()
            marker_array.markers.append(m)

        self.marker_pub.publish(marker_array)

    # ================================================================
    # RECOVERY
    # ================================================================

    def _recovery_sector_clearance(self, scan, sector_index):
        """Return a robust clearance estimate for one LiDAR sector."""
        n = max(4, int(self.recovery_scan_sectors))
        sector_width = 2.0 * math.pi / n
        ranges = []

        for i, raw_range in enumerate(scan.ranges):
            if not math.isfinite(raw_range):
                continue
            if raw_range < self.recovery_sector_min_range_m:
                continue

            r = min(float(raw_range), self.recovery_sector_max_range_m)

            angle = scan.angle_min + i * scan.angle_increment
            angle = math.atan2(math.sin(angle), math.cos(angle))

            sector = int((angle + math.pi) / sector_width)
            sector = max(0, min(n - 1, sector))

            if sector == sector_index:
                ranges.append(r)

        if not ranges:
            return self.recovery_sector_max_range_m

        ranges.sort()

        # 25th percentile prevents one long/invalid return from making a
        # mostly blocked sector appear safe.
        idx = max(0, int(0.25 * (len(ranges) - 1)))
        return ranges[idx]

    def choose_recovery_direction(self):
        """
        Select the clearest direction using the latest 360-degree LiDAR.

        Clearance is the primary criterion. When clearances are similar,
        the smaller rotation is preferred. If no LiDAR scan is available or
        every sector is too close to an obstacle, fall back to 180 degrees.
        """
        scan = self.latest_recovery_scan

        if scan is None or not scan.ranges:
            self.get_logger().warning(
                "No LiDAR scan available for recovery; "
                "using 180-degree fallback."
            )
            return math.pi

        n = max(4, int(self.recovery_scan_sectors))
        sector_width = 2.0 * math.pi / n
        candidates = []

        for sector in range(n):
            center_angle = -math.pi + (sector + 0.5) * sector_width
            clearance = self._recovery_sector_clearance(scan, sector)

            if clearance < self.recovery_clearance_margin_m:
                continue

            # Clearance dominates. A small turn penalty only breaks ties.
            turn_penalty = 0.05 * (abs(center_angle) / math.pi)
            score = clearance - turn_penalty

            candidates.append(
                (score, clearance, abs(center_angle), center_angle)
            )

        if not candidates:
            self.get_logger().warning(
                "LiDAR found no sufficiently clear recovery direction; "
                "using 180-degree fallback."
            )
            return math.pi

        _, clearance, _, best_angle = max(
            candidates,
            key=lambda item: item[0]
        )

        self.get_logger().info(
            f"LiDAR recovery selected {math.degrees(best_angle):.0f} deg "
            f"| clearance={clearance:.2f} m"
        )

        return math.atan2(
            math.sin(best_angle),
            math.cos(best_angle)
        )

    def _finish_recovery(self):
        self.recovery_in_progress = False
        self.goal_cancelled_for_recovery = False
        self.current_goal_handle = None
        self.is_navigating = False
        self.is_direct_targeting = False

        self.get_logger().info(
            "Recovery complete - frontier/target selection will be recomputed."
        )

    def clear_costmaps_for_recovery(self):
        """Clear Nav2 global/local costmaps before recovery rotation."""
        clients = [
            (
                self.global_costmap_clear_client,
                "global costmap",
            ),
            (
                self.local_costmap_clear_client,
                "local costmap",
            ),
        ]

        for client, name in clients:
            if not client.wait_for_service(timeout_sec=0.5):
                self.get_logger().warning(
                    f"{name} clear service not available; continuing recovery.")
                continue

            try:
                client.call_async(Empty.Request())
                self.get_logger().info(f"Requested {name} clear.")
            except Exception as exc:
                self.get_logger().warning(
                    f"Could not request {name} clear: {exc}")

    def trigger_recovery_spin(self):
        if self.recovery_in_progress:
            return

        self.recovery_in_progress = True

        # Clear stale/misaligned Nav2 costmaps before rotating so the
        # recovery spin is not immediately blocked by corrupted obstacle data.
        self.clear_costmaps_for_recovery()

        if not self.spin_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warning(
                "Spin action server not available; ending recovery."
            )
            self._finish_recovery()
            return

        target_yaw = self.choose_recovery_direction()

        goal = Spin.Goal()
        goal.target_yaw = target_yaw

        self.get_logger().info(
            f"Navigation recovery: turning {math.degrees(target_yaw):.0f} deg"
        )

        future = self.spin_client.send_goal_async(goal)

        def on_goal_response(response_future):
            try:
                goal_handle = response_future.result()

                if goal_handle is None or not goal_handle.accepted:
                    self.get_logger().warning("Recovery spin goal rejected.")
                    self._finish_recovery()
                    return

                result_future = goal_handle.get_result_async()

                def on_result(result_future_inner):
                    try:
                        status = result_future_inner.result().status
                        self.get_logger().info(
                            f"Recovery spin finished with status {status}"
                        )
                    except Exception as exc:
                        self.get_logger().warning(
                            f"Recovery spin result error: {exc}"
                        )
                    finally:
                        self._finish_recovery()

                result_future.add_done_callback(on_result)

            except Exception as exc:
                self.get_logger().warning(
                    f"Recovery spin error: {exc}"
                )
                self._finish_recovery()

        future.add_done_callback(on_goal_response)


    # ================================================================
    # NAVIGATION
    # ================================================================

    def navigate_to(self, x, y, path_length, goal_type='frontier', target_id=None):
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = self.map_frame
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.position.x = x
        goal_msg.pose.position.y = y
        goal_msg.pose.orientation.w = 1.0

        nav_goal = NavigateToPose.Goal()
        nav_goal.pose = goal_msg

        if goal_type == 'target':
            self.get_logger().info(f"NAVIGATING DIRECTLY TO TARGET #{target_id}: x={x:.2f}, y={y:.2f}")
        else:
            self.get_logger().info(
                f"NAVIGATING TO FRONTIER: x={x:.2f}, y={y:.2f} (battery={self.battery_pct:.1f}%)")

        if not self.nav_to_pose_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warning(
                "navigate_to_pose server not available"
            )
            self.is_direct_targeting = False
            self.trigger_recovery_spin()
            return

        self.is_navigating = True
        send_goal_future = self.nav_to_pose_client.send_goal_async(nav_goal)
        send_goal_future.add_done_callback(
            lambda future: self.goal_response_callback(future, x, y, path_length, goal_type, target_id))

    def _reset_navigation_progress_watchdog(self):
        self.nav_goal_started_time = time.time()
        pose = self.get_robot_pose()
        self.last_progress_pose = pose
        self.last_progress_time = time.time()

    def _check_navigation_progress(self):
        if not self.is_navigating:
            return

        if self.recovery_in_progress:
            return

        if self.nav_goal_started_time is None:
            return

        now = time.time()

        # Give Nav2 a short grace period to start moving.
        if now - self.nav_goal_started_time < self.progress_start_grace_sec:
            return

        pose = self.get_robot_pose()
        if pose is None:
            return

        if self.last_progress_pose is None:
            self.last_progress_pose = pose
            self.last_progress_time = now
            return

        displacement = math.hypot(
            pose[0] - self.last_progress_pose[0],
            pose[1] - self.last_progress_pose[1]
        )

        if displacement >= self.progress_min_distance_m:
            self.last_progress_pose = pose
            self.last_progress_time = now
            return

        if now - self.last_progress_time >= self.stuck_timeout_sec:
            self._handle_stuck_navigation()

    def _handle_stuck_navigation(self):
        if self.recovery_in_progress:
            return

        self.get_logger().warning(
            f"ROVER STUCK: no {self.progress_min_distance_m:.3f} m "
            f"progress for {self.stuck_timeout_sec:.1f}s"
        )

        self.goal_cancelled_for_recovery = True

        # Treat a stuck goal exactly like a failed navigation attempt.
        # This prevents the same bad frontier/target from being selected
        # immediately after recovery.
        if self.current_goal_type == 'frontier' and self.current_goal_x is not None:
            self.blacklist_point(self.current_goal_x, self.current_goal_y)
        elif self.current_goal_type == 'target' and self.current_target_id is not None:
            self._register_target_attempt_failure(self.current_target_id)

        if self.current_goal_handle is not None:
            try:
                cancel_future = self.current_goal_handle.cancel_goal_async()

                def after_cancel(_future):
                    self.trigger_recovery_spin()

                cancel_future.add_done_callback(after_cancel)
            except Exception as exc:
                self.get_logger().warning(
                    f"Could not cancel stuck Nav2 goal: {exc}"
                )
                self.trigger_recovery_spin()
        else:
            self.trigger_recovery_spin()

        self.last_progress_time = time.time()

    def goal_response_callback(self, future, x, y, path_length, goal_type, target_id):
        goal_handle = future.result()

        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warning("Goal rejected!")

            if goal_type == 'frontier':
                self.blacklist_point(x, y)
            elif goal_type == 'target':
                self._register_target_attempt_failure(target_id)

            self.is_navigating = False
            self.is_direct_targeting = False
            return

        self.get_logger().info("Goal accepted")
        self.current_goal_handle = goal_handle
        self.current_goal_x = x
        self.current_goal_y = y
        self.current_goal_type = goal_type
        self.current_target_id = target_id
        self._reset_navigation_progress_watchdog()

        if goal_type == 'frontier':
            self.visited_frontiers_world.append((x, y))

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda future: self.navigation_complete_callback(future, x, y, path_length, goal_type, target_id))

    # ================================================================
    # TARGET STATE
    # ================================================================

    def _target_is_available(self, t):
        """
        A target is a valid candidate if: it's the class we're
        searching for, hasn't been inspected, hasn't been abandoned
        (too many failed approach attempts), and clears the noise
        floor. Confidence plays no role beyond this floor check - it
        does NOT influence which available target gets picked.
        """
        return (
            t.class_name == self.target_class
            and t.id not in self.inspected_target_ids
            and t.id not in self.abandoned_target_ids
            and t.confidence >= self.min_target_confidence
        )

    def has_unvisited_known_targets(self):
        return any(self._target_is_available(t) for t in self.semantic_targets)

    def _register_target_attempt_failure(self, target_id):
        self.target_attempt_counts[target_id] = self.target_attempt_counts.get(target_id, 0) + 1
        attempts = self.target_attempt_counts[target_id]

        self.get_logger().warning(
            f"Target #{target_id}: failed approach attempt {attempts}/{self.max_investigate_attempts}")

        if attempts >= self.max_investigate_attempts:
            self.abandoned_target_ids.add(target_id)
            self.targets_abandoned_count = len(self.abandoned_target_ids)
            self.get_logger().warning(
                f"Target #{target_id}: ABANDONED after {attempts} failed attempts - "
                f"excluded from further consideration, does not count as inspected")

    # ================================================================
    # DIRECT TARGET NAVIGATION
    # ================================================================

    def navigate_direct_to_target(self, robot_x, robot_y):
        """
        Evaluate every known, uninspected, non-abandoned target above
        the noise floor using the actual Nav2 planned path length,
        then navigate to the reachable target with the shortest path.

        T* = argmin_i D_Nav2(i)

        Euclidean distance is NOT used for target selection.
        Confidence is NOT used for target selection (only as the
        entry-level noise floor already applied in
        _target_is_available).
        """
        candidates = [t for t in self.semantic_targets if self._target_is_available(t)]

        if not candidates:
            return

        # Prevent the main timer from starting another target-selection
        # cycle while asynchronous Nav2 path requests are running.
        self.is_direct_targeting = True

        self.get_logger().info(
            f"Evaluating {len(candidates)} uninspected target(s) using Nav2 path length...")

        results = []
        completed = 0

        def on_target_path(target, path_length):
            nonlocal completed
            completed += 1

            if path_length is not None and path_length > 0.0:
                results.append((target, path_length))
                self.get_logger().info(f"Target #{target.id}: Nav2 path = {path_length:.2f} m")
            else:
                self.get_logger().warning(f"Target #{target.id}: no reachable Nav2 path")

            if completed != len(candidates):
                return

        if not results:
            self.is_direct_targeting = False

            # Register a failed investigation attempt for every target
            # whose path evaluation failed, so unreachable targets cannot
            # trap the mission in an endless retry loop.
            for target_id in available_targets:
                self._register_target_attempt_failure(target_id)

            self.get_logger().warning(
                "No reachable uninspected targets found. "
                "Registered failed attempt(s) for unreachable target(s)."
            )

            return

        target = next(
            (
                t for t in self.semantic_targets
                if (
                    t.id == attempted_target_id
                    and t.class_name == self.target_class
                    and t.id not in self.inspected_target_ids
                )
            ),
            None
        )

        if target is None:
            self.get_logger().warning(
                f"Selected target #{attempted_target_id} "
                "is no longer available in the semantic database."
            )
            return False

        distance = math.hypot(
            goal_x - target.x,
            goal_y - target.y
        )

        if distance > self.visited_radius_m:
            self.get_logger().warning(
                f"Selected target #{target.id} not reached: "
                f"distance={distance:.2f} m > "
                f"visited_radius={self.visited_radius_m:.2f} m"
            )
            return False

        msg = PointStamped()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.point.x = target.x
        msg.point.y = target.y
        msg.point.z = 0.0
        self.mark_visited_pub.publish(msg)

        self.targets_found_count += 1
        self.inspected_target_ids.add(target.id)

        inspection_time = time.time() - self.experiment_start_time
        count = len(self.inspected_target_ids)
        self.targets_inspected_count = count

        if count == 1 and self.first_target_inspection_time_sec is None:
            self.first_target_inspection_time_sec = inspection_time

        self.inspection_milestone_times[count] = inspection_time
        self.save_inspection_milestone(count, inspection_time)

        self.get_logger().info("==================================================")
        self.get_logger().info(
            f"TARGET INSPECTED: {target.class_name} #{target.id}"
        )
        self.get_logger().info(
            f"Distance to selected target = {distance:.2f} m"
        )
        self.get_logger().info(f"Inspected count = {count}")
        self.get_logger().info("Decision: RESUME EXPLORATION")
        self.get_logger().info("==================================================")

        return True

    # ================================================================
    # NAVIGATION COMPLETE
    # ================================================================

    def navigation_complete_callback(self, future, x, y, path_length, goal_type, target_id):
        try:
            result = future.result()
            status = result.status

            if status == 4:
                self.get_logger().info(f"Reached goal ({x:.2f}, {y:.2f})")

                if path_length and path_length > 0:
                    self.total_distance_traveled += path_length
                    self.battery_pct = max(
                        0.0, self.battery_pct - path_length * self.battery_drain_pct_per_meter)

                if goal_type == 'target':
                    resolved = self.mark_target_visited_if_reached(x, y, attempted_target_id=target_id)
                    if not resolved:
                        # Nav2 reported success reaching the goal pose,
                        # but nothing was actually within visited_radius_m
                        # (e.g. stale/bad localization, or a false
                        # positive with nothing physically there).
                        self._register_target_attempt_failure(target_id)

            else:
                self.get_logger().warning(f"Navigation failed with status {status}")

                if goal_type == 'frontier':
                    self.blacklist_point(x, y)
                elif goal_type == 'target':
                    self._register_target_attempt_failure(target_id)

                # If the watchdog already cancelled this goal because the
                # rover was stuck, do not launch a second recovery spin.
                if not self.goal_cancelled_for_recovery:
                    self.trigger_recovery_spin()

        except Exception as e:
            self.get_logger().error(f"Navigation result error: {e}")

        finally:
            self.is_navigating = False
            self.is_direct_targeting = False
            self.nav_goal_started_time = None
            self.last_progress_pose = None
            self.last_progress_time = None

    # ================================================================
    # MAIN EXPLORATION / SEARCH LOOP
    # ================================================================

    def explore(self):
        if self.exploration_done:
            return

        if self.map_data is None:
            self.get_logger().info("Waiting for map...")
            return

        if (
            self.is_navigating
            or self.is_evaluating
            or self.is_direct_targeting
            or self.recovery_in_progress
        ):
            return

        robot_pose = self.get_robot_pose()
        if robot_pose is None:
            return

        robot_x, robot_y = robot_pose

        cycle_start = time.time()

        width = self.map_data.info.width
        height = self.map_data.info.height
        map_array = np.array(self.map_data.data, dtype=np.int8).reshape((height, width))

        self.final_coverage_pct = self.calculate_map_coverage(map_array)

        # ============================================================
        # DECISION 1: KNOWN, AVAILABLE TARGET?
        #
        # If a target is already detected, not inspected, not
        # abandoned, and above the noise floor, investigation takes
        # priority over frontier exploration.
        # ============================================================

        if self.has_unvisited_known_targets():
            self.empty_cycle_count = 0
            self.navigate_direct_to_target(robot_x, robot_y)
            return

        # ============================================================
        # DECISION 2: NO AVAILABLE TARGET -> EXPLORE
        # ============================================================

        frontier_start = time.time()
        frontier_cells = self.find_frontier_cells(map_array)
        clusters = self.cluster_frontiers(frontier_cells)
        frontier_elapsed = time.time() - frontier_start
        frontier_elapsed_ms = frontier_elapsed * 1000.0

        self.get_logger().info(
            f"[TIMING] frontier detection+clustering: {frontier_elapsed_ms:.1f} ms "
            f"| cells={len(frontier_cells)} clusters={len(clusters)}"
        )

        # ============================================================
        # MISSION COMPLETION CHECK
        #
        # clusters == [] is not the only way exploration can finish.
        # Frontier geometry can remain even when every detected
        # frontier is already visited or blacklisted.
        #
        # This check uses ALL detected clusters and is intentionally
        # separate from the 15 -> IG -> 8 -> Nav2 evaluation pipeline.
        # ============================================================

        if not clusters:
            self.empty_cycle_count += 1

            self.get_logger().info(
                f"[MISSION DEBUG] No frontier clusters found "
                f"({self.empty_cycle_count}/{self.empty_cycles_before_done})"
            )

        else:
            has_new_frontier = self.has_unvisited_frontier(clusters)

            if has_new_frontier:
                self.empty_cycle_count = 0

                self.get_logger().info(
                    f"[MISSION DEBUG] {len(clusters)} frontier clusters found "
                    f"| at least one unvisited frontier remains "
                    f"| completion counter reset"
                )

            else:
                self.empty_cycle_count += 1

                self.get_logger().info(
                    f"[MISSION DEBUG] Frontier clusters exist ({len(clusters)}) "
                    f"but no unvisited/available frontier remains "
                    f"({self.empty_cycle_count}/{self.empty_cycles_before_done})"
                )

        # ============================================================
        # MISSION COMPLETE AFTER CONSECUTIVE EMPTY EXPLORATION CYCLES
        # ============================================================

        if self.empty_cycle_count >= self.empty_cycles_before_done:

            self.get_logger().info("==================================================")
            self.get_logger().info("MISSION COMPLETE")
            self.get_logger().info(
                "No unvisited frontier remains and no known, "
                "available uninspected targets remain."
            )

            if self.abandoned_target_ids:
                self.get_logger().info(
                    f"{len(self.abandoned_target_ids)} target(s) were abandoned "
                    f"(never resolved after "
                    f"{self.max_investigate_attempts} attempts)."
                )

            self.get_logger().info("==================================================")

            self.exploration_done = True
            self.save_summary_results()
            return

        # ============================================================
        # FRONTIER EXPLORATION
        # ============================================================

        self.get_logger().info("Decision: EXPLORE")
        self.choose_frontier(clusters, map_array, robot_x, robot_y)

        self.frontier_timing_sum_ms += frontier_elapsed_ms
        self.timing_samples += 1

        # NOTE: This timing measures only the SYNCHRONOUS portion of explore() —
        # frontier detection, clustering, and info-gain evaluation. It does NOT
        # include the asynchronous Nav2 ComputePathToPose evaluation, since that
        # runs later through callbacks (on_candidate_path_length ->
        # process_batch_evaluations) once the ROS2 executor processes the
        # action server responses. Total real-world decision latency = this
        # synchronous time + async path-planning round-trip time (see below).
        cycle_elapsed = time.time() - cycle_start
        self.last_explore_sync_ms = cycle_elapsed * 1000.0
        self.get_logger().info(
            f"[TIMING] explore() total sync portion: {self.last_explore_sync_ms:.1f} ms"
        )

    # ================================================================
    # RESULT FILE INITIALIZATION
    # ================================================================

    def _initialize_result_files(self):
        """Create all per-run CSV files and their headers immediately."""

        os.makedirs(self.run_dir, exist_ok=True)

        # Frontier decisions
        if not os.path.isfile(self.decision_results_path):
            with open(self.decision_results_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'experiment_name', 'time_sec', 'frontier_x', 'frontier_y',
                    'info_gain_norm', 'travel_cost_norm', 'utility', 'chosen'
                ])

        # Experiment summary
        if not os.path.isfile(self.summary_results_path):
            with open(self.summary_results_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'experiment_name', 'target_class',
                    'min_target_confidence', 'max_investigate_attempts',
                    'alpha', 'beta',
                    'duration_sec', 'distance_traveled_m', 'final_coverage_pct',
                    'frontier_decisions',
                    'targets_detected_count', 'first_target_detection_time_sec',
                    'targets_inspected_count', 'targets_abandoned_count',
                    'first_target_inspection_time_sec', 'all_targets_inspection_time_sec',
                    'avg_utility', 'avg_info_gain_norm', 'avg_travel_cost_norm',
                    'avg_frontier_detect_ms', 'avg_info_gain_eval_ms',
                    'avg_async_path_planning_ms', 'avg_decision_latency_ms'
                ])

        # Target milestones
        self._ensure_milestone_file()

    # ================================================================
    # SUMMARY RESULTS
    # ================================================================

    def save_summary_results(self):
        if self.results_saved:
            return

        os.makedirs(os.path.dirname(self.summary_results_path), exist_ok=True)
        file_exists = os.path.isfile(self.summary_results_path)

        duration = round(time.time() - self.experiment_start_time, 2)

        # If every DETECTED target was either inspected or abandoned,
        # mission completion (w.r.t. targets) is the time of the
        # final inspection milestone.
        resolved_count = len(self.inspected_target_ids) + len(self.abandoned_target_ids)
        if (
            self.targets_detected_count > 0
            and resolved_count >= self.targets_detected_count
            and self.inspection_milestone_times
        ):
            final_inspection_count = max(self.inspection_milestone_times.keys())
            self.all_targets_inspection_time_sec = self.inspection_milestone_times[final_inspection_count]

        avg_utility = self.utility_sum / self.utility_samples if self.utility_samples > 0 else 0.0
        avg_info_gain = self.info_gain_sum / self.utility_samples if self.utility_samples > 0 else 0.0
        avg_travel_cost = self.travel_cost_sum / self.utility_samples if self.utility_samples > 0 else 0.0
        avg_frontier_detect_ms = (
            self.frontier_timing_sum_ms / self.timing_samples
            if self.timing_samples > 0 else 0.0
        )
        avg_info_gain_eval_ms = (
            self.info_gain_timing_sum_ms / self.timing_samples
            if self.timing_samples > 0 else 0.0
        )
        avg_async_path_planning_ms = (
            self.async_path_timing_sum_ms / self.decision_timing_samples
            if self.decision_timing_samples > 0 else 0.0
        )
        avg_decision_latency_ms = (
            self.decision_latency_sum_ms / self.decision_timing_samples
            if self.decision_timing_samples > 0 else 0.0
        )

        with open(self.summary_results_path, mode='a', newline='') as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow([
                    'experiment_name', 'target_class',
                    'min_target_confidence', 'max_investigate_attempts',
                    'alpha', 'beta',
                    'duration_sec', 'distance_traveled_m', 'final_coverage_pct',
                    'frontier_decisions',
                    'targets_detected_count', 'first_target_detection_time_sec',
                    'targets_inspected_count', 'targets_abandoned_count',
                    'first_target_inspection_time_sec', 'all_targets_inspection_time_sec',
                    'avg_utility', 'avg_info_gain_norm', 'avg_travel_cost_norm',
                    'avg_frontier_detect_ms', 'avg_info_gain_eval_ms',
                    'avg_async_path_planning_ms', 'avg_decision_latency_ms'
                ])

            writer.writerow([
                self.experiment_name, self.target_class,
                self.min_target_confidence, self.max_investigate_attempts,
                self.alpha, self.beta,
                duration, round(self.total_distance_traveled, 2), round(self.final_coverage_pct, 2),
                self.frontier_decisions,
                self.targets_detected_count,
                (round(self.first_target_detection_time_sec, 2)
                 if self.first_target_detection_time_sec is not None else ''),
                self.targets_inspected_count, self.targets_abandoned_count,
                (round(self.first_target_inspection_time_sec, 2)
                 if self.first_target_inspection_time_sec is not None else ''),
                (round(self.all_targets_inspection_time_sec, 2)
                 if self.all_targets_inspection_time_sec is not None else ''),
                round(avg_utility, 4), round(avg_info_gain, 4), round(avg_travel_cost, 4),
                round(avg_frontier_detect_ms, 3), round(avg_info_gain_eval_ms, 3),
                round(avg_async_path_planning_ms, 3), round(avg_decision_latency_ms, 3)
            ])

        self.save_target_milestones()
        self.results_saved = True
        self.get_logger().info(f"Summary metrics saved to {self.summary_results_path}")

    # ================================================================
    # TARGET MILESTONES
    # ================================================================

    def _ensure_milestone_file(self):
        os.makedirs(os.path.dirname(self.milestone_results_path), exist_ok=True)

        if not os.path.isfile(self.milestone_results_path):
            with open(self.milestone_results_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'run_number',
                    'experiment_name',
                    'target_class',
                    'milestone_type',
                    'target_count',
                    'time_sec'
                ])

    def _append_milestone(self, milestone_type, count, timestamp):
        self._ensure_milestone_file()

        with open(self.milestone_results_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.run_number,
                self.experiment_name,
                self.target_class,
                milestone_type,
                count,
                round(timestamp, 2)
            ])

        self.get_logger().info(
            f"Milestone saved immediately | {milestone_type} #{count} | "
            f"{timestamp:.2f}s | {self.milestone_results_path}"
        )

    def save_detection_milestone(self, count, timestamp):
        """Save a detection immediately; do not wait for mission completion."""
        if count in self.saved_detection_milestones:
            return

        self._append_milestone('detection', count, timestamp)
        self.saved_detection_milestones.add(count)

    def save_inspection_milestone(self, count, timestamp):
        """Save an inspection immediately; do not wait for mission completion."""
        if count in self.saved_inspection_milestones:
            return

        self._append_milestone('inspection', count, timestamp)
        self.saved_inspection_milestones.add(count)

    def save_target_milestones(self):
        """
        Final reconciliation at mission completion.

        KEEP this call. It ensures every in-memory milestone is present,
        while avoiding duplicate rows for milestones already saved live.
        """
        self._ensure_milestone_file()

        for count, timestamp in sorted(self.detection_milestone_times.items()):
            if count not in self.saved_detection_milestones:
                self._append_milestone('detection', count, timestamp)
                self.saved_detection_milestones.add(count)

        for count, timestamp in sorted(self.inspection_milestone_times.items()):
            if count not in self.saved_inspection_milestones:
                self._append_milestone('inspection', count, timestamp)
                self.saved_inspection_milestones.add(count)


# ====================================================================
# MAIN
# ====================================================================

def main(args=None):
    rclpy.init(args=args)
    node = ExplorerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Finalize whatever data has been collected, even when the
        # experiment is stopped manually with Ctrl+C.
        node.save_summary_results()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
