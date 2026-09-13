import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Point
from nav2_msgs.action import NavigateToPose, ComputePathToPose, Spin
from visualization_msgs.msg import Marker, MarkerArray

import tf2_ros
from tf2_ros import TransformException


class ExplorerNode(Node):
    def __init__(self):
        super().__init__('explorer')

        # ------------------------------------------------------------
        # ROS2 parameters - tunable at runtime via `ros2 param set`
        # ------------------------------------------------------------
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('explore_period_sec', 5.0)
        self.declare_parameter('min_cluster_size', 4)
        self.declare_parameter('visited_radius_m', 0.6)
        self.declare_parameter('obstacle_margin_cells', 3)
        self.declare_parameter('occupied_threshold', 65)
        self.declare_parameter('blacklist_timeout_sec', 30.0)
        self.declare_parameter('empty_cycles_before_done', 3)
        self.declare_parameter('path_plan_timeout_sec', 3.0)
        self.declare_parameter('path_plan_watchdog_period_sec', 0.5)

        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.explore_period_sec = self.get_parameter('explore_period_sec').value
        self.min_cluster_size = self.get_parameter('min_cluster_size').value
        self.visited_radius_m = self.get_parameter('visited_radius_m').value
        self.obstacle_margin_cells = self.get_parameter('obstacle_margin_cells').value
        self.occupied_threshold = self.get_parameter('occupied_threshold').value
        self.blacklist_timeout_sec = self.get_parameter('blacklist_timeout_sec').value
        self.empty_cycles_before_done = self.get_parameter('empty_cycles_before_done').value
        self.path_plan_timeout_sec = self.get_parameter('path_plan_timeout_sec').value
        self.path_plan_watchdog_period_sec = self.get_parameter(
            'path_plan_watchdog_period_sec').value

        self.get_logger().info("Explorer Node Started")

        # ------------------------------------------------------------
        # State
        # ------------------------------------------------------------
        self.map_data = None
        self.is_navigating = False
        self.visited_frontiers_world = []       # list of (x, y) already sent as goals
        self.blacklist = {}                     # (x, y) -> expiry timestamp
        self.current_goal_handle = None
        self.empty_cycle_count = 0
        self.exploration_done = False

        # Async candidate-evaluation state (see choose_frontier / evaluate_next_candidate)
        self.is_evaluating = False
        self._eval_queue = []
        self._eval_results = []
        self._eval_robot_pose = None

        # compute_path_to_pose timeout bookkeeping. A single long-lived
        # watchdog timer (created below) checks this against wall-clock
        # time rather than spinning up/tearing down a timer per
        # candidate - only one path request is ever in flight at once
        # since evaluation is sequential, so one slot is enough.
        self._path_token = 0
        self._path_pending = None  # dict: token/start_time/goal_handle/callback/goal_x/goal_y

        # ------------------------------------------------------------
        # TF listener - real robot pose, same source nav2 uses
        # ------------------------------------------------------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ------------------------------------------------------------
        # Subscriptions / action clients / publishers
        # ------------------------------------------------------------
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)

        self.nav_to_pose_client = ActionClient(
            self, NavigateToPose, 'navigate_to_pose')

        self.compute_path_client = ActionClient(
            self, ComputePathToPose, 'compute_path_to_pose')

        self.spin_client = ActionClient(self, Spin, 'spin')

        self.marker_pub = self.create_publisher(
            MarkerArray, '/explorer/frontier_markers', 10)

        self.timer = self.create_timer(
            self.explore_period_sec, self.explore)

        self.path_plan_watchdog_timer = self.create_timer(
            self.path_plan_watchdog_period_sec, self._check_path_plan_timeout)

    # ------------------------------------------------------------
    # Map callback
    # ------------------------------------------------------------
    def map_callback(self, msg):
        self.map_data = msg

    # ------------------------------------------------------------
    # Pose lookup via TF (same transform nav2 uses internally)
    # ------------------------------------------------------------
    def get_robot_pose(self):
        try:
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, now,
                timeout=Duration(seconds=0.5))
            return trans.transform.translation.x, trans.transform.translation.y
        except TransformException as ex:
            self.get_logger().warning(
                f"Could not get robot pose ({self.map_frame} -> "
                f"{self.base_frame}): {ex}")
            return None

    # ------------------------------------------------------------
    # Frontier detection (raw candidate cells)
    # ------------------------------------------------------------
    def find_frontier_cells(self, map_array):
        frontier_cells = []
        rows, cols = map_array.shape

        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                if map_array[r, c] != 0:
                    continue  # only known-free cells
                neighbors = map_array[r - 1:r + 2, c - 1:c + 2]
                if np.any(neighbors == -1):
                    frontier_cells.append((r, c))

        return frontier_cells

    # ------------------------------------------------------------
    # Cluster adjacent frontier cells (8-connected flood fill).
    # Flood fill is a perfectly adequate clustering method here;
    # DBSCAN etc. would be marginal gain for meaningfully more code.
    # ------------------------------------------------------------
    def cluster_frontiers(self, frontier_cells):
        if not frontier_cells:
            return []

        cell_set = set(frontier_cells)
        visited = set()
        clusters = []  # each: {"centroid": (r, c), "size": n, "cells": [(r, c), ...]}

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
                # Compute centroid of the cluster
                mean_r = sum(p[0] for p in cluster) / len(cluster)
                mean_c = sum(p[1] for p in cluster) / len(cluster)

                # Choose the actual frontier cell closest to the centroid (medoid)
                medoid = min(cluster,
                             key=lambda p: (p[0] - mean_r) ** 2 + (p[1] - mean_c) ** 2
            )

                clusters.append({
                    "goal_cell": medoid,
                    "size": len(cluster),
                    "cells": cluster,
                })

        return clusters

    # ------------------------------------------------------------
    # Obstacle safety margin check
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # Visited / blacklist checks (world-coordinate based, since the
    # grid resizes/re-origins as slam_toolbox grows the map)
    # ------------------------------------------------------------
    def is_already_visited(self, x, y):
        for (vx, vy) in self.visited_frontiers_world:
            if math.hypot(x - vx, y - vy) < self.visited_radius_m:
                return True
        return False

    def is_blacklisted(self, x, y):
        now = time.time()
        expired = [k for k, exp in self.blacklist.items() if exp < now]
        for k in expired:
            del self.blacklist[k]

        for (bx, by) in self.blacklist:
            if math.hypot(x - bx, y - by) < self.visited_radius_m:
                return True
        return False

    def blacklist_point(self, x, y):
        self.blacklist[(x, y)] = time.time() + self.blacklist_timeout_sec
        self.get_logger().warning(
            f"Blacklisting frontier ({x:.2f}, {y:.2f}) for "
            f"{self.blacklist_timeout_sec:.0f}s")

    # ------------------------------------------------------------
    # Pick the actual goal point within a cluster: the frontier cell
    # closest to the robot that also clears the obstacle margin,
    # rather than the raw centroid. Centroids of irregular (e.g.
    # L-shaped, or wrapped-around-an-obstacle) clusters can land in
    # occupied or unsafe space even when the cluster itself is fine.
    # ------------------------------------------------------------
    def pick_goal_cell(self, cluster, map_array, robot_r, robot_c):

        r, c = cluster["goal_cell"]

        if self.is_cell_safe(map_array, r, c):
            return (r, c)

        best = None
        best_dist = float("inf")

        for rr, cc in cluster["cells"]:
            if not self.is_cell_safe(map_array, rr, cc):
                continue

            d = (rr - robot_r) ** 2 + (cc - robot_c) ** 2

            if d < best_dist:
                best_dist = d
                best = (rr, cc)

        return best
    # ------------------------------------------------------------
    # Async path-length lookup via Nav2's planner. Fully callback
    # driven (mirrors navigate_to / goal_response_callback below) so
    # it never blocks the node's own executor with
    # spin_until_future_complete - calling that from inside a timer
    # callback that's running on the same executor thread that would
    # need to process the response is an rclpy anti-pattern and can
    # stall the node. Also serves as the reachability check: if
    # planning fails or times out, the frontier is treated as
    # unreachable this cycle (not permanently blacklisted, since the
    # map may open a route later).
    #
    # Timeout handling is delegated to the single long-lived
    # `path_plan_watchdog_timer` (see _check_path_plan_timeout) rather
    # than a fresh timer per call - candidates are evaluated one at a
    # time, so a single pending-request slot plus a monotonically
    # increasing token (to ignore stale completions) is all that's
    # needed, and it avoids create/destroy timer churn every cycle.
    # ------------------------------------------------------------
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
        self._path_pending = {
            "token": token,
            "start_time": time.time(),
            "goal_handle": None,
            "callback": callback,
            "goal_x": goal_x,
            "goal_y": goal_y,
        }

        def finish(length):
            # Ignore a completion that arrives after the watchdog already
            # timed this token out (and cleared _path_pending) or after a
            # newer request has since taken the pending slot.
            if self._path_pending is None or self._path_pending["token"] != token:
                return
            self._path_pending = None
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
            if self._path_pending is not None and self._path_pending["token"] == token:
                self._path_pending["goal_handle"] = goal_handle
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(on_result)

        send_goal_future = self.compute_path_client.send_goal_async(req)
        send_goal_future.add_done_callback(on_goal_response)

    def _check_path_plan_timeout(self):
        """Watchdog tick (runs continuously for the node's lifetime).
        No-op unless a compute_path_to_pose request is currently
        pending and has exceeded path_plan_timeout_sec."""
        pending = self._path_pending
        if pending is None:
            return

        elapsed = time.time() - pending["start_time"]
        if elapsed < self.path_plan_timeout_sec:
            return

        self._path_pending = None  # clear first: guards re-entrancy from the callback below
        self.get_logger().warning(
            f"compute_path_to_pose timed out for candidate "
            f"({pending['goal_x']:.2f}, {pending['goal_y']:.2f})")
        if pending["goal_handle"] is not None:
            pending["goal_handle"].cancel_goal_async()
        pending["callback"](None)

    # ------------------------------------------------------------
    # Build the candidate list (goal point per cluster, filtered by
    # safety / visited / blacklist) and kick off async scoring. Does
    # NOT return a value - navigate_to() (or the no-candidate log) is
    # invoked later from finish_evaluation() once every candidate has
    # been scored.
    # ------------------------------------------------------------
    def choose_frontier(self, clusters, map_array, robot_x, robot_y):
        robot_r, robot_c = self.world_to_grid(robot_x, robot_y)

        candidates = []
        for cluster in clusters:
            goal_cell = self.pick_goal_cell(cluster, map_array, robot_r, robot_c)
            if goal_cell is None:
                continue  # no cell in this cluster clears the obstacle margin

            r, c = goal_cell
            x, y = self.grid_to_world(r, c)

            if self.is_already_visited(x, y) or self.is_blacklisted(x, y):
                continue

            candidates.append({"x": x, "y": y, "size": cluster["size"]})

        if not candidates:
            self.publish_markers([], None)
            self.get_logger().warning(
                "No valid (safe / unvisited) frontier found this cycle")
            return

        self.is_evaluating = True
        self._eval_queue = candidates
        self._eval_results = []
        self._eval_robot_pose = (robot_x, robot_y)
        self.evaluate_next_candidate()

    def evaluate_next_candidate(self):
        if not self._eval_queue:
            self.finish_evaluation()
            return

        candidate = self._eval_queue.pop(0)
        robot_x, robot_y = self._eval_robot_pose

        def on_length(length):
            if length is not None and length > 0.0:
                score = candidate["size"] / length
                self._eval_results.append((candidate["x"], candidate["y"], score))
            # else: unreachable per the planner this cycle - skip silently
            self.evaluate_next_candidate()

        self.get_path_length_async(
            robot_x, robot_y, candidate["x"], candidate["y"], on_length)

    def finish_evaluation(self):
        self.is_evaluating = False

        best = None
        best_score = -1.0
        for (x, y, score) in self._eval_results:
            if score > best_score:
                best_score = score
                best = (x, y)

        self.publish_markers(self._eval_results, best)

        if best is None:
            self.get_logger().warning(
                "No valid (safe / reachable / unvisited) frontier found "
                "this cycle")
            return

        goal_x, goal_y = best
        self.navigate_to(goal_x, goal_y)

    # ------------------------------------------------------------
    # RViz visualization: all candidate frontiers + chosen goal
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # Recovery: rotate in place, used when navigation fails, to
    # get a fresh scan sweep before picking another frontier.
    # ------------------------------------------------------------
    def trigger_recovery_spin(self):
        if not self.spin_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warning("spin action server not available, skipping recovery")
            return

        goal = Spin.Goal()
        goal.target_yaw = math.pi  # half turn
        self.get_logger().info("Navigation failed - running recovery spin")
        self.spin_client.send_goal_async(goal)

    # ------------------------------------------------------------
    # Send a NavigateToPose goal (same action RViz's Nav2 Goal uses)
    # ------------------------------------------------------------
    def navigate_to(self, x, y):
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = self.map_frame
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.position.x = x
        goal_msg.pose.position.y = y
        goal_msg.pose.orientation.w = 1.0

        nav_goal = NavigateToPose.Goal()
        nav_goal.pose = goal_msg

        self.get_logger().info(f"Navigating to frontier: x={x:.2f}, y={y:.2f}")

        if not self.nav_to_pose_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warning("navigate_to_pose action server not available")
            return

        self.is_navigating = True
        send_goal_future = self.nav_to_pose_client.send_goal_async(nav_goal)
        send_goal_future.add_done_callback(
            lambda future: self.goal_response_callback(future, x, y))

    def goal_response_callback(self, future, x, y):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().warning("Goal rejected!")
            self.blacklist_point(x, y)
            self.is_navigating = False
            return

        self.get_logger().info("Goal accepted")
        self.current_goal_handle = goal_handle
        self.visited_frontiers_world.append((x, y))

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda future: self.navigation_complete_callback(future, x, y))

    def navigation_complete_callback(self, future, x, y):
        try:
            result = future.result()
            status = result.status
            # status 4 == SUCCEEDED in action_msgs/msg/GoalStatus
            if status != 4:
                self.get_logger().warning(
                    f"Navigation to ({x:.2f}, {y:.2f}) did not succeed "
                    f"(status={status})")
                self.blacklist_point(x, y)
                self.trigger_recovery_spin()
            else:
                self.get_logger().info("Navigation completed successfully")
        except Exception as e:
            self.get_logger().error(f"Navigation failed: {e}")
            self.blacklist_point(x, y)
        finally:
            self.is_navigating = False
            self.current_goal_handle = None

    # ------------------------------------------------------------
    # Main exploration loop
    # ------------------------------------------------------------
    def explore(self):
        if self.exploration_done:
            return

        if self.is_navigating or self.is_evaluating:
            return

        if self.map_data is None:
            self.get_logger().warning("No map data available yet")
            return

        pose = self.get_robot_pose()
        if pose is None:
            return
        robot_x, robot_y = pose

        map_array = np.array(self.map_data.data).reshape(
            (self.map_data.info.height, self.map_data.info.width))

        frontier_cells = self.find_frontier_cells(map_array)
        clusters = self.cluster_frontiers(frontier_cells) if frontier_cells else []

        if not clusters:
            self.empty_cycle_count += 1
            self.get_logger().info(
                f"No frontier clusters this cycle "
                f"({self.empty_cycle_count}/{self.empty_cycles_before_done})")
            if self.empty_cycle_count >= self.empty_cycles_before_done:
                self.get_logger().info(
                    "Exploration complete - no new frontiers across "
                    "consecutive cycles")
                self.exploration_done = True
            return

        self.empty_cycle_count = 0  # reset, we found something this cycle

        # Async: choose_frontier evaluates candidates via callbacks and
        # calls navigate_to() itself once done (see finish_evaluation).
        self.choose_frontier(clusters, map_array, robot_x, robot_y)


def main(args=None):
    rclpy.init(args=args)
    explorer_node = ExplorerNode()

    try:
        explorer_node.get_logger().info("Starting exploration...")
        rclpy.spin(explorer_node)
    except KeyboardInterrupt:
        explorer_node.get_logger().info("Exploration stopped by user")
    finally:
        explorer_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()