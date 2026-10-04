

# import math
# import time
# import csv
# from pathlib import Path
# from datetime import datetime

# import rclpy
# from rclpy.node import Node

# from geometry_msgs.msg import PoseArray, Pose, PointStamped

# from semantic_interfaces.msg import SemanticTarget, SemanticTargetArray


# class SemanticObject:
#     """One tracked object in the semantic database."""

#     _next_id = 1

#     def __init__(
#         self,
#         class_name,
#         x,
#         y,
#         confidence,
#         stamp_sec,
#         track_id=-1,
#         has_snapshot=False
#     ):
#         self.id = SemanticObject._next_id
#         SemanticObject._next_id += 1

#         self.class_name = class_name
#         self.track_id = track_id
#         self.has_snapshot = has_snapshot

#         self.x = x
#         self.y = y
#         self.confidence = confidence
#         self.last_seen = stamp_sec
#         self.num_observations = 1
#         self.visited = False

#     def update(
#         self,
#         x,
#         y,
#         confidence,
#         stamp_sec,
#         has_snapshot=False
#     ):
#         """
#         Merge a new observation of the SAME real-world object into this
#         entry. Position is updated with a simple running average
#         (weighted by observation count) rather than just overwriting -
#         this smooths out per-frame lidar/camera noise instead of letting
#         the latest single reading jump the stored position around.
#         """

#         # Once a snapshot exists for this object, never unset the flag.
#         if has_snapshot:
#             self.has_snapshot = True

#         n = self.num_observations

#         self.x = (self.x * n + x) / (n + 1)
#         self.y = (self.y * n + y) / (n + 1)

#         self.confidence = max(
#             self.confidence,
#             confidence
#         )

#         self.last_seen = stamp_sec

#         self.num_observations += 1


# class SemanticDatabaseNode(Node):
#     """
#     Persistent semantic object database.

#     Responsibility:
#       - receive individual detections on /semantic_targets
#       - decide whether each one is a new object or a repeat sighting
#         of something already known
#       - use ByteTrack ID as the primary identity when available
#       - fall back to class + distance deduplication when no track ID
#         match exists
#       - keep a running, deduplicated list of real-world objects
#       - publish that list
#       - accept "I reached this object" reports and mark it visited
#     """

#     def __init__(self):
#         super().__init__('semantic_database')

#         self.declare_parameter(
#             'duplicate_distance_m',
#             1.0
#         )

#         self.declare_parameter(
#             'max_track_jump_m',
#             1.0
#         )

#         self.declare_parameter(
#             'publish_period_sec',
#             1.0
#         )

#         self.declare_parameter(
#             'visited_radius_m',
#             0.5
#         )

#         self.declare_parameter(
#             'same_frame_duplicate_distance_m',
#             0.3
#         )

#         self.declare_parameter(
#             'min_confirmations',
#             2
#         )

#         self.declare_parameter(
#             'semantic_targets_topic',
#             '/semantic_targets'
#         )

#         self.declare_parameter(
#             'explorer_targets_topic',
#             '/explorer/semantic_targets'
#         )

#         self.declare_parameter(
#             'database_topic',
#             '/semantic_database'
#         )

#         self.declare_parameter(
#             'mark_visited_topic',
#             '/semantic_database/mark_visited'
#         )

#         self.declare_parameter(
#             'map_frame',
#             'map'
#         )

#         self.duplicate_distance_m = (
#             self.get_parameter(
#                 'duplicate_distance_m'
#             ).value
#         )

#         self.max_track_jump_m = (
#             self.get_parameter(
#                 'max_track_jump_m'
#             ).value
#         )

#         self.publish_period_sec = (
#             self.get_parameter(
#                 'publish_period_sec'
#             ).value
#         )

#         self.visited_radius_m = (
#             self.get_parameter(
#                 'visited_radius_m'
#             ).value
#         )

#         self.same_frame_duplicate_distance_m = (
#             self.get_parameter(
#                 'same_frame_duplicate_distance_m'
#             ).value
#         )

#         self.min_confirmations = (
#             self.get_parameter(
#                 'min_confirmations'
#             ).value
#         )

#         self.map_frame = (
#             self.get_parameter(
#                 'map_frame'
#             ).value
#         )

#         self.database = []

#         # ============================================================
#         # DEDUP DIAGNOSTIC LOG
#         #
#         # Store diagnostics under:
#         #
#         # ~/snapshots/run_YYYYMMDD_HHMMSS/
#         #
#         # instead of /tmp.
#         # ============================================================

#         snapshots_root = Path.home() / "snapshots"

#         run_name = (
#             "run_"
#             + datetime.now().strftime("%Y%m%d_%H%M%S")
#         )

#         self.diagnostic_dir = snapshots_root / run_name

#         self.diagnostic_dir.mkdir(
#             parents=True,
#             exist_ok=True
#         )

#         self.dedup_csv_path = (
#             self.diagnostic_dir
#             / "semantic_database_dedup_diagnostic.csv"
#         )

#         with open(
#             self.dedup_csv_path,
#             "w",
#             newline=""
#         ) as f:

#             writer = csv.writer(f)

#             writer.writerow([
#                 "timestamp_sec",
#                 "class_name",
#                 "incoming_x",
#                 "incoming_y",
#                 "confidence",
#                 "candidate_id",
#                 "candidate_x",
#                 "candidate_y",
#                 "distance_m",
#                 "excluded",
#                 "threshold_m",
#                 "decision"
#             ])

#         self.get_logger().info(
#             f"Semantic diagnostic CSV: "
#             f"{self.dedup_csv_path}"
#         )

#         # ------------------------------------------------------------
#         # Subscribers
#         # ------------------------------------------------------------

#         semantic_targets_topic = (
#             self.get_parameter(
#                 'semantic_targets_topic'
#             ).value
#         )

#         self.target_sub = self.create_subscription(
#             SemanticTargetArray,
#             semantic_targets_topic,
#             self.target_callback,
#             10
#         )

#         mark_visited_topic = (
#             self.get_parameter(
#                 'mark_visited_topic'
#             ).value
#         )

#         self.mark_visited_sub = self.create_subscription(
#             PointStamped,
#             mark_visited_topic,
#             self.mark_visited_callback,
#             10
#         )

#         # ------------------------------------------------------------
#         # Publishers
#         # ------------------------------------------------------------

#         explorer_targets_topic = (
#             self.get_parameter(
#                 'explorer_targets_topic'
#             ).value
#         )

#         self.explorer_pub = self.create_publisher(
#             PoseArray,
#             explorer_targets_topic,
#             10
#         )

#         database_topic = (
#             self.get_parameter(
#                 'database_topic'
#             ).value
#         )

#         self.database_pub = self.create_publisher(
#             SemanticTargetArray,
#             database_topic,
#             10
#         )

#         self.publish_timer = self.create_timer(
#             self.publish_period_sec,
#             self.publish_state
#         )

#         self.get_logger().info(
#             f"Semantic Database started | "
#             f"duplicate_distance_m="
#             f"{self.duplicate_distance_m} | "
#             f"ByteTrack-aware dedup enabled"
#         )

#     # ------------------------------------------------------------
#     # Incoming detections
#     # ------------------------------------------------------------

#     def target_callback(
#         self,
#         msg: SemanticTargetArray
#     ):

#         stamp_sec = (
#             msg.header.stamp.sec
#             + msg.header.stamp.nanosec * 1e-9
#         )

#         # IDs already matched/created THIS callback.
#         #
#         # Two same-class detections arriving in the SAME message
#         # are treated as distinct detections when they have different
#         # track IDs / cannot use the same database object.
#         used_ids_this_frame = set()

#         for target in msg.targets:

#             x = target.x
#             y = target.y

#             # ========================================================
#             # BYTE TRACK ID
#             # ========================================================

#             track_id = target.track_id

#             # ========================================================
#             # SNAPSHOT FLAG
#             # ========================================================

#             has_snapshot = target.has_snapshot

#             matched_id = self._add_or_update(
#                 target.class_name,
#                 x,
#                 y,
#                 target.confidence,
#                 stamp_sec,
#                 track_id,
#                 has_snapshot=has_snapshot,
#                 exclude_ids=used_ids_this_frame
#             )

#             used_ids_this_frame.add(
#                 matched_id
#             )

#     # ------------------------------------------------------------
#     # Add / update database object
#     # ------------------------------------------------------------

#     def _add_or_update(
#         self,
#         class_name,
#         x,
#         y,
#         confidence,
#         stamp_sec,
#         track_id=-1,
#         has_snapshot=False,
#         exclude_ids=None
#     ):

#         exclude_ids = exclude_ids or set()

#         best_match = None
#         best_dist = float('inf')

#         candidates_log = []

#         # ============================================================
#         # PASS 1:
#         # BYTE TRACK ID MATCH
#         #
#         # Same class + same valid ByteTrack ID means this is the
#         # same tracked detection, regardless of small position noise.
#         # ============================================================

#         if track_id != -1:

#             for obj in self.database:

#                 if obj.class_name != class_name:
#                     continue

#                 if obj.id in exclude_ids:
#                     continue

#                 if obj.track_id == track_id:

#                     dist = math.hypot(
#                         obj.x - x,
#                         obj.y - y
#                     )

#                     if dist > self.max_track_jump_m:
#                         # Track ID matches, but the position jumped too
#                         # far to trust — don't blindly merge. Fall
#                         # through to Pass 2's distance-based dedup
#                         # instead of trusting track continuity here.
#                         self.get_logger().warning(
#                             f"[TRACK MATCH REJECTED] "
#                             f"class={class_name} "
#                             f"track_id={track_id} "
#                             f"-> database #{obj.id} "
#                             f"dist={dist:.4f} exceeds "
#                             f"max_track_jump_m={self.max_track_jump_m:.2f}"
#                         )
#                         continue

#                     best_match = obj
#                     best_dist = dist

#                     self.get_logger().info(
#                         f"[TRACK MATCH] "
#                         f"class={class_name} "
#                         f"track_id={track_id} "
#                         f"-> database #{obj.id} "
#                         f"dist={best_dist:.4f}"
#                     )

#                     # =================================================
#                     # DIAGNOSTIC:
#                     # Also record ByteTrack-based matches in the CSV.
#                     # =================================================

#                     with open(
#                         self.dedup_csv_path,
#                         "a",
#                         newline=""
#                     ) as f:

#                         writer = csv.writer(f)

#                         writer.writerow([
#                             f"{stamp_sec:.6f}",
#                             class_name,
#                             f"{x:.4f}",
#                             f"{y:.4f}",
#                             f"{confidence:.4f}",
#                             obj.id,
#                             f"{obj.x:.4f}",
#                             f"{obj.y:.4f}",
#                             f"{best_dist:.4f}",
#                             False,
#                             f"{self.duplicate_distance_m:.4f}",
#                             f"TRACK_MATCH(track_id={track_id})"
#                         ])

#                     # Snapshot flag can only transition False -> True.
#                     if has_snapshot:
#                         obj.has_snapshot = True

#                     break

#         # ============================================================
#         # PASS 2:
#         # NORMAL DISTANCE DEDUP FALLBACK
#         #
#         # Only used if ByteTrack did not find a match.
#         # ============================================================

#         if best_match is None:

#             for obj in self.database:

#                 if obj.class_name != class_name:
#                     continue

#                 dist = math.hypot(
#                     obj.x - x,
#                     obj.y - y
#                 )

#                 excluded = (
#                     obj.id in exclude_ids
#                 )

#                 candidates_log.append(
#                     (
#                         obj.id,
#                         dist,
#                         excluded
#                     )
#                 )

#                 # ----------------------------------------------------
#                 # Diagnostic CSV
#                 # ----------------------------------------------------

#                 with open(
#                     self.dedup_csv_path,
#                     "a",
#                     newline=""
#                 ) as f:

#                     writer = csv.writer(f)

#                     writer.writerow([
#                         f"{stamp_sec:.6f}",
#                         class_name,
#                         f"{x:.4f}",
#                         f"{y:.4f}",
#                         f"{confidence:.4f}",
#                         obj.id,
#                         f"{obj.x:.4f}",
#                         f"{obj.y:.4f}",
#                         f"{dist:.4f}",
#                         excluded,
#                         f"{self.duplicate_distance_m:.4f}",
#                         (
#                             "EXCLUDED"
#                             if excluded
#                             else (
#                                 "WITHIN_THRESHOLD"
#                                 if dist <
#                                 self.duplicate_distance_m
#                                 else
#                                 "OUTSIDE_THRESHOLD"
#                             )
#                         )
#                     ])

#                 if excluded:
#                     continue

#                 # ----------------------------------------------------
#                 # Pick closest valid candidate.
#                 # ----------------------------------------------------

#                 if (
#                     dist < self.duplicate_distance_m
#                     and dist < best_dist
#                 ):

#                     best_match = obj
#                     best_dist = dist

#         # ============================================================
#         # SAME-FRAME DUPLICATE DISCARD
#         #
#         # If the closest excluded candidate is a REAL object already
#         # claimed by another detection this frame, AND it's within a
#         # tight same-object distance (not the full dedup radius), this
#         # is almost certainly a second detection of the SAME real
#         # object this frame - not a genuinely separate nearby object.
#         # Drop it instead of spawning a duplicate.
#         # ============================================================

#         same_frame_duplicate = None

#         for obj_id, dist, excluded in candidates_log:
#             if excluded and dist <= self.same_frame_duplicate_distance_m:
#                 if (
#                     same_frame_duplicate is None
#                     or dist < same_frame_duplicate[1]
#                 ):
#                     same_frame_duplicate = (obj_id, dist)

#         if best_match is None and same_frame_duplicate is not None:

#             discard_id, discard_dist = same_frame_duplicate

#             self.get_logger().info(
#                 f"[DEDUP DECISION] "
#                 f"incoming ({x:.3f},{y:.3f}) "
#                 f"class={class_name} "
#                 f"conf={confidence:.3f} "
#                 f"track_id={track_id} | "
#                 f"candidates={[(o, round(d,4), e) for o,d,e in candidates_log]} | "
#                 f"threshold={self.duplicate_distance_m:.3f} | "
#                 f"result=DISCARDED same-frame duplicate of #{discard_id} "
#                 f"dist={discard_dist:.4f}"
#             )

#             return discard_id

#         # ============================================================
#         # DECISION LOG
#         # ============================================================

#         if best_match is not None:

#             if (
#                 track_id != -1
#                 and best_match.track_id == track_id
#             ):

#                 decision_str = (
#                     f"TRACK MERGE into "
#                     f"#{best_match.id} "
#                     f"track_id={track_id} "
#                     f"dist={best_dist:.4f}"
#                 )

#             else:

#                 decision_str = (
#                     f"DISTANCE MERGE into "
#                     f"#{best_match.id} "
#                     f"dist={best_dist:.4f}"
#                 )

#         else:

#             decision_str = "NEW OBJECT"

#         candidates_summary = [
#             (oid, round(d, 4), ex)
#             for oid, d, ex in candidates_log
#         ]

#         self.get_logger().info(
#             f"[DEDUP DECISION] "
#             f"incoming ({x:.3f},{y:.3f}) "
#             f"class={class_name} "
#             f"conf={confidence:.3f} "
#             f"track_id={track_id} | "
#             f"candidates={candidates_summary} | "
#             f"threshold={self.duplicate_distance_m:.3f} | "
#             f"result={decision_str}"
#         )

#         # ============================================================
#         # UPDATE EXISTING OBJECT
#         # ============================================================

#         if best_match is not None:

#             best_match.update(
#                 x,
#                 y,
#                 confidence,
#                 stamp_sec,
#                 has_snapshot=has_snapshot
#             )

#             # Only claim/overwrite the tracked ID if the object didn't
#             # already own a DIFFERENT live track ID. A Pass-2 distance
#             # match finding a nearby object should not silently steal
#             # that object's existing track identity out from under it -
#             # doing so orphans the original track_id and causes it to
#             # spawn a phantom new object on its next sighting.
#             if track_id != -1:
#                 if best_match.track_id in (-1, track_id):
#                     best_match.track_id = track_id
#                 else:
#                     self.get_logger().warning(
#                         f"[TRACK OWNERSHIP GUARD] "
#                         f"incoming track_id={track_id} matched #{best_match.id} "
#                         f"via distance, but #{best_match.id} already owns "
#                         f"track_id={best_match.track_id} - NOT overwriting"
#                     )

#             return best_match.id

#         # ============================================================
#         # CREATE NEW OBJECT
#         # ============================================================

#         new_obj = SemanticObject(
#             class_name,
#             x,
#             y,
#             confidence,
#             stamp_sec,
#             track_id,
#             has_snapshot=has_snapshot
#         )

#         self.database.append(
#             new_obj
#         )

#         self.get_logger().info(
#             f"New object #{new_obj.id}: "
#             f"{class_name} "
#             f"track_id={track_id} "
#             f"snapshot={has_snapshot} "
#             f"at ({x:.2f}, {y:.2f}) "
#             f"conf={confidence:.2f}"
#         )

#         return new_obj.id

#     # ------------------------------------------------------------
#     # Visited marking
#     # ------------------------------------------------------------

#     def mark_visited_callback(
#         self,
#         msg: PointStamped
#     ):

#         x = msg.point.x
#         y = msg.point.y

#         nearest_obj = None
#         nearest_dist = float('inf')

#         for obj in self.database:

#             if obj.num_observations < self.min_confirmations:
#                 continue

#             if obj.visited:
#                 continue

#             dist = math.hypot(
#                 obj.x - x,
#                 obj.y - y
#             )

#             if dist < nearest_dist:

#                 nearest_dist = dist
#                 nearest_obj = obj

#         if (
#             nearest_obj is not None
#             and nearest_dist < self.visited_radius_m
#         ):

#             nearest_obj.visited = True

#             self.get_logger().info(
#                 f"Marked object "
#                 f"#{nearest_obj.id} "
#                 f"({nearest_obj.class_name}) "
#                 f"track_id={nearest_obj.track_id} "
#                 f"as visited "
#                 f"(reported point was "
#                 f"{nearest_dist:.2f}m away)"
#             )

#         else:

#             self.get_logger().warning(
#                 f"mark_visited: no unvisited "
#                 f"object within "
#                 f"{self.visited_radius_m}m "
#                 f"of ({x:.2f}, {y:.2f})"
#             )

#     # ------------------------------------------------------------
#     # Periodic publishing
#     # ------------------------------------------------------------

#     def publish_state(self):

#         # ============================================================
#         # FULL DATABASE
#         # ============================================================

#         full_msg = SemanticTargetArray()

#         full_msg.header.stamp = (
#             self.get_clock().now().to_msg()
#         )

#         full_msg.header.frame_id = self.map_frame

#         for obj in self.database:

#             if obj.num_observations < self.min_confirmations:
#                 continue

#             t = SemanticTarget()

#             t.id = obj.id

#             # ========================================================
#             # BYTE TRACK ID
#             # ========================================================

#             t.track_id = obj.track_id

#             t.class_name = obj.class_name

#             t.confidence = obj.confidence

#             # ========================================================
#             # SNAPSHOT FLAG
#             # ========================================================

#             t.has_snapshot = obj.has_snapshot

#             t.x = obj.x
#             t.y = obj.y
#             t.z = 0.0

#             full_msg.targets.append(
#                 t
#             )

#         self.database_pub.publish(
#             full_msg
#         )

#         # ============================================================
#         # UNVISITED OBJECTS -> EXPLORER
#         # ============================================================

#         pose_array = PoseArray()

#         pose_array.header.stamp = (
#             self.get_clock().now().to_msg()
#         )

#         pose_array.header.frame_id = (
#             self.map_frame
#         )

#         for obj in self.database:

#             if obj.num_observations < self.min_confirmations:
#                 continue

#             if obj.visited:
#                 continue

#             pose = Pose()

#             pose.position.x = obj.x
#             pose.position.y = obj.y
#             pose.position.z = 0.0

#             pose.orientation.w = 1.0

#             pose_array.poses.append(
#                 pose
#             )

#         self.explorer_pub.publish(
#             pose_array
#         )


# def main(args=None):

#     rclpy.init(args=args)

#     node = SemanticDatabaseNode()

#     try:

#         rclpy.spin(node)

#     except KeyboardInterrupt:
#         pass

#     finally:

#         node.destroy_node()

#         rclpy.shutdown()


# if __name__ == '__main__':
#     main()




import math
import time
import csv
from pathlib import Path
from datetime import datetime

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseArray, Pose, PointStamped

from semantic_interfaces.msg import SemanticTarget, SemanticTargetArray


class SemanticObject:
    """One tracked object in the semantic database."""

    _next_id = 1

    def __init__(
        self,
        class_name,
        x,
        y,
        confidence,
        stamp_sec,
        track_id=-1,
        has_snapshot=False,
        snapshot_path=""
    ):
        self.id = SemanticObject._next_id
        SemanticObject._next_id += 1

        self.class_name = class_name
        self.track_id = track_id
        self.has_snapshot = has_snapshot
        self.snapshot_path = snapshot_path

        self.x = x
        self.y = y
        self.confidence = confidence
        self.last_seen = stamp_sec
        self.num_observations = 1
        self.visited = False

    def update(
        self,
        x,
        y,
        confidence,
        stamp_sec,
        has_snapshot=False,
        snapshot_path=""
    ):
        """
        Merge a new observation of the SAME real-world object into this
        entry. Position is updated with a simple running average
        (weighted by observation count) rather than just overwriting -
        this smooths out per-frame lidar/camera noise instead of letting
        the latest single reading jump the stored position around.
        """

        # Once a snapshot exists for this object, never unset the flag.
        if has_snapshot:
            self.has_snapshot = True
        if snapshot_path and not self.snapshot_path:
            self.snapshot_path = snapshot_path

        n = self.num_observations

        self.x = (self.x * n + x) / (n + 1)
        self.y = (self.y * n + y) / (n + 1)

        self.confidence = max(
            self.confidence,
            confidence
        )

        self.last_seen = stamp_sec

        self.num_observations += 1


class SemanticDatabaseNode(Node):
    """
    Persistent semantic object database.

    Responsibility:
      - receive individual detections on /semantic_targets
      - decide whether each one is a new object or a repeat sighting
        of something already known
      - use ByteTrack ID as the primary identity when available
      - fall back to class + distance deduplication when no track ID
        match exists
      - keep a running, deduplicated list of real-world objects
      - publish that list
      - accept "I reached this object" reports and mark it visited
    """

    def __init__(self):
        super().__init__('semantic_database')

        self.declare_parameter(
            'duplicate_distance_m',
            1.0
        )

        self.declare_parameter(
            'max_track_jump_m',
            1.0
        )

        self.declare_parameter(
            'publish_period_sec',
            1.0
        )

        self.declare_parameter(
            'visited_radius_m',
            0.5
        )

        self.declare_parameter(
            'same_frame_duplicate_distance_m',
            0.3
        )

        self.declare_parameter(
            'min_confirmations',
            2
        )

        self.declare_parameter(
            'semantic_targets_topic',
            '/semantic_targets'
        )

        self.declare_parameter(
            'explorer_targets_topic',
            '/explorer/semantic_targets'
        )

        self.declare_parameter(
            'database_topic',
            '/semantic_database'
        )

        self.declare_parameter(
            'mark_visited_topic',
            '/semantic_database/mark_visited'
        )

        self.declare_parameter(
            'map_frame',
            'map'
        )

        self.duplicate_distance_m = (
            self.get_parameter(
                'duplicate_distance_m'
            ).value
        )

        self.max_track_jump_m = (
            self.get_parameter(
                'max_track_jump_m'
            ).value
        )

        self.publish_period_sec = (
            self.get_parameter(
                'publish_period_sec'
            ).value
        )

        self.visited_radius_m = (
            self.get_parameter(
                'visited_radius_m'
            ).value
        )

        self.same_frame_duplicate_distance_m = (
            self.get_parameter(
                'same_frame_duplicate_distance_m'
            ).value
        )

        self.min_confirmations = (
            self.get_parameter(
                'min_confirmations'
            ).value
        )

        self.map_frame = (
            self.get_parameter(
                'map_frame'
            ).value
        )

        self.database = []

        # ============================================================
        # DEDUP DIAGNOSTIC LOG
        #
        # Store diagnostics under:
        #
        # ~/snapshots/run_YYYYMMDD_HHMMSS/
        #
        # instead of /tmp.
        # ============================================================

        snapshots_root = Path.home() / "snapshots"

        run_name = (
            "run_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
        )

        self.diagnostic_dir = snapshots_root / run_name

        self.diagnostic_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.dedup_csv_path = (
            self.diagnostic_dir
            / "semantic_database_dedup_diagnostic.csv"
        )

        with open(
            self.dedup_csv_path,
            "w",
            newline=""
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                "timestamp_sec",
                "class_name",
                "incoming_x",
                "incoming_y",
                "confidence",
                "candidate_id",
                "candidate_x",
                "candidate_y",
                "distance_m",
                "excluded",
                "threshold_m",
                "decision"
            ])

        self.get_logger().info(
            f"Semantic diagnostic CSV: "
            f"{self.dedup_csv_path}"
        )

        # ------------------------------------------------------------
        # Subscribers
        # ------------------------------------------------------------

        semantic_targets_topic = (
            self.get_parameter(
                'semantic_targets_topic'
            ).value
        )

        self.target_sub = self.create_subscription(
            SemanticTargetArray,
            semantic_targets_topic,
            self.target_callback,
            10
        )

        mark_visited_topic = (
            self.get_parameter(
                'mark_visited_topic'
            ).value
        )

        self.mark_visited_sub = self.create_subscription(
            PointStamped,
            mark_visited_topic,
            self.mark_visited_callback,
            10
        )

        # ------------------------------------------------------------
        # Publishers
        # ------------------------------------------------------------

        explorer_targets_topic = (
            self.get_parameter(
                'explorer_targets_topic'
            ).value
        )

        self.explorer_pub = self.create_publisher(
            PoseArray,
            explorer_targets_topic,
            10
        )

        database_topic = (
            self.get_parameter(
                'database_topic'
            ).value
        )

        self.database_pub = self.create_publisher(
            SemanticTargetArray,
            database_topic,
            10
        )

        self.publish_timer = self.create_timer(
            self.publish_period_sec,
            self.publish_state
        )

        self.get_logger().info(
            f"Semantic Database started | "
            f"duplicate_distance_m="
            f"{self.duplicate_distance_m} | "
            f"ByteTrack-aware dedup enabled"
        )

    # ------------------------------------------------------------
    # Incoming detections
    # ------------------------------------------------------------

    def target_callback(
        self,
        msg: SemanticTargetArray
    ):

        stamp_sec = (
            msg.header.stamp.sec
            + msg.header.stamp.nanosec * 1e-9
        )

        # IDs already matched/created THIS callback.
        #
        # Two same-class detections arriving in the SAME message
        # are treated as distinct detections when they have different
        # track IDs / cannot use the same database object.
        used_ids_this_frame = set()

        for target in msg.targets:

            x = target.x
            y = target.y

            # ========================================================
            # BYTE TRACK ID
            # ========================================================

            track_id = target.track_id

            # ========================================================
            # SNAPSHOT FLAG
            # ========================================================

            has_snapshot = target.has_snapshot
            snapshot_path = target.snapshot_path

            matched_id = self._add_or_update(
                target.class_name,
                x,
                y,
                target.confidence,
                stamp_sec,
                track_id,
                has_snapshot=has_snapshot,
                snapshot_path=snapshot_path,
                exclude_ids=used_ids_this_frame
            )

            used_ids_this_frame.add(
                matched_id
            )

    # ------------------------------------------------------------
    # Add / update database object
    # ------------------------------------------------------------

    def _add_or_update(
        self,
        class_name,
        x,
        y,
        confidence,
        stamp_sec,
        track_id=-1,
        has_snapshot=False,
        snapshot_path="",
        exclude_ids=None
    ):

        exclude_ids = exclude_ids or set()

        best_match = None
        best_dist = float('inf')

        candidates_log = []

        # ============================================================
        # PASS 1:
        # BYTE TRACK ID MATCH
        #
        # Same class + same valid ByteTrack ID means this is the
        # same tracked detection, regardless of small position noise.
        # ============================================================

        if track_id != -1:

            for obj in self.database:

                if obj.class_name != class_name:
                    continue

                if obj.id in exclude_ids:
                    continue

                if obj.track_id == track_id:

                    dist = math.hypot(
                        obj.x - x,
                        obj.y - y
                    )

                    if dist > self.max_track_jump_m:
                        # Track ID matches, but the position jumped too
                        # far to trust — don't blindly merge. Fall
                        # through to Pass 2's distance-based dedup
                        # instead of trusting track continuity here.
                        self.get_logger().warning(
                            f"[TRACK MATCH REJECTED] "
                            f"class={class_name} "
                            f"track_id={track_id} "
                            f"-> database #{obj.id} "
                            f"dist={dist:.4f} exceeds "
                            f"max_track_jump_m={self.max_track_jump_m:.2f}"
                        )
                        continue

                    best_match = obj
                    best_dist = dist

                    self.get_logger().info(
                        f"[TRACK MATCH] "
                        f"class={class_name} "
                        f"track_id={track_id} "
                        f"-> database #{obj.id} "
                        f"dist={best_dist:.4f}"
                    )

                    # =================================================
                    # DIAGNOSTIC:
                    # Also record ByteTrack-based matches in the CSV.
                    # =================================================

                    with open(
                        self.dedup_csv_path,
                        "a",
                        newline=""
                    ) as f:

                        writer = csv.writer(f)

                        writer.writerow([
                            f"{stamp_sec:.6f}",
                            class_name,
                            f"{x:.4f}",
                            f"{y:.4f}",
                            f"{confidence:.4f}",
                            obj.id,
                            f"{obj.x:.4f}",
                            f"{obj.y:.4f}",
                            f"{best_dist:.4f}",
                            False,
                            f"{self.duplicate_distance_m:.4f}",
                            f"TRACK_MATCH(track_id={track_id})"
                        ])

                    # Snapshot flag can only transition False -> True.
                    if has_snapshot:
                        obj.has_snapshot = True
                    if snapshot_path and not obj.snapshot_path:
                        obj.snapshot_path = snapshot_path

                    break

        # ============================================================
        # PASS 2:
        # NORMAL DISTANCE DEDUP FALLBACK
        #
        # Only used if ByteTrack did not find a match.
        # ============================================================

        if best_match is None:

            for obj in self.database:

                if obj.class_name != class_name:
                    continue

                dist = math.hypot(
                    obj.x - x,
                    obj.y - y
                )

                excluded = (
                    obj.id in exclude_ids
                )

                candidates_log.append(
                    (
                        obj.id,
                        dist,
                        excluded
                    )
                )

                # ----------------------------------------------------
                # Diagnostic CSV
                # ----------------------------------------------------

                with open(
                    self.dedup_csv_path,
                    "a",
                    newline=""
                ) as f:

                    writer = csv.writer(f)

                    writer.writerow([
                        f"{stamp_sec:.6f}",
                        class_name,
                        f"{x:.4f}",
                        f"{y:.4f}",
                        f"{confidence:.4f}",
                        obj.id,
                        f"{obj.x:.4f}",
                        f"{obj.y:.4f}",
                        f"{dist:.4f}",
                        excluded,
                        f"{self.duplicate_distance_m:.4f}",
                        (
                            "EXCLUDED"
                            if excluded
                            else (
                                "WITHIN_THRESHOLD"
                                if dist <
                                self.duplicate_distance_m
                                else
                                "OUTSIDE_THRESHOLD"
                            )
                        )
                    ])

                if excluded:
                    continue

                # ----------------------------------------------------
                # Pick closest valid candidate.
                # ----------------------------------------------------

                if (
                    dist < self.duplicate_distance_m
                    and dist < best_dist
                ):

                    best_match = obj
                    best_dist = dist

        # ============================================================
        # SAME-FRAME DUPLICATE DISCARD
        #
        # If the closest excluded candidate is a REAL object already
        # claimed by another detection this frame, AND it's within a
        # tight same-object distance (not the full dedup radius), this
        # is almost certainly a second detection of the SAME real
        # object this frame - not a genuinely separate nearby object.
        # Drop it instead of spawning a duplicate.
        # ============================================================

        same_frame_duplicate = None

        for obj_id, dist, excluded in candidates_log:
            if excluded and dist <= self.same_frame_duplicate_distance_m:
                if (
                    same_frame_duplicate is None
                    or dist < same_frame_duplicate[1]
                ):
                    same_frame_duplicate = (obj_id, dist)

        if best_match is None and same_frame_duplicate is not None:

            discard_id, discard_dist = same_frame_duplicate

            self.get_logger().info(
                f"[DEDUP DECISION] "
                f"incoming ({x:.3f},{y:.3f}) "
                f"class={class_name} "
                f"conf={confidence:.3f} "
                f"track_id={track_id} | "
                f"candidates={[(o, round(d,4), e) for o,d,e in candidates_log]} | "
                f"threshold={self.duplicate_distance_m:.3f} | "
                f"result=DISCARDED same-frame duplicate of #{discard_id} "
                f"dist={discard_dist:.4f}"
            )

            return discard_id

        # ============================================================
        # DECISION LOG
        # ============================================================

        if best_match is not None:

            if (
                track_id != -1
                and best_match.track_id == track_id
            ):

                decision_str = (
                    f"TRACK MERGE into "
                    f"#{best_match.id} "
                    f"track_id={track_id} "
                    f"dist={best_dist:.4f}"
                )

            else:

                decision_str = (
                    f"DISTANCE MERGE into "
                    f"#{best_match.id} "
                    f"dist={best_dist:.4f}"
                )

        else:

            decision_str = "NEW OBJECT"

        candidates_summary = [
            (oid, round(d, 4), ex)
            for oid, d, ex in candidates_log
        ]

        self.get_logger().info(
            f"[DEDUP DECISION] "
            f"incoming ({x:.3f},{y:.3f}) "
            f"class={class_name} "
            f"conf={confidence:.3f} "
            f"track_id={track_id} | "
            f"candidates={candidates_summary} | "
            f"threshold={self.duplicate_distance_m:.3f} | "
            f"result={decision_str}"
        )

        # ============================================================
        # UPDATE EXISTING OBJECT
        # ============================================================

        if best_match is not None:

            best_match.update(
                x,
                y,
                confidence,
                stamp_sec,
                has_snapshot=has_snapshot,
                snapshot_path=snapshot_path
            )

            # Only claim/overwrite the tracked ID if the object didn't
            # already own a DIFFERENT live track ID. A Pass-2 distance
            # match finding a nearby object should not silently steal
            # that object's existing track identity out from under it -
            # doing so orphans the original track_id and causes it to
            # spawn a phantom new object on its next sighting.
            if track_id != -1:
                if best_match.track_id in (-1, track_id):
                    best_match.track_id = track_id
                else:
                    self.get_logger().warning(
                        f"[TRACK OWNERSHIP GUARD] "
                        f"incoming track_id={track_id} matched #{best_match.id} "
                        f"via distance, but #{best_match.id} already owns "
                        f"track_id={best_match.track_id} - NOT overwriting"
                    )

            return best_match.id

        # ============================================================
        # CREATE NEW OBJECT
        # ============================================================

        new_obj = SemanticObject(
            class_name,
            x,
            y,
            confidence,
            stamp_sec,
            track_id,
            has_snapshot=has_snapshot,
            snapshot_path=snapshot_path
        )

        self.database.append(
            new_obj
        )

        self.get_logger().info(
            f"New object #{new_obj.id}: "
            f"{class_name} "
            f"track_id={track_id} "
            f"snapshot={has_snapshot} "
            f"at ({x:.2f}, {y:.2f}) "
            f"conf={confidence:.2f}"
        )

        return new_obj.id

    # ------------------------------------------------------------
    # Visited marking
    # ------------------------------------------------------------

    def mark_visited_callback(
        self,
        msg: PointStamped
    ):

        x = msg.point.x
        y = msg.point.y

        nearest_obj = None
        nearest_dist = float('inf')

        for obj in self.database:

            if obj.num_observations < self.min_confirmations:
                continue

            if obj.visited:
                continue

            dist = math.hypot(
                obj.x - x,
                obj.y - y
            )

            if dist < nearest_dist:

                nearest_dist = dist
                nearest_obj = obj

        if (
            nearest_obj is not None
            and nearest_dist < self.visited_radius_m
        ):

            nearest_obj.visited = True

            self.get_logger().info(
                f"Marked object "
                f"#{nearest_obj.id} "
                f"({nearest_obj.class_name}) "
                f"track_id={nearest_obj.track_id} "
                f"as visited "
                f"(reported point was "
                f"{nearest_dist:.2f}m away)"
            )

        else:

            self.get_logger().warning(
                f"mark_visited: no unvisited "
                f"object within "
                f"{self.visited_radius_m}m "
                f"of ({x:.2f}, {y:.2f})"
            )

    # ------------------------------------------------------------
    # Periodic publishing
    # ------------------------------------------------------------

    def publish_state(self):

        # ============================================================
        # FULL DATABASE
        # ============================================================

        full_msg = SemanticTargetArray()

        full_msg.header.stamp = (
            self.get_clock().now().to_msg()
        )

        full_msg.header.frame_id = self.map_frame

        for obj in self.database:

            if obj.num_observations < self.min_confirmations:
                continue

            t = SemanticTarget()

            t.id = obj.id

            # ========================================================
            # BYTE TRACK ID
            # ========================================================

            t.track_id = obj.track_id

            t.class_name = obj.class_name

            t.confidence = obj.confidence

            # ========================================================
            # SNAPSHOT FLAG
            # ========================================================

            t.has_snapshot = obj.has_snapshot
            t.snapshot_path = obj.snapshot_path

            t.x = obj.x
            t.y = obj.y
            t.z = 0.0

            full_msg.targets.append(
                t
            )

        self.database_pub.publish(
            full_msg
        )

        # ============================================================
        # UNVISITED OBJECTS -> EXPLORER
        # ============================================================

        pose_array = PoseArray()

        pose_array.header.stamp = (
            self.get_clock().now().to_msg()
        )

        pose_array.header.frame_id = (
            self.map_frame
        )

        for obj in self.database:

            if obj.num_observations < self.min_confirmations:
                continue

            if obj.visited:
                continue

            pose = Pose()

            pose.position.x = obj.x
            pose.position.y = obj.y
            pose.position.z = 0.0

            pose.orientation.w = 1.0

            pose_array.poses.append(
                pose
            )

        self.explorer_pub.publish(
            pose_array
        )


def main(args=None):

    rclpy.init(args=args)

    node = SemanticDatabaseNode()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:

        node.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':
    main()


