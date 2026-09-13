import math
import csv
from pathlib import Path

import numpy as np
import cv2
import torch
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

import message_filters
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from ultralytics import YOLO

import tf2_ros
from tf2_ros import TransformException
from geometry_msgs.msg import PointStamped
import tf2_geometry_msgs  # noqa: F401  (registers PointStamped transform support)

from semantic_interfaces.msg import SemanticTarget, SemanticTargetArray


class YOLODetector(Node):

    def __init__(self):
        super().__init__("yolo_detector")

        # ============================================================
        # 1. ROS 2 PARAMETERS
        # ============================================================
        default_device = "cpu"

        self.declare_parameter(
            "model_path",
            "/home/yuvan/4wdtillnav2working/yolo26n_openvino_model/"
        )

        self.declare_parameter("conf_threshold", 0.5)
        self.declare_parameter("device", default_device)
        self.declare_parameter("sync_slop_sec", 0.15)
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("tf_timeout_sec", 0.3)
        self.declare_parameter("show_window", False)

        model_path = self.get_parameter("model_path").value
        self.conf_thresh = self.get_parameter("conf_threshold").value
        self.device = self.get_parameter("device").value
        self.sync_slop = self.get_parameter("sync_slop_sec").value
        self.map_frame = self.get_parameter("map_frame").value
        self.tf_timeout = self.get_parameter("tf_timeout_sec").value
        self.show_window = self.get_parameter("show_window").value

        # ============================================================
        # RAW YOLO DIAGNOSTIC CSV
        # ============================================================
        self.yolo_csv_path = Path("/tmp/yolo_raw_detections.csv")

        with open(self.yolo_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "frame",
                "timestamp_sec",
                "detection_count",
                "detection_index",
                "class_name",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
                "bbox_width",
                "bbox_height"
            ])

        self.yolo_frame_count = 0

        # ============================================================
        # SAME-FRAME IoU DIAGNOSTIC
        # ============================================================
        self.iou_csv_path = Path("/tmp/yolo_bbox_iou_diagnostic.csv")

        with open(self.iou_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "frame",
                "timestamp_sec",
                "class_name",
                "box_i",
                "box_j",
                "conf_i",
                "conf_j",
                "iou",
                "map_x_i",
                "map_y_i",
                "map_x_j",
                "map_y_j"
            ])

        self.get_logger().info(
            f"Raw YOLO diagnostic CSV: {self.yolo_csv_path}"
        )

        self.get_logger().info(
            f"YOLO bbox IoU diagnostic CSV: {self.iou_csv_path}"
        )

        # ============================================================
        # 2. CV BRIDGE + YOLO
        # ============================================================
        self.bridge = CvBridge()

        self.get_logger().info(
            f"Loading YOLO model '{model_path}' "
            f"on device '{self.device}'..."
        )

        self.model = YOLO(model_path, task="detect")

        # ============================================================
        # 3. CAMERA INTRINSICS
        # ============================================================
        self.fx = None
        self.cx = None

        # ============================================================
        # 4. TF2
        # ============================================================
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer,
            self
        )

        # ============================================================
        # 5. PUBLISHER
        # ============================================================
        self.publisher = self.create_publisher(
            SemanticTargetArray,
            "/semantic_targets",
            10
        )

        # ============================================================
        # 6. CAMERA INFO
        # ============================================================
        self.info_sub = self.create_subscription(
            CameraInfo,
            "/camera/depth/image_raw/camera_info",
            self.camera_info_callback,
            10
        )

        # ============================================================
        # 7. SYNCHRONIZED COLOR + DEPTH IMAGE
        # ============================================================

        # Callback group for the expensive synchronized
        # RGB + depth processing callback.
        self.image_cb_group = MutuallyExclusiveCallbackGroup()

        self.image_sub = message_filters.Subscriber(
            self,
            Image,
            "/camera/depth/image_raw/image",
            qos_profile=10,
            callback_group=self.image_cb_group
        )

        self.depth_sub = message_filters.Subscriber(
            self,
            Image,
            "/camera/depth/image_raw/depth_image",
            qos_profile=10,
            callback_group=self.image_cb_group
        )

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [
                self.image_sub,
                self.depth_sub
            ],
            queue_size=10,
            slop=self.sync_slop
        )

        self.ts.registerCallback(self.sync_callback)

        self.get_logger().info(
            "YOLO + RGBD Fusion Node Started! "
            "Waiting for CameraInfo & synchronized Color/Depth images..."
        )

    # ================================================================
    # CAMERA INFO
    # ================================================================
    def camera_info_callback(self, msg: CameraInfo):

        # K matrix:
        # [fx, 0, cx,
        #  0, fy, cy,
        #  0,  0,  1]

        self.fx = float(msg.k[0])
        self.cx = float(msg.k[2])

        self.get_logger().info(
            f"CameraInfo received! fx={self.fx:.2f}, "
            f"cx={self.cx:.2f}. "
            f"Unsubscribing from /camera/depth/image_raw/camera_info."
        )

        self.destroy_subscription(self.info_sub)
        self.info_sub = None

    # ================================================================
    # MAIN SYNCHRONIZED CALLBACK
    # ================================================================
    def sync_callback(
        self,
        img_msg: Image,
        depth_msg: Image
    ):

        if self.fx is None or self.cx is None:
            return

        # ------------------------------------------------------------
        # ROS Image -> OpenCV
        # ------------------------------------------------------------
        frame = self.bridge.imgmsg_to_cv2(
            img_msg,
            desired_encoding="bgr8"
        )

        # ------------------------------------------------------------
        # Depth image
        #
        # Gazebo RGBD camera normally publishes depth as float32
        # values in meters. We use passthrough so we don't alter
        # the actual depth values.
        # ------------------------------------------------------------
        depth_frame = self.bridge.imgmsg_to_cv2(
            depth_msg,
            desired_encoding="passthrough"
        )

        # ------------------------------------------------------------
        # YOLO Inference + ByteTrack
        # ------------------------------------------------------------
        results = self.model.track(
            frame,
            conf=self.conf_thresh,
            device=self.device,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False
        )

        # ============================================================
        # RAW YOLO DIAGNOSTICS
        #
        # Records YOLO output BEFORE:
        # - depth processing
        # - TF
        # - semantic targets
        # - database processing
        # ============================================================
        self.yolo_frame_count += 1

        raw_boxes = results[0].boxes
        detection_count = len(raw_boxes)

        timestamp_sec = (
            img_msg.header.stamp.sec
            + img_msg.header.stamp.nanosec * 1e-9
        )

        with open(self.yolo_csv_path, "a", newline="") as f:

            writer = csv.writer(f)

            if detection_count == 0:

                writer.writerow([
                    self.yolo_frame_count,
                    timestamp_sec,
                    0,
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    ""
                ])

            else:

                for detection_index, box in enumerate(
                    raw_boxes,
                    start=1
                ):

                    cls_id = int(box.cls[0])
                    class_name = self.model.names[cls_id]
                    confidence = float(box.conf[0])

                    x1, y1, x2, y2 = [
                        float(v)
                        for v in box.xyxy[0]
                    ]

                    bbox_width = x2 - x1
                    bbox_height = y2 - y1

                    writer.writerow([
                        self.yolo_frame_count,
                        timestamp_sec,
                        detection_count,
                        detection_index,
                        class_name,
                        f"{confidence:.4f}",
                        f"{x1:.2f}",
                        f"{y1:.2f}",
                        f"{x2:.2f}",
                        f"{y2:.2f}",
                        f"{bbox_width:.2f}",
                        f"{bbox_height:.2f}"
                    ])

        # ------------------------------------------------------------
        # Output message
        #
        # Everything is now derived from the RGBD image timestamp.
        # ------------------------------------------------------------
        semantic_array = SemanticTargetArray()

        semantic_array.header.stamp = img_msg.header.stamp

        semantic_array.header.frame_id = self.map_frame

        # ------------------------------------------------------------
        # Process detections
        # ------------------------------------------------------------
        for box in results[0].boxes:

            cls_id = int(box.cls[0])

            class_name = self.model.names[cls_id]

            confidence = float(box.conf[0])

            # ========================================================
            # BYTE TRACK ID
            # ========================================================
            track_id = int(box.id[0]) if box.id is not None else -1

            x1, y1, x2, y2 = box.xyxy[0]

            # ========================================================
            # DEPTH IMAGE SAMPLING
            # ========================================================

            # Depth image dimensions
            dh, dw = depth_frame.shape[:2]

            # Clip bbox to image boundaries
            ix1 = max(0, int(x1))
            iy1 = max(0, int(y1))

            ix2 = min(dw, int(x2))
            iy2 = min(dh, int(y2))

            if ix2 <= ix1 or iy2 <= iy1:
                continue

            # --------------------------------------------------------
            # Shrink the sampling region inward.
            #
            # Bbox edges are more likely to contain background
            # pixels, so we ignore 15% around the edges.
            # --------------------------------------------------------
            margin_x = max(
                1,
                int((ix2 - ix1) * 0.15)
            )

            margin_y = max(
                1,
                int((iy2 - iy1) * 0.15)
            )

            sx1 = min(
                ix1 + margin_x,
                ix2 - 1
            )

            sy1 = min(
                iy1 + margin_y,
                iy2 - 1
            )

            sx2 = max(
                ix2 - margin_x,
                sx1 + 1
            )

            sy2 = max(
                iy2 - margin_y,
                sy1 + 1
            )

            # --------------------------------------------------------
            # Extract depth ROI
            # --------------------------------------------------------
            depth_roi = depth_frame[
                sy1:sy2,
                sx1:sx2
            ].astype(np.float32).ravel()

            # Keep only:
            # - finite values
            # - positive values
            valid_depths = depth_roi[
                np.isfinite(depth_roi)
                & (depth_roi > 0.0)
            ]

            if valid_depths.size < 2:
                continue

            # --------------------------------------------------------
            # Nearest-cluster logic
            #
            # The foreground object should be the closest group
            # of valid depth readings in the bbox.
            # --------------------------------------------------------
            sorted_depths = np.sort(valid_depths)

            min_depth = sorted_depths[0]

            close_mask = (
                sorted_depths
                <= (min_depth + 0.3)
            )

            if close_mask.sum() < 2:
                continue

            # Use median of the closest cluster rather than one
            # potentially noisy depth pixel.
            distance = float(
                np.median(
                    sorted_depths[close_mask]
                )
            )

            # ========================================================
            # PIXEL -> ANGLE
            #
            # Use bbox center for horizontal bearing.
            # ========================================================
            cx_pix = (
                float(x1) + float(x2)
            ) / 2.0

            theta = math.atan(
                (cx_pix - self.cx) / self.fx
            )

            # ========================================================
            # POLAR -> CARTESIAN
            #
            # Coordinates are initially in the camera frame.
            # ========================================================
            raw_x = (
                distance
                * math.cos(theta)
            )

            raw_y = (
                distance
                * math.sin(theta)
            )

            # ========================================================
            # TF2: CAMERA -> MAP
            # ========================================================
            point = PointStamped()

            point.header.stamp = img_msg.header.stamp

            point.header.frame_id = (
                img_msg.header.frame_id
            )

            point.point.x = raw_x
            point.point.y = raw_y
            point.point.z = 0.0

            try:

                transformed = self.tf_buffer.transform(
                    point,
                    self.map_frame,
                    timeout=Duration(
                        seconds=self.tf_timeout
                    )
                )

            except TransformException as ex:

                self.get_logger().warn(
                    f"TF transform failed "
                    f"({img_msg.header.frame_id} -> "
                    f"{self.map_frame}) "
                    f"for {class_name}: {ex}",
                    throttle_duration_sec=2.0
                )

                continue

            # ========================================================
            # BUILD SEMANTIC TARGET
            # ========================================================
            target = SemanticTarget()

            target.class_name = class_name

            target.class_id = cls_id

            # ========================================================
            # BYTE TRACK ID
            # ========================================================
            target.track_id = track_id

            target.confidence = confidence

            target.bbox_left = float(x1)
            target.bbox_top = float(y1)
            target.bbox_right = float(x2)
            target.bbox_bottom = float(y2)

            target.x = transformed.point.x
            target.y = transformed.point.y
            target.z = transformed.point.z

            semantic_array.targets.append(target)

            self.get_logger().debug(
                f"{class_name}: "
                f"track_id={track_id} "
                f"depth={distance:.2f}m "
                f"angle={math.degrees(theta):.1f}deg "
                f"map=({target.x:.2f}, "
                f"{target.y:.2f})"
            )

        # ============================================================
        # DIAGNOSTIC:
        # SAME-FRAME BBOX + MAP-POSITION OVERLAP
        # ============================================================
        def _iou(a, b):

            ax1 = a.bbox_left
            ay1 = a.bbox_top
            ax2 = a.bbox_right
            ay2 = a.bbox_bottom

            bx1 = b.bbox_left
            by1 = b.bbox_top
            bx2 = b.bbox_right
            by2 = b.bbox_bottom

            ix1 = max(ax1, bx1)
            iy1 = max(ay1, by1)

            ix2 = min(ax2, bx2)
            iy2 = min(ay2, by2)

            iw = max(
                0.0,
                ix2 - ix1
            )

            ih = max(
                0.0,
                iy2 - iy1
            )

            inter = iw * ih

            if inter <= 0:
                return 0.0

            area_a = (
                (ax2 - ax1)
                * (ay2 - ay1)
            )

            area_b = (
                (bx2 - bx1)
                * (by2 - by1)
            )

            return inter / (
                area_a + area_b - inter
            )

        targets = semantic_array.targets

        if len(targets) > 1:

            with open(
                self.iou_csv_path,
                "a",
                newline=""
            ) as f:

                writer = csv.writer(f)

                for i in range(len(targets)):

                    for j in range(
                        i + 1,
                        len(targets)
                    ):

                        a = targets[i]
                        b = targets[j]

                        if a.class_name != b.class_name:
                            continue

                        iou = _iou(a, b)

                        writer.writerow([
                            self.yolo_frame_count,
                            timestamp_sec,
                            a.class_name,
                            i,
                            j,
                            f"{a.confidence:.4f}",
                            f"{b.confidence:.4f}",
                            f"{iou:.4f}",
                            f"{a.x:.3f}",
                            f"{a.y:.3f}",
                            f"{b.x:.3f}",
                            f"{b.y:.3f}"
                        ])

        # ============================================================
        # PUBLISH
        # ============================================================
        self.publisher.publish(
            semantic_array
        )

        self.get_logger().info(
            f"Published "
            f"{len(semantic_array.targets)} "
            f"targets in '{self.map_frame}' frame",
            throttle_duration_sec=1.0
        )

        # ============================================================
        # DEBUG GUI
        # ============================================================
        if self.show_window:

            annotated = results[0].plot()

            cv2.imshow(
                "YOLO26 RGBD Fusion",
                annotated
            )

            cv2.waitKey(1)


def main(args=None):

    rclpy.init(args=args)

    node = YOLODetector()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()

    except KeyboardInterrupt:
        pass

    finally:
        executor.shutdown()
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == "__main__":
    main()