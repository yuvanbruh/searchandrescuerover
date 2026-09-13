#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from yolo_msgs.msg import DetectionArray
from semantic_interfaces.msg import SemanticTarget, SemanticTargetArray


class YoloSemanticAdapter(Node):

    def __init__(self):
        super().__init__('yolo_semantic_adapter')

        self.declare_parameter(
            'input_topic',
            '/yolo/detections_3d'
        )

        self.declare_parameter(
            'output_topic',
            '/semantic_targets'
        )

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value

        self.publisher = self.create_publisher(
            SemanticTargetArray,
            output_topic,
            10
        )

        self.subscription = self.create_subscription(
            DetectionArray,
            input_topic,
            self.detection_callback,
            10
        )

        self.get_logger().info(
            f'YOLO Semantic Adapter started: '
            f'{input_topic} -> {output_topic}'
        )

    def detection_callback(self, msg):

        output = SemanticTargetArray()
        output.header = msg.header

        for detection in msg.detections:

            target = SemanticTarget()

            # Database will assign the permanent ID.
            target.id = 0

            # YOLO tracking ID.
            target.track_id = -1

            target.class_name = detection.class_name
            target.class_id = detection.class_id
            target.confidence = float(detection.score)

            # New detections have not been inspected yet.
            target.has_snapshot = False

            # 2D bounding box.
            target.bbox_left = float(
                detection.bbox.center.position.x -
                detection.bbox.size.x / 2.0
            )

            target.bbox_top = float(
                detection.bbox.center.position.y -
                detection.bbox.size.y / 2.0
            )

            target.bbox_right = float(
                detection.bbox.center.position.x +
                detection.bbox.size.x / 2.0
            )

            target.bbox_bottom = float(
                detection.bbox.center.position.y +
                detection.bbox.size.y / 2.0
            )

            # 3D position from YOLO.
            target.x = float(
                detection.bbox3d.center.position.x
            )

            target.y = float(
                detection.bbox3d.center.position.y
            )

            target.z = float(
                detection.bbox3d.center.position.z
            )

            output.targets.append(target)

        self.publisher.publish(output)

        self.get_logger().info(
            f'Published {len(output.targets)} semantic target(s)'
        )


def main(args=None):
    rclpy.init(args=args)

    node = YoloSemanticAdapter()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
