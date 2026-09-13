#!/usr/bin/env python3

import csv
import os
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, LaserScan


class TFRotationDiagnostic(Node):

    def __init__(self):
        super().__init__('tf_rotation_diagnostic')

        # Change these only if your frame names are different.
        self.map_frame = 'map'
        self.odom_frame = 'odom'
        self.base_frame = 'base_link'
        self.camera_frame = 'camera_link'

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        self.log_dir = os.path.expanduser(
            f'~/tf_diagnostic_{timestamp}'
        )
        os.makedirs(self.log_dir, exist_ok=True)

        self.get_logger().info(f'Logging diagnostics to: {self.log_dir}')

        # TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )

        # CSV files
        self.tf_file = open(
            os.path.join(self.log_dir, 'tf_transforms.csv'),
            'w',
            newline=''
        )

        self.odom_file = open(
            os.path.join(self.log_dir, 'odom.csv'),
            'w',
            newline=''
        )

        self.imu_file = open(
            os.path.join(self.log_dir, 'imu.csv'),
            'w',
            newline=''
        )

        self.scan_file = open(
            os.path.join(self.log_dir, 'scan.csv'),
            'w',
            newline=''
        )

        self.tf_writer = csv.writer(self.tf_file)
        self.odom_writer = csv.writer(self.odom_file)
        self.imu_writer = csv.writer(self.imu_file)
        self.scan_writer = csv.writer(self.scan_file)

        self.tf_writer.writerow([
            'time',
            'transform',
            'x',
            'y',
            'z',
            'qx',
            'qy',
            'qz',
            'qw'
        ])

        self.odom_writer.writerow([
            'time',
            'position_x',
            'position_y',
            'position_z',
            'linear_x',
            'linear_y',
            'angular_z'
        ])

        self.imu_writer.writerow([
            'time',
            'angular_x',
            'angular_y',
            'angular_z',
            'linear_accel_x',
            'linear_accel_y',
            'linear_accel_z'
        ])

        self.scan_writer.writerow([
            'time',
            'min_range',
            'max_range',
            'valid_ranges'
        ])

        self.latest_odom = None
        self.latest_imu = None
        self.latest_scan = None

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/imu',
            self.imu_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # Log TF at 10 Hz
        self.timer = self.create_timer(
            0.1,
            self.timer_callback
        )

        self.get_logger().info('TF rotation diagnostic started.')

    def now_string(self):
        return self.get_clock().now().nanoseconds / 1e9

    def odom_callback(self, msg):
        self.latest_odom = msg

        p = msg.pose.pose.position
        t = msg.twist.twist

        self.odom_writer.writerow([
            msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9,
            p.x,
            p.y,
            p.z,
            t.linear.x,
            t.linear.y,
            t.angular.z
        ])

        self.odom_file.flush()

    def imu_callback(self, msg):
        self.latest_imu = msg

        a = msg.angular_velocity
        acc = msg.linear_acceleration

        self.imu_writer.writerow([
            msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9,
            a.x,
            a.y,
            a.z,
            acc.x,
            acc.y,
            acc.z
        ])

        self.imu_file.flush()

    def scan_callback(self, msg):
        self.latest_scan = msg

        valid_ranges = [
            r for r in msg.ranges
            if msg.range_min <= r <= msg.range_max
        ]

        min_range = min(valid_ranges) if valid_ranges else 0.0
        max_range = max(valid_ranges) if valid_ranges else 0.0

        self.scan_writer.writerow([
            msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9,
            min_range,
            max_range,
            len(valid_ranges)
        ])

        self.scan_file.flush()

    def log_transform(self, target, source, label):
        try:
            transform = self.tf_buffer.lookup_transform(
                target,
                source,
                Time()
            )

            t = transform.transform.translation
            r = transform.transform.rotation

            self.tf_writer.writerow([
                self.now_string(),
                label,
                t.x,
                t.y,
                t.z,
                r.x,
                r.y,
                r.z,
                r.w
            ])

            return t.x, t.y, t.z

        except Exception as e:
            self.get_logger().warn(
                f'Could not get {target} -> {source}: {str(e)}',
                throttle_duration_sec=5.0
            )

            self.tf_writer.writerow([
                self.now_string(),
                label,
                'TF_ERROR',
                '',
                '',
                '',
                '',
                '',
                ''
            ])

            return None

    def timer_callback(self):
        map_base = self.log_transform(
            self.map_frame,
            self.base_frame,
            'map_to_base_link'
        )

        map_odom = self.log_transform(
            self.map_frame,
            self.odom_frame,
            'map_to_odom'
        )

        odom_base = self.log_transform(
            self.odom_frame,
            self.base_frame,
            'odom_to_base_link'
        )

        base_camera = self.log_transform(
            self.base_frame,
            self.camera_frame,
            'base_link_to_camera_link'
        )

        self.tf_file.flush()

        # Print the most important transform once per second.
        if map_base is not None:
            self.get_logger().info(
                f'map->base_link: '
                f'x={map_base[0]:.3f}, '
                f'y={map_base[1]:.3f}, '
                f'z={map_base[2]:.3f}'
            )

    def destroy_node(self):
        self.get_logger().info(
            f'Diagnostics saved in: {self.log_dir}'
        )

        self.tf_file.close()
        self.odom_file.close()
        self.imu_file.close()
        self.scan_file.close()

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = TFRotationDiagnostic()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
