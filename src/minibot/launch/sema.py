import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    # ------------------------------------------------------------
    # Frontier Exploration Utility
    # ------------------------------------------------------------

    frontier_explorer = Node(
        package='frontier_explorer',
        executable='mission2',
        name='mission2',
        output='screen',
        parameters=[
            {
                'use_sim_time': True,
            }
        ]
    )

    # ------------------------------------------------------------
    # YOLO ROS Detector
    # ------------------------------------------------------------

    yolo_detector = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('yolo_bringup'),
                'launch',
                'yolo.launch.py'
            )
        ),
        launch_arguments={
            # Model
            'model': '/home/yuvan/last/yolo26n_openvino_model/',

            # Camera topics
            'input_image_topic': '/camera/depth/image_raw/image',
            'input_depth_topic': '/camera/depth/image_raw/depth_image',
            'input_depth_info_topic': '/camera/depth/image_raw/camera_info',

            # QoS: Best Effort for Gazebo camera topics
            'image_reliability': '2',
            'depth_image_reliability': '2',
            'depth_info_reliability': '2',

            # Detection settings
            'threshold': '0.5',

            # 3D detections
            'target_frame': 'map',
            'use_3d': 'True',

            # Tracking
            'use_tracking': 'True',
        }.items()
    )

    # ------------------------------------------------------------
    # Semantic Adapter
    # YOLO Detection3D -> SemanticTargetArray
    # ------------------------------------------------------------

    semantic_adapter = Node(
        package='semantic_perception',
        executable='semantic_adapter',
        name='semantic_adapter',
        output='screen',
        parameters=[
            {
                'use_sim_time': True,
            }
        ]
    )

    # ------------------------------------------------------------
    # Semantic Database
    # ------------------------------------------------------------

    semantic_database = Node(
        package='semantic_perception',
        executable='semantic_database',
        name='semantic_database',
        output='screen',
        parameters=[
            {
                'use_sim_time': True,
                'map_frame': 'map',
            }
        ]
    )

    # ------------------------------------------------------------
    # Launch Description
    # ------------------------------------------------------------

    return LaunchDescription([
        frontier_explorer,
        yolo_detector,
        semantic_adapter,
        semantic_database,
    ])