from launch import LaunchDescription
from launch_ros.actions import Node


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
    # YOLO Semantic Detector
    # ------------------------------------------------------------

    yolo_detector = Node(
        package='semantic_perception',
        executable='yolo_detector',
        name='yolo_detector',
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
            }
        ]
    )

    # ------------------------------------------------------------
    # Launch Description
    # ------------------------------------------------------------

    return LaunchDescription([
        frontier_explorer,
        yolo_detector,
        semantic_database,
    ])