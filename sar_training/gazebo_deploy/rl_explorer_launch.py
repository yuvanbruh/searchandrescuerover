"""
Launch file for the RL-vs-Mission-2 test. It replaces your 3rd launch file:
starts YOLO + semantic database (same as before) and runs explorer_rl_node.py
INSTEAD of the 'mission2' executable.

Run it from any folder (ROS sourced, sar_venv NOT active):

  ros2 launch ~/last/sar_training/gazebo_deploy/rl_explorer_launch.py \
      base_module:=frontier_explorer.<your_mission2_file> \
      experiment_name:=rl_test01

Arguments (all have defaults):
  policy_mode      rl | mission2        (default rl)
  base_module      python module of your ORIGINAL explorer node (see setup.py,
                   the line  mission2 = <this module>:main)
  model_path       path to policy.npz   (default ~/last/sar_training/policy.npz)
  script_path      path to explorer_rl_node.py
  experiment_name  label written to the CSV files
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    home = os.path.expanduser('~')
    args = [
        DeclareLaunchArgument('policy_mode', default_value='rl'),
        DeclareLaunchArgument('base_module', default_value='explorer_node'),
        DeclareLaunchArgument('model_path',
                              default_value=os.path.join(home, 'last/sar_training/policy.npz')),
        DeclareLaunchArgument('script_path',
                              default_value=os.path.join(
                                  home, 'last/sar_training/gazebo_deploy/explorer_rl_node.py')),
        DeclareLaunchArgument('experiment_name', default_value='rl_test01'),
    ]

    rl_explorer = ExecuteProcess(
        cmd=['python3', LaunchConfiguration('script_path'), '--ros-args',
             '-p', 'use_sim_time:=true',
             '-p', ['policy_mode:=', LaunchConfiguration('policy_mode')],
             '-p', ['rl_model_path:=', LaunchConfiguration('model_path')],
             '-p', ['experiment_name:=', LaunchConfiguration('experiment_name')]],
        additional_env={'EXPLORER_BASE_MODULE': LaunchConfiguration('base_module'),
                        'PYTHONUNBUFFERED': '1'},
        output='screen',
    )

    yolo_detector = Node(package='semantic_perception', executable='yolo_detector',
                         name='yolo_detector', output='screen',
                         parameters=[{'use_sim_time': True}])
    semantic_database = Node(package='semantic_perception', executable='semantic_database',
                             name='semantic_database', output='screen',
                             parameters=[{'use_sim_time': True}])

    return LaunchDescription(args + [yolo_detector, semantic_database, rl_explorer])
