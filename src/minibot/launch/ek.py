import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression

from launch_ros.actions import Node


def generate_launch_description():

    package_name = 'minibot'
    package_dir = get_package_share_directory(package_name)

    use_sim_time = LaunchConfiguration('use_sim_time')
    map = LaunchConfiguration('map')
    use_slam_option = LaunchConfiguration('use_slam_option')

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation clock'
    )

    declare_map = DeclareLaunchArgument(
        'map',
        default_value='./src/minibot/maps/sample_map.yaml',
        description='Map file'
    )

    declare_use_slam_option = DeclareLaunchArgument(
        'use_slam_option',
        default_value='online_async_slam',
        description='Choose: online_async_slam, mapper_params_localization, amcl'
    )

    # ------------------------------------------------------------
    # Configuration files
    # ------------------------------------------------------------

    joy_params_file = os.path.join(
        package_dir,
        'config',
        'joystick_params.yaml'
    )

    mapper_params_online_async_file = os.path.join(
        package_dir,
        'config',
        'mapper_params_online_async.yaml'
    )

    mapper_params_localization_file = os.path.join(
        package_dir,
        'config',
        'mapper_params_localization.yaml'
    )

    nav2_params_file = os.path.join(
        package_dir,
        'config',
        'nav2_params.yaml'
    )

    ekf_params_file = os.path.join(
        package_dir,
        'config',
        'ekf.yaml'
    )

    # ------------------------------------------------------------
    # Joy Node
    # ------------------------------------------------------------

    joy_node = Node(
        package='joy',
        executable='joy_node',
        parameters=[joy_params_file]
    )

    # ------------------------------------------------------------
    # Teleop
    # ------------------------------------------------------------

    teleop_node = Node(
        package='teleop_twist_joy',
        executable='teleop_node',
        name='teleop_node',
        parameters=[joy_params_file],
        remappings=[
            ('/cmd_vel', '/joy_vel')
        ]
    )

    # ------------------------------------------------------------
    # EKF
    # ------------------------------------------------------------

    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[
            ekf_params_file,
            {'use_sim_time': use_sim_time}
        ]
    )

    # ------------------------------------------------------------
    # SLAM Toolbox (Online Mapping)
    # ------------------------------------------------------------

    online_async_slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('slam_toolbox'),
                'launch',
                'online_async_launch.py'
            )
        ),
        launch_arguments={
            'slam_params_file': mapper_params_online_async_file,
            'use_sim_time': use_sim_time
        }.items(),
        condition=IfCondition(
            PythonExpression(
                ["'", use_slam_option, "' == 'online_async_slam'"]
            )
        )
    )

    # ------------------------------------------------------------
    # SLAM Toolbox Localization
    # ------------------------------------------------------------

    mapper_params_localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('slam_toolbox'),
                'launch',
                'localization_launch.py'
            )
        ),
        launch_arguments={
            'slam_params_file': mapper_params_localization_file,
            'use_sim_time': use_sim_time
        }.items(),
        condition=IfCondition(
            PythonExpression(
                ["'", use_slam_option, "' == 'mapper_params_localization'"]
            )
        )
    )

    # ------------------------------------------------------------
    # AMCL
    # ------------------------------------------------------------
    
    amcl = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('nav2_bringup'),
                'launch',
                'localization_launch.py'
            )
        ),
        launch_arguments={
            'map': map,
            'use_sim_time': use_sim_time
        }.items(),
        condition=IfCondition(
            PythonExpression(
                ["'", use_slam_option, "' == 'amcl'"]
            )
        )
    )

    # ------------------------------------------------------------
    # Navigation2
    # ------------------------------------------------------------

    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('nav2_bringup'),
                'launch',
                'navigation_launch.py'
            )
        ),
        launch_arguments={
            'params_file': nav2_params_file,
            'use_sim_time': use_sim_time
        }.items()
    )

    # ------------------------------------------------------------
    # Launch Description
    # ------------------------------------------------------------

    ld = LaunchDescription()

    ld.add_action(declare_use_sim_time)
    ld.add_action(declare_map)
    ld.add_action(declare_use_slam_option)

    ld.add_action(joy_node)
    ld.add_action(teleop_node)

    # EKF
    ld.add_action(ekf_node)

    # SLAM
    ld.add_action(online_async_slam)
    ld.add_action(mapper_params_localization)
    ld.add_action(amcl)

    # Navigation
    ld.add_action(navigation)

    return ld