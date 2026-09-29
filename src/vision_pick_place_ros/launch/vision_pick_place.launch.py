import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, AppendEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    target_id = LaunchConfiguration('target_id')

    pythonpath = AppendEnvironmentVariable(
        name='PYTHONPATH',
        value=os.pathsep + '/mnt/e/Robotics_Assignment/.venv/lib/python3.12/site-packages',
    )

    camera_node = Node(
        package='vision_pick_place_ros',
        executable='camera_node',
        name='camera_node',
        output='screen',
    )

    detection_node = Node(
        package='vision_pick_place_ros',
        executable='object_detection_node',
        name='object_detection_node',
        output='screen',
    )

    transform_node = Node(
        package='vision_pick_place_ros',
        executable='coordinate_transform_node',
        name='coordinate_transform_node',
        parameters=[
            {'target_id': target_id}
        ],
        output='screen',
    )

    motion_node = Node(
        package='vision_pick_place_ros',
        executable='motion_planning_node',
        name='motion_planning_node',
        output='screen',
    )

    selection_node = Node(
        package='vision_pick_place_ros',
        executable='selection_controller_node',
        name='selection_controller_node',
        output='screen',
    )

    mission_ui_node = Node(
        package='vision_pick_place_ros',
        executable='mission_ui_node',
        name='mission_ui_node',
        output='screen',
    )

    return LaunchDescription([
        pythonpath,

        DeclareLaunchArgument(
            'target_id',
            default_value='C',
            description='Initial detected target ID: A, B, or C'
        ),

        camera_node,
        detection_node,
        transform_node,
        motion_node,
        selection_node,
        mission_ui_node,
    ])
