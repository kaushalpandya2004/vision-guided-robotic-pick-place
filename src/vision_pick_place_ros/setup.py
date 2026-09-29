from setuptools import setup

package_name = "vision_pick_place_ros"

setup(
    name=package_name,
    version="1.0.0",
    packages=[package_name],
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        (
            "share/" + package_name,
            ["package.xml"],
        ),
        (
            "share/" + package_name + "/launch",
            ["launch/vision_pick_place.launch.py"],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Kaushal Pandya",
    maintainer_email="kaushal@example.com",
    description=(
        "ROS2 architecture for vision-guided "
        "robotic pick-and-place"
    ),
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "camera_node = vision_pick_place_ros.camera_node:main",
        "selection_controller_node = vision_pick_place_ros.selection_controller_node:main",
        "mission_ui_node = vision_pick_place_ros.mission_ui_node:main",
            "object_detection_node = vision_pick_place_ros.object_detection_node:main",
            "coordinate_transform_node = vision_pick_place_ros.coordinate_transform_node:main",
            "motion_planning_node = vision_pick_place_ros.motion_planning_node:main",
            "robot_controller_node = vision_pick_place_ros.robot_controller_node:main",
            "gripper_controller_node = vision_pick_place_ros.gripper_controller_node:main",
        ],
    },
)
