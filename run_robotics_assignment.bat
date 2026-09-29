@echo off
title Vision-Guided Robotic Pick and Place - ROS2 Jazzy

echo ============================================================
echo   VISION-GUIDED ROBOTIC PICK AND PLACE
echo   ROS2 Jazzy + PyBullet + OpenCV + RRT
echo ============================================================
echo.

echo Starting WSL...
echo.

wsl.exe -- bash -lc "cd /mnt/e/Robotics_Assignment && set +u && source /opt/ros/jazzy/setup.bash && source .venv/bin/activate && source install/setup.bash && echo 'ROS2 environment ready.' && echo 'Starting complete robotics system...' && echo && ros2 launch vision_pick_place_ros vision_pick_place.launch.py"

echo.
echo ============================================================
echo   Robotics system has stopped.
echo ============================================================
echo.
pause
