# Vision-Guided Robotic Pick-and-Place with Motion Planning

<p align="center">

### ROS2 Jazzy • PyBullet • OpenCV • RRT • Franka Panda

A simulation-based robotic manipulation system that combines computer vision, coordinate transformation, inverse kinematics, collision-aware motion planning, obstacle avoidance, and automated pick-and-place execution.

</p>

---

## 📌 Project Overview

This project implements a complete **vision-guided robotic pick-and-place pipeline** using a simulated Franka Panda robotic arm in PyBullet with a modular ROS2 architecture.

The system operates in a simulated workspace containing:

- A 7-DOF Franka Panda robotic arm
- A simulated RGB camera
- A table/work surface
- Three colored objects
- A physical obstacle
- A predefined placement plate
- A simulated gripper

The robot observes the workspace using the camera, detects the available objects using OpenCV-based HSV segmentation, estimates the selected object's world position, calculates an inverse-kinematic solution, generates a collision-free trajectory using RRT, avoids the obstacle, performs the pick-and-place operation, releases the object, and returns to the home configuration.

The complete pipeline is:

```text
                    ┌─────────────────────┐
                    │   Simulated RGB     │
                    │       Camera        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Object Detection    │
                    │   OpenCV + HSV      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Object Localization │
                    │ Pixel Centroid      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Camera → World      │
                    │ Transformation      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Target Selection    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Inverse Kinematics  │
                    │ Position + Pose     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Motion Planning     │
                    │ Direct Path / RRT   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Collision Checking  │
                    └──────────┬──────────┘
                               │
                               ▼
             ┌─────────────────────────────────────┐
             │        Pick-and-Place Sequence      │
             │                                     │
             │ HOME → PRE-GRASP → GRASP → LIFT     │
             │      → OBSTACLE AVOIDANCE           │
             │      → PLACE → RELEASE → HOME       │
             └─────────────────────────────────────┘
