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
⚠️ Limitations

The current implementation is a simulation prototype and has several limitations.

Camera

The camera is simulated rather than physically calibrated.

The system therefore does not currently include:

Real camera intrinsics
Real camera extrinsics
Lens distortion calibration
Real depth-camera noise
Real image noise
Coordinate Transformation

The current transformation is designed for the known simulated camera/workspace geometry.

A real-world implementation would require a calibrated transformation between the camera and robot base frames.

Computer Vision

HSV thresholds are designed for the controlled simulated environment.

Performance may change under:

Different lighting
Shadows
Reflections
Similar object colors
Occlusions
Camera noise
Robot

The current system is simulation-only.

Real deployment would introduce:

Servo errors
Joint backlash
Gear compliance
Motor dynamics
Sensor noise
Calibration errors
Communication delays
Real collision dynamics
Gripper

The current gripper is simulated.

Real grasping would require additional consideration of:

Object friction
Gripper force
Contact dynamics
Object geometry
Slip detection
Force feedback
🚀 Future Improvements

Potential extensions include:

Vision
RGB-D camera
Depth-based localization
Camera calibration
YOLO-based object detection
Robust detection under lighting changes
Object pose estimation
Robotics
ROS2 TF2 integration
MoveIt 2 integration
Real robot deployment
Cartesian trajectory generation
Dynamic obstacle avoidance
Velocity and acceleration constraints
Motion Planning
RRT*
Informed RRT*
Trajectory optimization
Cost-based path selection
Dynamic replanning
Manipulation
Automatic grasp pose generation
Force/torque feedback
Slip detection
Grasp quality estimation
Multi-object task planning
System Integration
Hardware-in-the-loop testing
Real camera integration
Real-time monitoring
Industrial robot controller integration
Automated experiment logging
