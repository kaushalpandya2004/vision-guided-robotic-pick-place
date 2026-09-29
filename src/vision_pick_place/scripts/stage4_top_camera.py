#!/usr/bin/env python3

import time

import cv2
import numpy as np
import pybullet as p
import pybullet_data


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_WIDTH = 800
IMAGE_HEIGHT = 800

CAMERA_POSITION = [0.0, 0.0, 2.50]
CAMERA_TARGET = [0.0, 0.0, 0.0]
CAMERA_UP = [0.0, 1.0, 0.0]

CAMERA_FOV = 60.0
CAMERA_NEAR = 0.01
CAMERA_FAR = 5.0


# ============================================================
# TABLE
# ============================================================

def create_table():

    half_x = 1.20
    half_y = 1.00
    half_z = 0.05

    collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[
            half_x,
            half_y,
            half_z
        ]
    )

    visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[
            half_x,
            half_y,
            half_z
        ],
        rgbaColor=[
            0.70,
            0.70,
            0.70,
            1.0
        ]
    )

    return p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=[
            0.0,
            0.0,
            -0.05
        ]
    )


# ============================================================
# COLORED OBJECT
# ============================================================

def create_cylinder(position, color):

    radius = 0.035
    height = 0.08

    collision = p.createCollisionShape(
        p.GEOM_CYLINDER,
        radius=radius,
        height=height
    )

    visual = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=radius,
        length=height,
        rgbaColor=color
    )

    return p.createMultiBody(
        baseMass=0.05,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position
    )


# ============================================================
# WHITE OBSTACLE
# ============================================================

def create_obstacle():

    half_x = 0.10
    half_y = 0.10
    half_z = 0.15

    collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[
            half_x,
            half_y,
            half_z
        ]
    )

    visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[
            half_x,
            half_y,
            half_z
        ],
        rgbaColor=[
            1.0,
            1.0,
            1.0,
            1.0
        ]
    )

    return p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=[
            0.25,
            0.0,
            0.15
        ]
    )


# ============================================================
# BLACK DROP PLATE
# ============================================================

def create_drop_plate():

    half_x = 0.13
    half_y = 0.13
    half_z = 0.008

    collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[
            half_x,
            half_y,
            half_z
        ]
    )

    visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[
            half_x,
            half_y,
            half_z
        ],
        rgbaColor=[
            0.02,
            0.02,
            0.02,
            1.0
        ]
    )

    return p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=[
            0.55,
            0.40,
            0.008
        ]
    )


# ============================================================
# FRANKA PANDA
# ============================================================

def load_panda():

    panda_id = p.loadURDF(
        "franka_panda/panda.urdf",
        basePosition=[
            0.0,
            0.0,
            0.0
        ],
        baseOrientation=p.getQuaternionFromEuler(
            [0.0, 0.0, 0.0]
        ),
        useFixedBase=True,
        flags=p.URDF_USE_INERTIA_FROM_FILE
    )

    home_configuration = [
        0.0,
        -0.785398163,
        0.0,
        -2.356194490,
        0.0,
        1.570796327,
        0.785398163
    ]

    for joint_index in range(7):

        p.resetJointState(
            panda_id,
            joint_index,
            home_configuration[joint_index]
        )

    return panda_id


# ============================================================
# CAMERA MATRICES
# ============================================================

def create_camera_matrices():

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=CAMERA_POSITION,
        cameraTargetPosition=CAMERA_TARGET,
        cameraUpVector=CAMERA_UP
    )

    projection_matrix = p.computeProjectionMatrixFOV(
        fov=CAMERA_FOV,
        aspect=IMAGE_WIDTH / IMAGE_HEIGHT,
        nearVal=CAMERA_NEAR,
        farVal=CAMERA_FAR
    )

    return view_matrix, projection_matrix


# ============================================================
# CAPTURE RGB IMAGE
# ============================================================

def capture_top_camera(view_matrix, projection_matrix):

    result = p.getCameraImage(
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_TINY_RENDERER
    )

    if len(result) < 3:
        raise RuntimeError(
            "PyBullet camera returned an invalid result."
        )

    rgba_buffer = np.asarray(
        result[2],
        dtype=np.uint8
    )

    expected_pixels = (
        IMAGE_WIDTH *
        IMAGE_HEIGHT *
        4
    )

    if rgba_buffer.size != expected_pixels:

        raise RuntimeError(
            "Unexpected camera buffer size: "
            f"{rgba_buffer.size}. "
            f"Expected {expected_pixels}."
        )

    rgba = rgba_buffer.reshape(
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        4
    )

    rgb = rgba[:, :, :3]

    bgr = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR
    )

    return bgr


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("STAGE 4 - TOP VIEW RGB CAMERA")
    print("=" * 70)

    physics_client = p.connect(
        p.GUI
    )

    if physics_client < 0:

        raise RuntimeError(
            "Unable to connect to PyBullet."
        )

    p.setAdditionalSearchPath(
        pybullet_data.getDataPath()
    )

    p.setGravity(
        0.0,
        0.0,
        -9.81
    )

    p.setTimeStep(
        1.0 / 240.0
    )

    p.setRealTimeSimulation(0)

    # --------------------------------------------------------
    # SCENE
    # --------------------------------------------------------

    create_table()

    load_panda()

    create_cylinder(
        [-0.45, -0.35, 0.08],
        [1.0, 0.0, 0.0, 1.0]
    )

    create_cylinder(
        [-0.15, 0.35, 0.08],
        [0.0, 0.0, 1.0, 1.0]
    )

    create_cylinder(
        [0.05, -0.35, 0.08],
        [0.0, 1.0, 0.0, 1.0]
    )

    create_obstacle()

    create_drop_plate()

    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------

    view_matrix, projection_matrix = (
        create_camera_matrices()
    )

    print()
    print("Camera configuration")
    print("-" * 40)
    print(f"Resolution : {IMAGE_WIDTH} x {IMAGE_HEIGHT}")
    print(f"Position   : {CAMERA_POSITION}")
    print(f"Target     : {CAMERA_TARGET}")
    print(f"FOV        : {CAMERA_FOV} degrees")
    print()

    # --------------------------------------------------------
    # CAMERA TEST
    # --------------------------------------------------------

    frame_count = 0

    print("Starting camera stream...")
    print("Press Q in the OpenCV window to stop.")
    print()

    while p.isConnected():

        p.stepSimulation()

        frame = capture_top_camera(
            view_matrix,
            projection_matrix
        )

        frame_count += 1

        # Display frame number for verification.
        cv2.putText(
            frame,
            f"Frame: {frame_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow(
            "Top View RGB Camera",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            break

        time.sleep(
            1.0 / 240.0
        )

    cv2.destroyAllWindows()

    if p.isConnected():
        p.disconnect()

    print()
    print("=" * 70)
    print("STAGE 4 CAMERA TEST COMPLETE")
    print(f"Frames captured: {frame_count}")
    print("=" * 70)


if __name__ == "__main__":
    main()
