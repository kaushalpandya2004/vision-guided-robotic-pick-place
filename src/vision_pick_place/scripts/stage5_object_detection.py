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

MIN_CONTOUR_AREA = 150.0


# ============================================================
# TABLE
# ============================================================

def create_table():

    collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[1.20, 1.00, 0.05]
    )

    visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[1.20, 1.00, 0.05],
        rgbaColor=[0.70, 0.70, 0.70, 1.0]
    )

    return p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=[0.0, 0.0, -0.05]
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

    collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[0.10, 0.10, 0.15]
    )

    visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[0.10, 0.10, 0.15],
        rgbaColor=[1.0, 1.0, 1.0, 1.0]
    )

    return p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=[0.25, 0.0, 0.15]
    )


# ============================================================
# BLACK DROP PLATE
# ============================================================

def create_drop_plate():

    collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[0.13, 0.13, 0.008]
    )

    visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[0.13, 0.13, 0.008],
        rgbaColor=[0.02, 0.02, 0.02, 1.0]
    )

    return p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=[0.55, 0.40, 0.008]
    )


# ============================================================
# PANDA
# ============================================================

def load_panda():

    panda_id = p.loadURDF(
        "franka_panda/panda.urdf",
        basePosition=[0.0, 0.0, 0.0],
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
# CAMERA
# ============================================================

def create_camera():

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


def capture_camera(view_matrix, projection_matrix):

    result = p.getCameraImage(
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_TINY_RENDERER
    )

    rgba_buffer = np.asarray(
        result[2],
        dtype=np.uint8
    )

    expected_size = IMAGE_WIDTH * IMAGE_HEIGHT * 4

    if rgba_buffer.size != expected_size:

        raise RuntimeError(
            f"Invalid camera buffer size: "
            f"{rgba_buffer.size}, "
            f"expected {expected_size}"
        )

    rgba = rgba_buffer.reshape(
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        4
    )

    rgb = rgba[:, :, :3]

    frame = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR
    )

    return frame


# ============================================================
# HSV OBJECT DETECTION
# ============================================================

def detect_colored_objects(frame):

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

    detections = {}

    # --------------------------------------------------------
    # RED
    # --------------------------------------------------------

    red_lower_1 = np.array(
        [0, 100, 80],
        dtype=np.uint8
    )

    red_upper_1 = np.array(
        [10, 255, 255],
        dtype=np.uint8
    )

    red_lower_2 = np.array(
        [170, 100, 80],
        dtype=np.uint8
    )

    red_upper_2 = np.array(
        [179, 255, 255],
        dtype=np.uint8
    )

    red_mask_1 = cv2.inRange(
        hsv,
        red_lower_1,
        red_upper_1
    )

    red_mask_2 = cv2.inRange(
        hsv,
        red_lower_2,
        red_upper_2
    )

    red_mask = cv2.bitwise_or(
        red_mask_1,
        red_mask_2
    )

    detections["RED"] = find_object(
        red_mask
    )

    # --------------------------------------------------------
    # BLUE
    # --------------------------------------------------------

    blue_lower = np.array(
        [100, 100, 80],
        dtype=np.uint8
    )

    blue_upper = np.array(
        [135, 255, 255],
        dtype=np.uint8
    )

    blue_mask = cv2.inRange(
        hsv,
        blue_lower,
        blue_upper
    )

    detections["BLUE"] = find_object(
        blue_mask
    )

    # --------------------------------------------------------
    # GREEN
    # --------------------------------------------------------

    green_lower = np.array(
        [40, 80, 60],
        dtype=np.uint8
    )

    green_upper = np.array(
        [85, 255, 255],
        dtype=np.uint8
    )

    green_mask = cv2.inRange(
        hsv,
        green_lower,
        green_upper
    )

    detections["GREEN"] = find_object(
        green_mask
    )

    return detections


# ============================================================
# CONTOUR PROCESSING
# ============================================================

def find_object(mask):

    kernel = np.ones(
        (5, 5),
        dtype=np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    valid_contours = [
        contour
        for contour in contours
        if cv2.contourArea(contour) >= MIN_CONTOUR_AREA
    ]

    if not valid_contours:

        return None

    contour = max(
        valid_contours,
        key=cv2.contourArea
    )

    area = cv2.contourArea(
        contour
    )

    moments = cv2.moments(
        contour
    )

    if moments["m00"] == 0:

        return None

    center_x = int(
        moments["m10"] /
        moments["m00"]
    )

    center_y = int(
        moments["m01"] /
        moments["m00"]
    )

    x, y, width, height = cv2.boundingRect(
        contour
    )

    return {
        "center": (
            center_x,
            center_y
        ),
        "area": area,
        "bbox": (
            x,
            y,
            width,
            height
        )
    }


# ============================================================
# DRAW DETECTIONS
# ============================================================

def draw_detections(frame, detections):

    display_colors = {
        "RED": (0, 0, 255),
        "BLUE": (255, 0, 0),
        "GREEN": (0, 255, 0)
    }

    for color_name, detection in detections.items():

        if detection is None:

            continue

        center_x, center_y = detection["center"]

        x, y, width, height = detection["bbox"]

        display_color = display_colors[
            color_name
        ]

        # Bounding box
        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            display_color,
            3
        )

        # Center point
        cv2.circle(
            frame,
            (center_x, center_y),
            7,
            display_color,
            -1
        )

        # Crosshair
        cv2.line(
            frame,
            (center_x - 15, center_y),
            (center_x + 15, center_y),
            display_color,
            2
        )

        cv2.line(
            frame,
            (center_x, center_y - 15),
            (center_x, center_y + 15),
            display_color,
            2
        )

        # Label
        label = (
            f"{color_name} OBJECT"
        )

        cv2.putText(
            frame,
            label,
            (x, max(y - 35, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            display_color,
            2,
            cv2.LINE_AA
        )

        # Coordinates
        coordinate_text = (
            f"Center: "
            f"({center_x}, {center_y})"
        )

        cv2.putText(
            frame,
            coordinate_text,
            (x, max(y - 10, 50)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # Area
        area_text = (
            f"Area: "
            f"{detection['area']:.0f}px"
        )

        cv2.putText(
            frame,
            area_text,
            (x, y + height + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    return frame


# ============================================================
# STATUS PANEL
# ============================================================

def draw_status_panel(frame, detections):

    detected_count = sum(
        detection is not None
        for detection in detections.values()
    )

    panel_text = (
        f"Objects detected: "
        f"{detected_count}/3"
    )

    cv2.rectangle(
        frame,
        (10, 10),
        (280, 55),
        (20, 20, 20),
        -1
    )

    cv2.putText(
        frame,
        panel_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return frame


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("STAGE 5 - HSV OBJECT DETECTION")
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
        create_camera()
    )

    print()
    print("Camera initialized.")
    print("HSV detector initialized.")
    print()
    print("Expected objects:")
    print("  RED")
    print("  BLUE")
    print("  GREEN")
    print()
    print("Press Q in the camera window to stop.")
    print()

    frame_count = 0

    while p.isConnected():

        p.stepSimulation()

        frame = capture_camera(
            view_matrix,
            projection_matrix
        )

        detections = detect_colored_objects(
            frame
        )

        frame = draw_detections(
            frame,
            detections
        )

        frame = draw_status_panel(
            frame,
            detections
        )

        frame_count += 1

        cv2.putText(
            frame,
            f"Frame: {frame_count}",
            (IMAGE_WIDTH - 180, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow(
            "Object Detection - Top View",
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
    print("STAGE 5 COMPLETE")
    print(f"Frames processed: {frame_count}")
    print("=" * 70)


if __name__ == "__main__":
    main()
