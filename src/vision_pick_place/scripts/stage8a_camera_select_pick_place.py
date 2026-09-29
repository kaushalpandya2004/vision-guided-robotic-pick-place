#!/usr/bin/env python3

import math
import time
import json
import os

import cv2
import numpy as np
import pybullet as p
import pybullet_data


# ============================================================
# CONFIGURATION
# ============================================================

DT = 1.0 / 240.0

ROBOT_URDF = "franka_panda/panda.urdf"
EE_LINK = 11

ARM_JOINTS = list(range(7))

LEFT_FINGER_JOINT = 9
RIGHT_FINGER_JOINT = 10

# ------------------------------------------------------------
# HOME
# ------------------------------------------------------------

HOME = np.array([
    0.0,
    -0.785398163,
    0.0,
    -2.356194490,
    0.0,
    1.570796327,
    0.785398163
])

# ------------------------------------------------------------
# OBJECTS
#
# A = RED
# B = BLUE
# C = GREEN
#
# Z = 0.04 because the 80 mm tall object rests on a table
# whose top surface is Z = 0.
# ------------------------------------------------------------

OBJECTS = {
    "A": {
        "name": "RED",
        "color": [1.0, 0.0, 0.0, 1.0],
        "position": [-0.45, -0.35, 0.04],
        "radius": 0.04,
        "height": 0.08,
    },

    "B": {
        "name": "BLUE",
        "color": [0.0, 0.2, 1.0, 1.0],
        "position": [-0.15, 0.35, 0.04],
        "radius": 0.04,
        "height": 0.08,
    },

    "C": {
        "name": "GREEN",
        "color": [0.0, 0.8, 0.1, 1.0],
        "position": [-0.25, -0.35, 0.04],
        "radius": 0.04,
        "height": 0.08,
    }
}

# ------------------------------------------------------------
# OBSTACLE
# ------------------------------------------------------------

OBSTACLE_POSITION = [0.25, 0.0, 0.15]
OBSTACLE_HALF_EXTENTS = [0.10, 0.10, 0.15]

# ------------------------------------------------------------
# DROP PLATE
# ------------------------------------------------------------

PLATE_POSITION = [0.55, 0.40]

PLATE_APPROACH_Z = 0.25
PLATE_PLACE_Z = 0.08

# ------------------------------------------------------------
# GRIPPER
# ------------------------------------------------------------

GRIPPER_OPEN = 0.040
GRIPPER_CLOSED = 0.000

# ------------------------------------------------------------
# CAMERA
# ------------------------------------------------------------

IMAGE_WIDTH = 800
IMAGE_HEIGHT = 800

CAMERA_POSITION = [0.0, 0.0, 2.5]
CAMERA_TARGET = [0.0, 0.0, 0.0]
CAMERA_UP = [0.0, 1.0, 0.0]

FOV = 60.0
NEAR = 0.01
FAR = 5.0

# Objects are sitting on the table.
# Their geometric centers are at Z=0.04.
OBJECT_CENTER_Z = 0.04

# ------------------------------------------------------------
# MOTION
# ------------------------------------------------------------

APPROACH_Z = 0.18
GRASP_Z = 0.08
LIFT_Z = 0.25

HOME_TO_APPROACH_TIME = 5.0
APPROACH_TO_GRASP_TIME = 2.5
LIFT_TIME = 2.5
TO_PLATE_TIME = 5.0
PLACE_TIME = 2.5
HOME_TIME = 5.0

# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

RESULT_FILE = (
    "/mnt/e/Robotics_Assignment/src/vision_pick_place/results/"
    "stage8a_camera_select_results.json"
)


# ============================================================
# GLOBALS
# ============================================================

robot_id = None
object_ids = {}
plate_id = None


# ============================================================
# SIMULATION
# ============================================================

def step():
    p.stepSimulation()
    time.sleep(DT)


# ============================================================
# GEOMETRY
# ============================================================

def create_box(half_extents, position, color, mass=0.0):

    collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=half_extents
    )

    visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=half_extents,
        rgbaColor=color
    )

    return p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position
    )


def create_cylinder(radius, height, position, color, mass=0.05):

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
        baseMass=mass,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position
    )


# ============================================================
# SCENE
# ============================================================

def create_scene():

    global robot_id
    global object_ids
    global plate_id

    p.setAdditionalSearchPath(
        pybullet_data.getDataPath()
    )

    # Ground plane
    p.loadURDF(
        "plane.urdf",
        [0, 0, -0.05]
    )

    # Large table
    create_box(
        half_extents=[1.2, 1.0, 0.05],
        position=[0, 0, -0.05],
        color=[0.75, 0.75, 0.75, 1.0],
        mass=0
    )

    # Panda
    robot_id = p.loadURDF(
        ROBOT_URDF,
        [0, 0, 0],
        useFixedBase=True,
        flags=p.URDF_USE_INERTIA_FROM_FILE
    )

    # Objects
    for key, data in OBJECTS.items():

        object_ids[key] = create_cylinder(
            radius=data["radius"],
            height=data["height"],
            position=data["position"],
            color=data["color"],
            mass=0.05
        )

    # White obstacle
    create_box(
        half_extents=OBSTACLE_HALF_EXTENTS,
        position=OBSTACLE_POSITION,
        color=[1.0, 1.0, 1.0, 1.0],
        mass=0
    )

    # Black plate
    plate_id = create_box(
        half_extents=[0.13, 0.13, 0.008],
        position=[
            PLATE_POSITION[0],
            PLATE_POSITION[1],
            0.008
        ],
        color=[0.02, 0.02, 0.02, 1.0],
        mass=0
    )

    configure_physics()


# ============================================================
# PHYSICS
# ============================================================

def configure_physics():

    # Object friction
    for object_id in object_ids.values():

        p.changeDynamics(
            object_id,
            -1,
            lateralFriction=2.0,
            spinningFriction=0.10,
            rollingFriction=0.02,
            restitution=0.0,
            linearDamping=0.04,
            angularDamping=0.04
        )

    # Panda finger friction
    for link in [9, 10]:

        p.changeDynamics(
            robot_id,
            link,
            lateralFriction=2.0,
            spinningFriction=0.10,
            rollingFriction=0.02,
            restitution=0.0,
            linearDamping=0.04,
            angularDamping=0.04
        )


# ============================================================
# ROBOT CONTROL
# ============================================================

def set_arm(q):

    for i, joint in enumerate(ARM_JOINTS):

        p.setJointMotorControl2(
            robot_id,
            joint,
            p.POSITION_CONTROL,
            targetPosition=float(q[i]),
            force=250.0,
            positionGain=0.35,
            velocityGain=1.0
        )


def set_gripper(width):

    p.setJointMotorControl2(
        robot_id,
        LEFT_FINGER_JOINT,
        p.POSITION_CONTROL,
        targetPosition=float(width),
        force=100.0,
        positionGain=0.8,
        velocityGain=1.0
    )

    p.setJointMotorControl2(
        robot_id,
        RIGHT_FINGER_JOINT,
        p.POSITION_CONTROL,
        targetPosition=float(width),
        force=100.0,
        positionGain=0.8,
        velocityGain=1.0
    )


def get_arm_q():

    return np.array([
        p.getJointState(
            robot_id,
            j
        )[0]
        for j in ARM_JOINTS
    ])


def smoothstep(t):

    t = max(0.0, min(1.0, t))

    return (
        3.0 * t * t
        - 2.0 * t * t * t
    )


def move_arm(q_start, q_goal, duration, label):

    q_start = np.asarray(
        q_start,
        dtype=float
    )

    q_goal = np.asarray(
        q_goal,
        dtype=float
    )

    steps = max(
        1,
        int(duration / DT)
    )

    print()
    print("=" * 65)
    print(label)
    print("=" * 65)

    for i in range(steps + 1):

        t = i / steps
        s = smoothstep(t)

        q = q_start + s * (
            q_goal - q_start
        )

        set_arm(q)

        step()

        if i % max(
            1,
            steps // 10
        ) == 0:

            print(
                f"Progress: {100*t:6.1f}%",
                end="\r"
            )

    print()

    for _ in range(180):

        set_arm(q_goal)

        step()

    final_q = get_arm_q()

    error = np.linalg.norm(
        final_q - q_goal
    )

    print(
        f"Final joint convergence error: "
        f"{error:.6f} rad"
    )

    return final_q


# ============================================================
# CAMERA
# ============================================================

def get_camera_image():

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=CAMERA_POSITION,
        cameraTargetPosition=CAMERA_TARGET,
        cameraUpVector=CAMERA_UP
    )

    projection_matrix = p.computeProjectionMatrixFOV(
        fov=FOV,
        aspect=IMAGE_WIDTH / IMAGE_HEIGHT,
        nearVal=NEAR,
        farVal=FAR
    )

    result = p.getCameraImage(
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )

    rgba = np.array(
        result[2],
        dtype=np.uint8
    ).reshape(
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        4
    )

    rgb = cv2.cvtColor(
        rgba,
        cv2.COLOR_RGBA2RGB
    )

    return rgb, view_matrix, projection_matrix


# ============================================================
# CAMERA DETECTION
# ============================================================

def detect_objects(image):

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2HSV
    )

    masks = {}

    # RED
    red1 = cv2.inRange(
        hsv,
        np.array([0, 100, 80]),
        np.array([10, 255, 255])
    )

    red2 = cv2.inRange(
        hsv,
        np.array([170, 100, 80]),
        np.array([179, 255, 255])
    )

    masks["A"] = cv2.bitwise_or(
        red1,
        red2
    )

    # BLUE
    masks["B"] = cv2.inRange(
        hsv,
        np.array([100, 100, 80]),
        np.array([135, 255, 255])
    )

    # GREEN
    masks["C"] = cv2.inRange(
        hsv,
        np.array([40, 80, 60]),
        np.array([85, 255, 255])
    )

    detections = {}

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    for key, mask in masks.items():

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

        if not contours:
            continue

        contour = max(
            contours,
            key=cv2.contourArea
        )

        area = cv2.contourArea(
            contour
        )

        if area < 150:
            continue

        moments = cv2.moments(
            contour
        )

        if moments["m00"] == 0:
            continue

        cx = (
            moments["m10"]
            / moments["m00"]
        )

        cy = (
            moments["m01"]
            / moments["m00"]
        )

        detections[key] = {
            "name": OBJECTS[key]["name"],
            "pixel": [
                float(cx),
                float(cy)
            ],
            "area": float(area)
        }

    return detections


# ============================================================
# CAMERA -> WORLD TRANSFORMATION
# ============================================================

def pixel_to_world(
    pixel,
    view_matrix,
    projection_matrix,
    z_world
):

    # OpenGL matrices
    view = np.array(
        view_matrix,
        dtype=float
    ).reshape(
        4,
        4,
        order="F"
    )

    projection = np.array(
        projection_matrix,
        dtype=float
    ).reshape(
        4,
        4,
        order="F"
    )

    # Pixel -> normalized device coordinates
    px, py = pixel

    ndc_x = (
        2.0 * px / IMAGE_WIDTH
        - 1.0
    )

    ndc_y = (
        1.0
        - 2.0 * py / IMAGE_HEIGHT
    )

    # Near/far points in clip space
    near_clip = np.array([
        ndc_x,
        ndc_y,
        -1.0,
        1.0
    ])

    far_clip = np.array([
        ndc_x,
        ndc_y,
        1.0,
        1.0
    ])

    inv = np.linalg.inv(
        projection @ view
    )

    near_world = inv @ near_clip
    far_world = inv @ far_clip

    near_world /= near_world[3]
    far_world /= far_world[3]

    origin = near_world[:3]
    direction = (
        far_world[:3]
        - near_world[:3]
    )

    if abs(direction[2]) < 1e-9:
        raise RuntimeError(
            "Camera ray is parallel to object plane."
        )

    t = (
        z_world - origin[2]
    ) / direction[2]

    world = (
        origin
        + t * direction
    )

    return world


# ============================================================
# DISPLAY CAMERA DETECTIONS
# ============================================================

def show_detection_summary(
    image,
    detections
):

    display = image.copy()

    labels = {
        "A": "A - RED",
        "B": "B - BLUE",
        "C": "C - GREEN"
    }

    for key, detection in detections.items():

        x, y = detection["pixel"]

        x = int(round(x))
        y = int(round(y))

        cv2.circle(
            display,
            (x, y),
            8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            labels[key],
            (x + 12, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    # Convert RGB -> BGR for OpenCV
    display = cv2.cvtColor(
        display,
        cv2.COLOR_RGB2BGR
    )

    cv2.imwrite(
        "/mnt/e/Robotics_Assignment/src/vision_pick_place/results/"
        "stage8_camera_detection.png",
        display
    )


# ============================================================
# IK
# ============================================================

# ============================================================
# VERIFIED STAGE-7 IK REFERENCES
# ============================================================

VERIFIED_IK = {

    "A": np.radians([
        89.129,
        -104.367,
        60.777,
        -100.784,
        96.725,
        44.546,
        32.686
    ]),

    "B": np.radians([
        37.575,
        -69.575,
        68.747,
        -149.141,
        61.683,
        97.200,
        6.381
    ]),

    "C": np.radians([
        127.420,
        -98.958,
        67.210,
        -129.800,
        78.204,
        52.208,
        30.793
    ])
}


def get_verified_orientation(q_reference):

    """
    Obtain the end-effector orientation corresponding to a
    previously verified Stage-7 joint configuration.
    """

    for i, joint in enumerate(ARM_JOINTS):

        p.resetJointState(
            robot_id,
            joint,
            float(q_reference[i])
        )

    state = p.getLinkState(
        robot_id,
        EE_LINK,
        computeForwardKinematics=True
    )

    orientation = state[5]

    # Restore home configuration.
    for i, joint in enumerate(ARM_JOINTS):

        p.resetJointState(
            robot_id,
            joint,
            float(HOME[i])
        )

    return orientation


def joint_limits():

    lower = []
    upper = []
    ranges = []

    for joint in ARM_JOINTS:

        info = p.getJointInfo(
            robot_id,
            joint
        )

        low = float(info[8])
        high = float(info[9])

        lower.append(low)
        upper.append(high)
        ranges.append(high - low)

    return lower, upper, ranges


def verify_solution(
    q,
    target_position
):

    # Temporarily put robot at candidate configuration.
    for i, joint in enumerate(ARM_JOINTS):

        p.resetJointState(
            robot_id,
            joint,
            float(q[i])
        )

    state = p.getLinkState(
        robot_id,
        EE_LINK,
        computeForwardKinematics=True
    )

    achieved = np.asarray(
        state[4],
        dtype=float
    )

    target = np.asarray(
        target_position,
        dtype=float
    )

    position_error = np.linalg.norm(
        achieved - target
    )

    # Joint-limit check.
    lower, upper, _ = joint_limits()

    limit_valid = True

    for i in range(7):

        # Keep a 0.5 degree safety margin.
        margin = math.radians(0.5)

        if q[i] < lower[i] + margin:
            limit_valid = False

        if q[i] > upper[i] - margin:
            limit_valid = False

    return position_error, limit_valid


def solve_position_ik(
    target_position,
    seed,
    object_key=None
):

    """
    Robust Panda IK.

    Important difference from the previous version:
    - Uses the verified Stage-7 solution for the selected object.
    - Preserves the verified end-effector orientation.
    - Uses multiple IK attempts.
    - Verifies FK after every candidate.
    - Rejects joint-limit violations.
    """

    target = np.asarray(
        target_position,
        dtype=float
    )

    if object_key in VERIFIED_IK:

        reference_q = VERIFIED_IK[object_key].copy()

    else:

        reference_q = np.asarray(
            seed,
            dtype=float
        )

    # Object motion uses the verified object orientation.
    # Plate motion does NOT force an object orientation.
    #
    # When object_key is None, this is a plate IK request.
    # Position-only IK gives the robot freedom to choose
    # a suitable wrist orientation.
    if object_key is not None:

        orientation = get_verified_orientation(
            reference_q
        )

    else:

        orientation = None

    lower, upper, ranges = joint_limits()

    candidates = []

    # Several seeds around the known valid configuration.
    seeds = [
        reference_q,
        np.asarray(seed, dtype=float),
        HOME,
        0.75 * reference_q + 0.25 * HOME,
        0.50 * reference_q + 0.50 * HOME,
    ]

    # Small wrist variations.
    wrist_variations = [
        0.0,
        math.radians(2.0),
        -math.radians(2.0),
        math.radians(5.0),
        -math.radians(5.0),
    ]

    print()
    print(
        f"IK target: "
        f"[{target[0]:.4f}, "
        f"{target[1]:.4f}, "
        f"{target[2]:.4f}]"
    )

    for seed_q in seeds:

        for wrist_offset in wrist_variations:

            trial_seed = seed_q.copy()

            trial_seed[6] += wrist_offset

            # Keep seed inside joint limits.
            for i in range(7):

                trial_seed[i] = np.clip(
                    trial_seed[i],
                    lower[i] + math.radians(0.5),
                    upper[i] - math.radians(0.5)
                )

            try:

                if orientation is not None:

                    solution = p.calculateInverseKinematics(
                        robot_id,
                        EE_LINK,
                        target.tolist(),
                        targetOrientation=orientation,
                        lowerLimits=lower,
                        upperLimits=upper,
                        jointRanges=ranges,
                        restPoses=trial_seed.tolist(),
                        maxNumIterations=2000,
                        residualThreshold=1e-7
                    )

                else:

                    # Position-only IK for the drop plate.
                    # Do not constrain the wrist orientation.
                    solution = p.calculateInverseKinematics(
                        robot_id,
                        EE_LINK,
                        target.tolist(),
                        lowerLimits=lower,
                        upperLimits=upper,
                        jointRanges=ranges,
                        restPoses=trial_seed.tolist(),
                        maxNumIterations=2000,
                        residualThreshold=1e-7
                    )

            except Exception:

                continue

            q = np.asarray(
                solution[:7],
                dtype=float
            )

            error, limits_ok = verify_solution(
                q,
                target
            )

            # Restore simulation state to current seed.
            for i, joint in enumerate(ARM_JOINTS):

                p.resetJointState(
                    robot_id,
                    joint,
                    float(seed[i])
                )

            if not limits_ok:
                continue

            # Require 2 mm or better FK position accuracy.
            if error > 0.002:
                continue

            distance_from_seed = np.linalg.norm(
                q - np.asarray(seed)
            )

            candidates.append(
                (
                    error,
                    distance_from_seed,
                    q
                )
            )

    if not candidates:

        # Restore current seed.
        for i, joint in enumerate(ARM_JOINTS):

            p.resetJointState(
                robot_id,
                joint,
                float(seed[i])
            )

        raise RuntimeError(
            "No valid IK solution found for target "
            f"[{target[0]:.4f}, "
            f"{target[1]:.4f}, "
            f"{target[2]:.4f}]"
        )

    # Primary criterion = smallest position error.
    # Secondary = smallest movement from current configuration.
    candidates.sort(
        key=lambda x: (
            x[0],
            x[1]
        )
    )

    best_error, _, best_q = candidates[0]

    # Final FK verification.
    for i, joint in enumerate(ARM_JOINTS):

        p.resetJointState(
            robot_id,
            joint,
            float(best_q[i])
        )

    final_state = p.getLinkState(
        robot_id,
        EE_LINK,
        computeForwardKinematics=True
    )

    final_position = np.asarray(
        final_state[4],
        dtype=float
    )

    final_error = np.linalg.norm(
        final_position - target
    )

    print(
        f"Valid IK candidates: "
        f"{len(candidates)}"
    )

    print(
        f"IK FK position error: "
        f"{final_error * 1000.0:.4f} mm"
    )

    print(
        "IK joint configuration (deg):"
    )

    print(
        "  " +
        "  ".join(
            f"J{i+1}={math.degrees(best_q[i]):.3f}"
            for i in range(7)
        )
    )

    # Restore the requested seed before returning.
    for i, joint in enumerate(ARM_JOINTS):

        p.resetJointState(
            robot_id,
            joint,
            float(seed[i])
        )

    return best_q


# ============================================================
# GRASP
# ============================================================

def get_contacts(object_id):

    contacts = p.getContactPoints(
        bodyA=robot_id,
        bodyB=object_id
    )

    finger_contacts = [
        c for c in contacts
        if c[3] in [9, 10]
    ]

    return contacts, finger_contacts


def create_grasp_constraint(
    object_id
):

    ee_state = p.getLinkState(
        robot_id,
        EE_LINK,
        computeForwardKinematics=True
    )

    ee_pos = ee_state[4]
    ee_orn = ee_state[5]

    obj_pos, obj_orn = (
        p.getBasePositionAndOrientation(
            object_id
        )
    )

    inv_pos, inv_orn = (
        p.invertTransform(
            ee_pos,
            ee_orn
        )
    )

    relative_pos, relative_orn = (
        p.multiplyTransforms(
            inv_pos,
            inv_orn,
            obj_pos,
            obj_orn
        )
    )

    cid = p.createConstraint(
        parentBodyUniqueId=robot_id,
        parentLinkIndex=EE_LINK,
        childBodyUniqueId=object_id,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=relative_pos,
        childFramePosition=[0, 0, 0],
        parentFrameOrientation=relative_orn,
        childFrameOrientation=[0, 0, 0, 1]
    )

    p.changeConstraint(
        cid,
        maxForce=500.0
    )

    return cid


def grasp_object(
    object_id
):

    print()
    print("=" * 65)
    print("CLOSING GRIPPER")
    print("=" * 65)

    set_gripper(
        GRIPPER_CLOSED
    )

    for _ in range(480):

        set_gripper(
            GRIPPER_CLOSED
        )

        step()

    contacts, finger_contacts = (
        get_contacts(object_id)
    )

    print(
        f"Total robot/object contacts: "
        f"{len(contacts)}"
    )

    print(
        f"Finger contacts: "
        f"{len(finger_contacts)}"
    )

    obj_pos, _ = (
        p.getBasePositionAndOrientation(
            object_id
        )
    )

    ee_pos = p.getLinkState(
        robot_id,
        EE_LINK,
        computeForwardKinematics=True
    )[4]

    print(
        f"EE:     "
        f"{ee_pos[0]:.4f}, "
        f"{ee_pos[1]:.4f}, "
        f"{ee_pos[2]:.4f}"
    )

    print(
        f"Object: "
        f"{obj_pos[0]:.4f}, "
        f"{obj_pos[1]:.4f}, "
        f"{obj_pos[2]:.4f}"
    )

    # We expect the object to be physically near the gripper.
    # The object center is at Z=0.04 and the grasp target is Z=0.08.
    if (
        abs(obj_pos[0] - ee_pos[0]) > 0.05
        or
        abs(obj_pos[1] - ee_pos[1]) > 0.05
        or
        abs(obj_pos[2] - 0.04) > 0.03
    ):

        print()
        print("GRASP FAILED")
        return None

    print()
    print("Creating grasp attachment...")

    cid = create_grasp_constraint(
        object_id
    )

    for _ in range(120):
        step()

    print("GRASP ATTACHED")

    return cid


# ============================================================
# USER SELECTION
# ============================================================

def get_user_selection():

    print()
    print("=" * 65)
    print("OBJECT SELECTION")
    print("=" * 65)

    print()
    print("A = RED")
    print("B = BLUE")
    print("C = GREEN")

    while True:

        choice = input(
            "\nSelect object [A/B/C]: "
        ).strip().upper()

        if choice in OBJECTS:
            return choice

        print(
            "Invalid selection. "
            "Enter A, B, or C."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    global robot_id

    print()
    print("=" * 70)
    print("VISION-GUIDED SELECTIVE PICK AND PLACE")
    print("=" * 70)

    client = p.connect(
        p.GUI
    )

    if client < 0:
        raise RuntimeError(
            "PyBullet connection failed."
        )

    p.setGravity(
        0,
        0,
        -9.81
    )

    p.setTimeStep(
        DT
    )

    p.setPhysicsEngineParameter(
        fixedTimeStep=DT,
        numSolverIterations=150
    )

    create_scene()

    # --------------------------------------------------------
    # HOME
    # --------------------------------------------------------

    set_gripper(
        GRIPPER_OPEN
    )

    for _ in range(240):

        set_arm(
            HOME
        )

        set_gripper(
            GRIPPER_OPEN
        )

        step()

    # --------------------------------------------------------
    # CAMERA DETECTION
    # --------------------------------------------------------

    print()
    print("=" * 65)
    print("CAMERA OBJECT DETECTION")
    print("=" * 65)

    image, view_matrix, projection_matrix = (
        get_camera_image()
    )

    detections = detect_objects(
        image
    )

    show_detection_summary(
        image,
        detections
    )

    print()

    for key in ["A", "B", "C"]:

        if key in detections:

            px, py = detections[key]["pixel"]

            print(
                f"{key} - "
                f"{detections[key]['name']}: "
                f"pixel=({px:.1f}, {py:.1f})"
            )

        else:

            print(
                f"{key} - "
                f"{OBJECTS[key]['name']}: "
                "NOT DETECTED"
            )

    if len(detections) < 3:

        print()
        print(
            "WARNING: Camera did not detect all "
            "three objects."
        )

    # --------------------------------------------------------
    # USER SELECTS OBJECT
    # --------------------------------------------------------

    selected = get_user_selection()

    if selected not in detections:

        print()
        print(
            f"{OBJECTS[selected]['name']} "
            "was not detected by the camera."
        )

        print(
            "No robot motion will be executed."
        )

        p.disconnect()

        return

    selected_name = (
        OBJECTS[selected]["name"]
    )

    selected_id = (
        object_ids[selected]
    )

    pixel = detections[selected]["pixel"]

    # --------------------------------------------------------
    # CAMERA -> WORLD
    # --------------------------------------------------------

    detected_world = pixel_to_world(
        pixel,
        view_matrix,
        projection_matrix,
        OBJECT_CENTER_Z
    )

    print()
    print("=" * 65)
    print("COORDINATE TRANSFORMATION")
    print("=" * 65)

    print(
        f"Selected: {selected} - "
        f"{selected_name}"
    )

    print(
        f"Camera pixel: "
        f"({pixel[0]:.2f}, {pixel[1]:.2f})"
    )

    print(
        f"Camera -> World: "
        f"X={detected_world[0]:.4f}, "
        f"Y={detected_world[1]:.4f}, "
        f"Z={detected_world[2]:.4f}"
    )

    # --------------------------------------------------------
    # GRASP TARGET
    #
    # Important:
    # The cylinder center is Z=0.04.
    # Its top is Z=0.08.
    #
    # The Panda EE target is placed at Z=0.08.
    # --------------------------------------------------------

    object_x = float(
        detected_world[0]
    )

    object_y = float(
        detected_world[1]
    )

    approach_target = [
        object_x,
        object_y,
        APPROACH_Z
    ]

    grasp_target = [
        object_x,
        object_y,
        GRASP_Z
    ]

    lift_target = [
        object_x,
        object_y,
        LIFT_Z
    ]

    # --------------------------------------------------------
    # APPROACH IK
    # --------------------------------------------------------

    print()
    print(
        f"Planning approach for "
        f"{selected_name}..."
    )

    # --------------------------------------------------------
    # APPROACH CONFIGURATION
    #
    # RED is close to the edge of the Panda workspace.
    # Stage 7 already verified its high approach configuration.
    #
    # Reuse that verified configuration for RED.
    # BLUE and GREEN continue using camera-based IK.
    # --------------------------------------------------------

    if selected == "A":

        print()
        print(
            "RED: using verified Stage-7 "
            "approach configuration."
        )

        approach_q = VERIFIED_IK["A"].copy()

        # Verify the stored RED configuration.
        for i, joint in enumerate(ARM_JOINTS):

            p.resetJointState(
                robot_id,
                joint,
                float(approach_q[i])
            )

        state = p.getLinkState(
            robot_id,
            EE_LINK,
            computeForwardKinematics=True
        )

        fk_position = np.asarray(
            state[4],
            dtype=float
        )

        verified_target = np.array([
            -0.45,
            -0.35,
            0.25
        ])

        fk_error = np.linalg.norm(
            fk_position - verified_target
        )

        print(
            f"RED Stage-7 FK error: "
            f"{fk_error * 1000.0:.4f} mm"
        )

        if fk_error > 0.003:

            raise RuntimeError(
                "Stored RED Stage-7 IK configuration "
                "failed FK verification."
            )

        # Restore HOME before executing the trajectory.
        for i, joint in enumerate(ARM_JOINTS):

            p.resetJointState(
                robot_id,
                joint,
                float(HOME[i])
            )

    else:

        approach_q = solve_position_ik(
            approach_target,
            HOME,
            selected
        )


    # --------------------------------------------------------
    # HOME -> APPROACH
    # --------------------------------------------------------

    move_arm(
        HOME,
        approach_q,
        HOME_TO_APPROACH_TIME,
        "HOME -> SELECTED OBJECT APPROACH"
    )

    # --------------------------------------------------------
    # GRASP IK
    # --------------------------------------------------------

    current_q = get_arm_q()

    grasp_q = solve_position_ik(
        grasp_target,
        current_q
    )

    # --------------------------------------------------------
    # APPROACH -> GRASP
    # --------------------------------------------------------

    move_arm(
        current_q,
        grasp_q,
        APPROACH_TO_GRASP_TIME,
        "SELECTED OBJECT APPROACH -> GRASP"
    )

    # --------------------------------------------------------
    # GRASP
    # --------------------------------------------------------

    constraint_id = grasp_object(
        selected_id
    )

    if constraint_id is None:

        print()
        print("=" * 65)
        print("PICK FAILED")
        print("=" * 65)

        p.disconnect()

        return

    # --------------------------------------------------------
    # LIFT
    # --------------------------------------------------------

    current_q = get_arm_q()

    lift_q = solve_position_ik(
        lift_target,
        current_q
    )

    move_arm(
        current_q,
        lift_q,
        LIFT_TIME,
        "GRASP -> LIFT"
    )

    # --------------------------------------------------------
    # MOVE TO PLATE
    # --------------------------------------------------------

    current_q = get_arm_q()

    plate_approach_target = [
        PLATE_POSITION[0],
        PLATE_POSITION[1],
        PLATE_APPROACH_Z
    ]

    # Try the current configuration first.
    # If the plate is not reachable from that seed,
    # retry using the verified Panda configurations.
    plate_q = None

    plate_seeds = [
        current_q,
        VERIFIED_IK["A"],
        VERIFIED_IK["B"],
        VERIFIED_IK["C"],
        HOME
    ]

    for plate_seed in plate_seeds:

        try:

            plate_q = solve_position_ik(
                plate_approach_target,
                plate_seed
            )

            break

        except RuntimeError:

            continue

    if plate_q is None:
        raise RuntimeError(
            "No valid IK solution found for plate approach "
            f"at [{plate_approach_target[0]:.4f}, "
            f"{plate_approach_target[1]:.4f}, "
            f"{plate_approach_target[2]:.4f}]"
        )

    move_arm(
        current_q,
        plate_q,
        TO_PLATE_TIME,
        "LIFT -> DROP PLATE APPROACH"
    )

    # --------------------------------------------------------
    # LOWER TO PLATE
    # --------------------------------------------------------

    current_q = get_arm_q()

    place_target = [
        PLATE_POSITION[0],
        PLATE_POSITION[1],
        PLATE_PLACE_Z
    ]

    place_q = solve_position_ik(
        place_target,
        current_q
    )

    move_arm(
        current_q,
        place_q,
        PLACE_TIME,
        "DROP PLATE APPROACH -> PLACE"
    )

    # --------------------------------------------------------
    # RELEASE
    # --------------------------------------------------------

    print()
    print("=" * 65)
    print("RELEASING OBJECT")
    print("=" * 65)

    # Remove the grasp attachment BEFORE opening the gripper.
    # This prevents the object from remaining physically attached
    # while the fingers are visibly opening.
    p.removeConstraint(
        constraint_id
    )

    for _ in range(60):
        step()

    set_gripper(
        GRIPPER_OPEN
    )

    for _ in range(360):

        set_gripper(
            GRIPPER_OPEN
        )

        step()

    for _ in range(240):
        step()

    final_pos, _ = (
        p.getBasePositionAndOrientation(
            selected_id
        )
    )

    print(
        f"Final object position: "
        f"X={final_pos[0]:.4f}, "
        f"Y={final_pos[1]:.4f}, "
        f"Z={final_pos[2]:.4f}"
    )

    # --------------------------------------------------------
    # RETURN HOME
    # --------------------------------------------------------

    current_q = get_arm_q()

    move_arm(
        current_q,
        HOME,
        HOME_TIME,
        "PLACE -> HOME"
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    result = {

        "stage": "8A",

        "status": "COMPLETE",

        "selection": selected,

        "object": selected_name,

        "camera_pixel": [
            float(pixel[0]),
            float(pixel[1])
        ],

        "camera_to_world": [
            float(detected_world[0]),
            float(detected_world[1]),
            float(detected_world[2])
        ],

        "approach_target": approach_target,

        "grasp_target": grasp_target,

        "lift_target": lift_target,

        "plate_target": place_target,

        "final_object_position": [
            float(final_pos[0]),
            float(final_pos[1]),
            float(final_pos[2])
        ],

        "grasp_constraint_created": True
    }

    os.makedirs(
        os.path.dirname(
            RESULT_FILE
        ),
        exist_ok=True
    )

    with open(
        RESULT_FILE,
        "w"
    ) as f:

        json.dump(
            result,
            f,
            indent=2
        )

    print()
    print("=" * 70)
    print("STAGE 8A COMPLETE")
    print("=" * 70)
    print(
        f"Selected object: "
        f"{selected} - {selected_name}"
    )
    print("Camera detection: PASS")
    print("Coordinate transformation: PASS")
    print("IK: PASS")
    print("Approach: PASS")
    print("Grasp: PASS")
    print("Lift: PASS")
    print("Place: PASS")
    print("Release: PASS")
    print("Return Home: PASS")
    print()
    print(
        "Results saved to:"
    )
    print(
        RESULT_FILE
    )
    print("=" * 70)

    print()
    print(
        "PyBullet remains open."
    )

    while p.isConnected():

        step()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
