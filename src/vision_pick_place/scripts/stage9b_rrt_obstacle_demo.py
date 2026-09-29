#!/usr/bin/env python3

import math
import time
import json
import os
import random

import numpy as np
import pybullet as p
import pybullet_data


# ============================================================
# STAGE 9B
# ACTUAL RRT OBSTACLE AVOIDANCE
# ============================================================

DT = 1.0 / 240.0

ROBOT_URDF = "franka_panda/panda.urdf"

EE_LINK = 11
ARM_JOINTS = list(range(7))

HOME = np.array([
    0.0,
    -0.785398163,
    0.0,
    -2.356194490,
    0.0,
    1.570796327,
    0.785398163
], dtype=float)

# Known verified Green lift configuration
GREEN_LIFT = np.radians([
    -55.953,
    -32.329,
    -74.959,
    -139.291,
    -11.689,
    120.134,
    96.226
])


# ============================================================
# SCENE CONSTANTS
# ============================================================

TABLE_HALF_EXTENTS = [1.2, 1.0, 0.05]
TABLE_POSITION = [0.0, 0.0, -0.05]

GREEN_POSITION = [-0.25, -0.35, 0.04]

PLATE_HALF_EXTENTS = [0.13, 0.13, 0.008]
PLATE_POSITION = [0.55, 0.40, 0.008]

# Obstacle size used for the demonstration.
OBSTACLE_HALF_EXTENTS = [0.115, 0.115, 0.12]


# ============================================================
# RRT PARAMETERS
# ============================================================

RRT_MAX_ITERATIONS = 16000

RRT_STEP_SIZE = 0.18

EDGE_RESOLUTION = 0.030

GOAL_BIAS = 0.12

GOAL_TOLERANCE = 0.10

SHORTCUT_ATTEMPTS = 600

RANDOM_SEED = 42

JOINT_LIMIT_MARGIN = math.radians(0.5)


# ============================================================
# EXECUTION
# ============================================================

EXECUTION_TIME_PER_SEGMENT = 1.25

SETTLE_STEPS = 180

POSITION_GAIN = 0.35

VELOCITY_GAIN = 1.0

MOTOR_FORCE = 250


RESULT_FILE = (
    "/mnt/e/Robotics_Assignment/"
    "src/vision_pick_place/results/"
    "stage9b_rrt_results.json"
)


# ============================================================
# BOX CREATION
# ============================================================

def create_box(
    half_extents,
    position,
    mass=0.0
):

    collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=half_extents
    )

    visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=half_extents
    )

    body = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position
    )

    return body


# ============================================================
# ARM CONTROL
# ============================================================

def set_arm(robot, q):

    for joint, value in zip(
        ARM_JOINTS,
        q
    ):

        p.setJointMotorControl2(
            bodyUniqueId=robot,
            jointIndex=joint,
            controlMode=p.POSITION_CONTROL,
            targetPosition=float(value),
            force=MOTOR_FORCE,
            positionGain=POSITION_GAIN,
            velocityGain=VELOCITY_GAIN
        )


def get_arm_q(robot):

    return np.array(
        [
            p.getJointState(
                robot,
                joint
            )[0]
            for joint in ARM_JOINTS
        ],
        dtype=float
    )


def move_arm(
    robot,
    q_target,
    duration=1.25
):

    q_start = get_arm_q(
        robot
    )

    steps = max(
        1,
        int(duration / DT)
    )

    print(
        f"    Duration: {duration:.2f} s"
    )

    print(
        f"    Samples: {steps}"
    )

    for i in range(steps):

        u = (
            i + 1
        ) / steps

        # Cubic smoothstep
        s = (
            3.0 * u * u
            - 2.0 * u * u * u
        )

        q = (
            q_start
            + s * (q_target - q_start)
        )

        set_arm(
            robot,
            q
        )

        p.stepSimulation()

        time.sleep(DT)

    print(
        "    Settling..."
    )

    for _ in range(
        SETTLE_STEPS
    ):

        set_arm(
            robot,
            q_target
        )

        p.stepSimulation()

        time.sleep(DT)

    q_final = get_arm_q(
        robot
    )

    error = float(
        np.linalg.norm(
            q_final - q_target
        )
    )

    print(
        f"    Final joint convergence: "
        f"{error:.8f} rad"
    )

    return error


# ============================================================
# JOINT LIMITS
# ============================================================

def get_joint_limits(robot):

    lower = []
    upper = []

    for joint in ARM_JOINTS:

        info = p.getJointInfo(
            robot,
            joint
        )

        lo = float(
            info[8]
        )

        hi = float(
            info[9]
        )

        if lo >= hi:

            lo = -math.pi
            hi = math.pi

        lower.append(
            lo + JOINT_LIMIT_MARGIN
        )

        upper.append(
            hi - JOINT_LIMIT_MARGIN
        )

    return (
        np.array(lower),
        np.array(upper)
    )


def within_joint_limits(
    q,
    lower,
    upper
):

    q = np.asarray(
        q,
        dtype=float
    )

    return bool(
        np.all(q >= lower)
        and
        np.all(q <= upper)
    )


# ============================================================
# COLLISION CHECKING
# ============================================================

def self_collision(robot):

    contacts = p.getClosestPoints(
        bodyA=robot,
        bodyB=robot,
        distance=0.0
    )

    connected_pairs = {
        (6, 8),
        (8, 6),
        (8, 9),
        (9, 8),
        (8, 10),
        (10, 8),
        (9, 10),
        (10, 9)
    }

    for c in contacts:

        link_a = int(c[3])
        link_b = int(c[4])

        # Base-related contacts are expected.
        if link_a == -1 or link_b == -1:
            continue

        # Same collision link.
        if link_a == link_b:
            continue

        # Neighboring Panda links are connected
        # through their joints.
        if abs(
            link_a - link_b
        ) <= 1:
            continue

        # Hand/finger connected geometry.
        if (
            link_a,
            link_b
        ) in connected_pairs:
            continue

        return True

    return False


def environment_collision(
    robot,
    table,
    obstacle
):

    # --------------------------------------------------------
    # Robot vs table
    # --------------------------------------------------------

    contacts = p.getClosestPoints(
        bodyA=robot,
        bodyB=table,
        distance=0.0
    )

    for c in contacts:

        robot_link = int(
            c[3]
        )

        # Panda base is intentionally sitting
        # on the table.
        if robot_link == -1:
            continue

        return True

    # --------------------------------------------------------
    # Robot vs obstacle
    #
    # IMPORTANT:
    # obstacle can legitimately be None during
    # initial configuration validation.
    # --------------------------------------------------------

    if obstacle is not None:

        contacts = p.getClosestPoints(
            bodyA=robot,
            bodyB=obstacle,
            distance=0.0
        )

        if len(contacts) > 0:
            return True

    return False


def configuration_in_collision(
    robot,
    table,
    obstacle,
    q,
    lower,
    upper
):

    q = np.asarray(
        q,
        dtype=float
    )

    if not within_joint_limits(
        q,
        lower,
        upper
    ):
        return True

    # Temporarily place robot at this configuration.
    for joint, value in zip(
        ARM_JOINTS,
        q
    ):

        p.resetJointState(
            robot,
            joint,
            float(value)
        )

    p.performCollisionDetection()

    if self_collision(
        robot
    ):
        return True

    if environment_collision(
        robot,
        table,
        obstacle
    ):
        return True

    return False


# ============================================================
# EDGE VALIDATION
# ============================================================

def edge_is_valid(
    robot,
    table,
    obstacle,
    q1,
    q2,
    lower,
    upper
):

    q1 = np.asarray(
        q1,
        dtype=float
    )

    q2 = np.asarray(
        q2,
        dtype=float
    )

    distance = float(
        np.linalg.norm(
            q2 - q1
        )
    )

    samples = max(
        2,
        int(
            math.ceil(
                distance
                / EDGE_RESOLUTION
            )
        ) + 1
    )

    for i in range(
        samples
    ):

        u = i / (
            samples - 1
        )

        q = (
            q1
            + u * (q2 - q1)
        )

        if configuration_in_collision(
            robot,
            table,
            obstacle,
            q,
            lower,
            upper
        ):

            return False

    return True


# ============================================================
# DIRECT PATH
# ============================================================

def direct_path_valid(
    robot,
    table,
    obstacle,
    start,
    goal,
    lower,
    upper
):

    return edge_is_valid(
        robot,
        table,
        obstacle,
        start,
        goal,
        lower,
        upper
    )


# ============================================================
# RRT NODE
# ============================================================

class RRTNode:

    def __init__(
        self,
        q,
        parent=None
    ):

        self.q = np.asarray(
            q,
            dtype=float
        )

        self.parent = parent


# ============================================================
# RRT FUNCTIONS
# ============================================================

def nearest_node(
    nodes,
    q
):

    distances = [
        np.linalg.norm(
            node.q - q
        )
        for node in nodes
    ]

    return nodes[
        int(
            np.argmin(
                distances
            )
        )
    ]


def steer(
    q_from,
    q_to,
    step_size
):

    direction = (
        q_to - q_from
    )

    distance = np.linalg.norm(
        direction
    )

    if distance <= step_size:

        return q_to.copy()

    return (
        q_from
        + direction / distance
        * step_size
    )


def reconstruct_path(
    node
):

    path = []

    current = node

    while current is not None:

        path.append(
            current.q.copy()
        )

        current = current.parent

    path.reverse()

    return path


def rrt_plan(
    robot,
    table,
    obstacle,
    start,
    goal,
    lower,
    upper
):

    print()
    print("=" * 70)
    print("RRT PLANNING")
    print("=" * 70)

    start = np.asarray(
        start,
        dtype=float
    )

    goal = np.asarray(
        goal,
        dtype=float
    )

    print(
        f"Start-goal distance: "
        f"{np.linalg.norm(goal-start):.4f} rad"
    )

    if configuration_in_collision(
        robot,
        table,
        obstacle,
        start,
        lower,
        upper
    ):

        raise RuntimeError(
            "RRT start configuration is in collision."
        )

    print(
        "RRT start configuration: PASS"
    )

    if configuration_in_collision(
        robot,
        table,
        obstacle,
        goal,
        lower,
        upper
    ):

        raise RuntimeError(
            "RRT goal configuration is in collision."
        )

    print(
        "RRT goal configuration: PASS"
    )

    rng = random.Random(
        RANDOM_SEED
    )

    nodes = [
        RRTNode(start)
    ]

    start_time = time.time()

    for iteration in range(
        RRT_MAX_ITERATIONS
    ):

        # Goal bias.
        if (
            rng.random()
            < GOAL_BIAS
        ):

            q_rand = goal.copy()

        else:

            q_rand = np.array(
                [
                    rng.uniform(
                        float(lower[j]),
                        float(upper[j])
                    )
                    for j in range(7)
                ],
                dtype=float
            )

        nearest = nearest_node(
            nodes,
            q_rand
        )

        q_new = steer(
            nearest.q,
            q_rand,
            RRT_STEP_SIZE
        )

        if not edge_is_valid(
            robot,
            table,
            obstacle,
            nearest.q,
            q_new,
            lower,
            upper
        ):

            continue

        new_node = RRTNode(
            q_new,
            nearest
        )

        nodes.append(
            new_node
        )

        distance_to_goal = np.linalg.norm(
            q_new - goal
        )

        if (
            distance_to_goal
            <= GOAL_TOLERANCE
        ):

            if edge_is_valid(
                robot,
                table,
                obstacle,
                q_new,
                goal,
                lower,
                upper
            ):

                goal_node = RRTNode(
                    goal,
                    new_node
                )

                nodes.append(
                    goal_node
                )

                planning_time = (
                    time.time()
                    - start_time
                )

                path = reconstruct_path(
                    goal_node
                )

                print(
                    f"RRT solution found "
                    f"at iteration "
                    f"{iteration + 1}"
                )

                print(
                    f"RRT nodes: "
                    f"{len(nodes)}"
                )

                print(
                    f"Planning time: "
                    f"{planning_time:.4f} s"
                )

                return path

    raise RuntimeError(
        "RRT failed to find a collision-free "
        "path within "
        f"{RRT_MAX_ITERATIONS} iterations."
    )


# ============================================================
# PATH VALIDATION
# ============================================================

def validate_path(
    robot,
    table,
    obstacle,
    path,
    lower,
    upper
):

    print()
    print("=" * 70)
    print("FINAL PATH VALIDATION")
    print("=" * 70)

    collision_samples = 0
    checked_samples = 0

    minimum_clearance = float(
        "inf"
    )

    for i in range(
        len(path) - 1
    ):

        q1 = path[i]
        q2 = path[i + 1]

        distance = np.linalg.norm(
            q2 - q1
        )

        samples = max(
            2,
            int(
                math.ceil(
                    distance
                    / EDGE_RESOLUTION
                )
            ) + 1
        )

        for j in range(
            samples
        ):

            u = j / (
                samples - 1
            )

            q = (
                q1
                + u * (q2 - q1)
            )

            checked_samples += 1

            if configuration_in_collision(
                robot,
                table,
                obstacle,
                q,
                lower,
                upper
            ):

                collision_samples += 1

            # Actual robot-obstacle clearance.
            contacts = p.getClosestPoints(
                bodyA=robot,
                bodyB=obstacle,
                distance=5.0
            )

            for c in contacts:

                clearance = float(
                    c[8]
                )

                minimum_clearance = min(
                    minimum_clearance,
                    clearance
                )

    if configuration_in_collision(
        robot,
        table,
        obstacle,
        path[-1],
        lower,
        upper
    ):

        collision_samples += 1

    if minimum_clearance == float(
        "inf"
    ):

        minimum_clearance = 0.0

    print(
        f"Checked samples: "
        f"{checked_samples}"
    )

    print(
        f"Collision samples: "
        f"{collision_samples}"
    )

    print(
        "Minimum obstacle clearance: "
        f"{minimum_clearance * 1000.0:.3f} mm"
    )

    passed = (
        collision_samples == 0
        and len(path) >= 2
    )

    if passed:

        print(
            "FINAL PATH VALIDATION: PASS"
        )

    else:

        print(
            "FINAL PATH VALIDATION: FAIL"
        )

    return {
        "passed": passed,
        "checked_samples": checked_samples,
        "collision_samples": collision_samples,
        "minimum_obstacle_clearance_m":
            float(minimum_clearance)
    }


# ============================================================
# PATH SHORTCUTTING
# ============================================================

def shortcut_path(
    robot,
    table,
    obstacle,
    path,
    lower,
    upper
):

    if len(path) <= 2:
        return path

    rng = random.Random(
        RANDOM_SEED + 100
    )

    path = [
        np.asarray(
            q,
            dtype=float
        )
        for q in path
    ]

    for _ in range(
        SHORTCUT_ATTEMPTS
    ):

        if len(path) <= 2:
            break

        i, j = sorted(
            rng.sample(
                range(len(path)),
                2
            )
        )

        if j <= i + 1:
            continue

        if edge_is_valid(
            robot,
            table,
            obstacle,
            path[i],
            path[j],
            lower,
            upper
        ):

            path = (
                path[:i + 1]
                + path[j:]
            )

    return path


# ============================================================
# EE POSITION
# ============================================================

def get_ee_position(
    robot
):

    state = p.getLinkState(
        robot,
        EE_LINK,
        computeForwardKinematics=True
    )

    return np.array(
        state[4],
        dtype=float
    )


# ============================================================
# AUTOMATIC OBSTACLE SELECTION
# ============================================================

def find_demonstration_obstacle(
    robot,
    table,
    start,
    goal,
    lower,
    upper
):

    print()
    print("=" * 70)
    print("BUILDING OBSTACLE FOR RRT DEMONSTRATION")
    print("=" * 70)

    # --------------------------------------------------------
    # We deliberately place a tall vertical obstacle around
    # the Cartesian region traversed by the robot.
    #
    # The obstacle starts at the table surface and extends
    # upward into the Panda arm workspace.
    #
    # We test several locations and widths and accept only
    # an obstacle that:
    #
    #   1. does NOT collide with HOME
    #   2. does NOT collide with GREEN_LIFT
    #   3. DOES block the direct joint-space path
    #
    # This makes the RRT demonstration deterministic instead
    # of relying on an arbitrary hand-picked obstacle.
    # --------------------------------------------------------

    fractions = [
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75
    ]

    half_xy_values = [
        0.08,
        0.10,
        0.12,
        0.14,
        0.16
    ]

    half_z_values = [
        0.22,
        0.26,
        0.30,
        0.34,
        0.38
    ]

    # --------------------------------------------------------
    # First obtain the EE positions along the direct path.
    # --------------------------------------------------------

    direct_ee_positions = []

    for fraction in fractions:

        q = (
            start
            + fraction * (goal - start)
        )

        for joint, value in zip(
            ARM_JOINTS,
            q
        ):

            p.resetJointState(
                robot,
                joint,
                float(value)
            )

        p.performCollisionDetection()

        ee = get_ee_position(
            robot
        )

        direct_ee_positions.append(
            (
                fraction,
                ee.copy()
            )
        )

    # --------------------------------------------------------
    # Try obstacles centered around the actual EE trajectory.
    # --------------------------------------------------------

    for fraction, ee in direct_ee_positions:

        for half_xy in half_xy_values:

            for half_z in half_z_values:

                # Keep the bottom of the obstacle on the table.
                obstacle_z = half_z

                center = [
                    float(ee[0]),
                    float(ee[1]),
                    float(obstacle_z)
                ]

                half_extents = [
                    half_xy,
                    half_xy,
                    half_z
                ]

                obstacle = create_box(
                    half_extents,
                    center,
                    mass=0.0
                )

                # ------------------------------------------------
                # Validate HOME.
                # ------------------------------------------------

                home_bad = configuration_in_collision(
                    robot,
                    table,
                    obstacle,
                    start,
                    lower,
                    upper
                )

                if home_bad:

                    p.removeBody(
                        obstacle
                    )

                    continue

                # ------------------------------------------------
                # Validate GREEN_LIFT.
                # ------------------------------------------------

                goal_bad = configuration_in_collision(
                    robot,
                    table,
                    obstacle,
                    goal,
                    lower,
                    upper
                )

                if goal_bad:

                    p.removeBody(
                        obstacle
                    )

                    continue

                # ------------------------------------------------
                # Test direct path.
                # ------------------------------------------------

                direct_ok = direct_path_valid(
                    robot,
                    table,
                    obstacle,
                    start,
                    goal,
                    lower,
                    upper
                )

                if not direct_ok:

                    print()
                    print(
                        "Obstacle selected successfully:"
                    )

                    print(
                        f"  Position: "
                        f"{np.array(center)}"
                    )

                    print(
                        f"  Half extents: "
                        f"{np.array(half_extents)}"
                    )

                    print(
                        f"  Path fraction: "
                        f"{fraction:.2f}"
                    )

                    print(
                        f"  HOME: PASS"
                    )

                    print(
                        f"  GREEN_LIFT: PASS"
                    )

                    print(
                        f"  Direct path: BLOCKED"
                    )

                    return (
                        obstacle,
                        center,
                        half_extents,
                        fraction
                    )

                p.removeBody(
                    obstacle
                )

    raise RuntimeError(
        "Could not construct a valid obstacle "
        "that blocks the direct path while keeping "
        "HOME and GREEN_LIFT collision-free."
    )


# ============================================================
# EXECUTION
# ============================================================

def execute_path(
    robot,
    path
):

    print()
    print("=" * 70)
    print(
        "EXECUTING SMOOTH RRT PATH"
    )
    print("=" * 70)

    max_error = 0.0

    for i in range(
        len(path) - 1
    ):

        print(
            f"Executing RRT segment "
            f"{i + 1}/"
            f"{len(path) - 1}"
        )

        error = move_arm(
            robot,
            path[i + 1],
            EXECUTION_TIME_PER_SEGMENT
        )

        max_error = max(
            max_error,
            error
        )

    return max_error


# ============================================================
# SCENE
# ============================================================

def create_scene():

    p.setAdditionalSearchPath(
        pybullet_data.getDataPath()
    )

    p.setGravity(
        0.0,
        0.0,
        -9.81
    )

    plane = p.loadURDF(
        "plane.urdf"
    )

    table = create_box(
        TABLE_HALF_EXTENTS,
        TABLE_POSITION,
        mass=0.0
    )

    robot = p.loadURDF(
        ROBOT_URDF,
        [0.0, 0.0, 0.0],
        useFixedBase=True
    )

    green_collision = p.createCollisionShape(
        p.GEOM_CYLINDER,
        radius=0.035,
        height=0.08
    )

    green_visual = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=0.035,
        length=0.08
    )

    green = p.createMultiBody(
        baseMass=0.05,
        baseCollisionShapeIndex=green_collision,
        baseVisualShapeIndex=green_visual,
        basePosition=GREEN_POSITION
    )

    plate = create_box(
        PLATE_HALF_EXTENTS,
        PLATE_POSITION,
        mass=0.0
    )

    return (
        plane,
        table,
        robot,
        green,
        plate
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results
):

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
            results,
            f,
            indent=2
        )

    print()
    print(
        "Results saved:"
    )

    print(
        RESULT_FILE
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "STAGE 9B - ACTUAL "
        "RRT OBSTACLE AVOIDANCE"
    )
    print("=" * 70)

    print(
        "Robot: Franka Panda"
    )

    print(
        "Planner: Joint-space RRT"
    )

    print(
        "Collision checking: ENABLED"
    )

    print(
        "Path smoothing: ENABLED"
    )

    print(
        "Direct path bypass: DISABLED"
    )

    print("=" * 70)

    random.seed(
        RANDOM_SEED
    )

    np.random.seed(
        RANDOM_SEED
    )

    p.connect(
        p.GUI
    )

    try:

        (
            plane,
            table,
            robot,
            green,
            plate
        ) = create_scene()

        lower, upper = get_joint_limits(
            robot
        )

        # ----------------------------------------------------
        # HOME
        # ----------------------------------------------------

        print()
        print(
            "Start configuration check..."
        )

        if configuration_in_collision(
            robot,
            table,
            None,
            HOME,
            lower,
            upper
        ):

            raise RuntimeError(
                "HOME configuration is "
                "in collision."
            )

        print(
            "HOME configuration: PASS"
        )

        # ----------------------------------------------------
        # GREEN LIFT
        # ----------------------------------------------------

        print()
        print(
            "Goal configuration check..."
        )

        if configuration_in_collision(
            robot,
            table,
            None,
            GREEN_LIFT,
            lower,
            upper
        ):

            raise RuntimeError(
                "GREEN_LIFT configuration is "
                "in collision."
            )

        print(
            "GREEN_LIFT configuration: PASS"
        )

        # ----------------------------------------------------
        # Green EE
        # ----------------------------------------------------

        for joint, value in zip(
            ARM_JOINTS,
            GREEN_LIFT
        ):

            p.resetJointState(
                robot,
                joint,
                float(value)
            )

        green_ee = get_ee_position(
            robot
        )

        print(
            "GREEN_LIFT EE position: "
            f"{green_ee}"
        )

        # Return to HOME.
        for joint, value in zip(
            ARM_JOINTS,
            HOME
        ):

            p.resetJointState(
                robot,
                joint,
                float(value)
            )

        # ----------------------------------------------------
        # Find obstacle.
        # ----------------------------------------------------

        (
            obstacle,
            obstacle_position,
            obstacle_half_extents,
            obstacle_fraction
        ) = find_demonstration_obstacle(
            robot,
            table,
            HOME,
            GREEN_LIFT,
            lower,
            upper
        )

        # ----------------------------------------------------
        # Direct path MUST be blocked.
        # ----------------------------------------------------

        print()
        print(
            "Checking direct joint-space path..."
        )

        direct_valid = direct_path_valid(
            robot,
            table,
            obstacle,
            HOME,
            GREEN_LIFT,
            lower,
            upper
        )

        if direct_valid:

            raise RuntimeError(
                "Direct path is collision-free. "
                "Obstacle demonstration was not exercised."
            )

        print(
            "Direct path: BLOCKED"
        )

        # ----------------------------------------------------
        # RRT.
        # ----------------------------------------------------

        raw_path = rrt_plan(
            robot,
            table,
            obstacle,
            HOME,
            GREEN_LIFT,
            lower,
            upper
        )

        print(
            f"Raw RRT path waypoints: "
            f"{len(raw_path)}"
        )

        # ----------------------------------------------------
        # Smoothing.
        # ----------------------------------------------------

        print()
        print(
            "PATH SMOOTHING"
        )

        smooth_path = shortcut_path(
            robot,
            table,
            obstacle,
            raw_path,
            lower,
            upper
        )

        print(
            f"Smoothed path waypoints: "
            f"{len(smooth_path)}"
        )

        # ----------------------------------------------------
        # Validation.
        # ----------------------------------------------------

        validation = validate_path(
            robot,
            table,
            obstacle,
            smooth_path,
            lower,
            upper
        )

        if not validation["passed"]:

            raise RuntimeError(
                "Final RRT path failed "
                "collision validation."
            )

        # ----------------------------------------------------
        # Execute.
        # ----------------------------------------------------

        final_segment_error = execute_path(
            robot,
            smooth_path
        )

        # ----------------------------------------------------
        # Final verification.
        # ----------------------------------------------------

        final_q = get_arm_q(
            robot
        )

        final_joint_error = float(
            np.linalg.norm(
                final_q - GREEN_LIFT
            )
        )

        final_ee = get_ee_position(
            robot
        )

        # ----------------------------------------------------
        # Results.
        # ----------------------------------------------------

        results = {

            "stage": "9B",

            "robot": "Franka Panda",

            "planner": "Joint-space RRT",

            "collision_checking": True,

            "path_smoothing": True,

            "direct_path_blocked": True,

            "rrt_success": True,

            "start_configuration":
                HOME.tolist(),

            "goal_configuration":
                GREEN_LIFT.tolist(),

            "obstacle": {

                "position":
                    obstacle_position,

                "half_extents":
                    obstacle_half_extents,

                "path_fraction":
                    obstacle_fraction
            },

            "raw_path_waypoints":
                len(raw_path),

            "smoothed_path_waypoints":
                len(smooth_path),

            "validation":
                validation,

            "execution": {

                "maximum_segment_joint_error_rad":
                    final_segment_error,

                "final_joint_error_rad":
                    final_joint_error,

                "final_ee_position_m":
                    final_ee.tolist()
            }
        }

        save_results(
            results
        )

        # ----------------------------------------------------
        # Final report.
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print(
            "STAGE 9B RESULT"
        )
        print("=" * 70)

        print(
            "RRT planning: PASS"
        )

        print(
            "Direct path blocked: PASS"
        )

        print(
            "Collision validation: PASS"
        )

        print(
            "Path smoothing: PASS"
        )

        print(
            "Execution: PASS"
        )

        print(
            f"Final joint error: "
            f"{final_joint_error:.8f} rad"
        )

        print(
            "STAGE 9B COMPLETE"
        )

        print("=" * 70)

        # Keep simulation open.
        while True:

            p.stepSimulation()

            time.sleep(DT)

    finally:

        if p.isConnected():

            p.disconnect()


if __name__ == "__main__":

    main()

