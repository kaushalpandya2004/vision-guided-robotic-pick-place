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
# STAGE 9 - RRT MOTION PLANNING
# ============================================================

DT = 1.0 / 240.0

ROBOT_URDF = "franka_panda/panda.urdf"

EE_LINK = 11

ARM_JOINTS = list(range(7))


# ============================================================
# HOME
# ============================================================

HOME = np.array([
    0.0,
    -0.785398163,
    0.0,
    -2.356194490,
    0.0,
    1.570796327,
    0.785398163
], dtype=float)


# ============================================================
# VERIFIED GREEN LIFT CONFIGURATION
# ============================================================

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
# SCENE
# ============================================================

TABLE_HALF_EXTENTS = [
    1.2,
    1.0,
    0.05
]

TABLE_POSITION = [
    0.0,
    0.0,
    -0.05
]

OBSTACLE_HALF_EXTENTS = [
    0.10,
    0.10,
    0.15
]

OBSTACLE_POSITION = [
    0.25,
    0.0,
    0.15
]

PLATE_HALF_EXTENTS = [
    0.13,
    0.13,
    0.008
]

PLATE_POSITION = [
    0.55,
    0.40,
    0.008
]


# ============================================================
# RRT PARAMETERS
# ============================================================

RRT_MAX_ITERATIONS = 6000

RRT_STEP_SIZE = 0.20

EDGE_RESOLUTION = 0.035

GOAL_BIAS = 0.15

GOAL_TOLERANCE = 0.12

SHORTCUT_ATTEMPTS = 400

RANDOM_SEED = 42


# ============================================================
# JOINT SAFETY MARGIN
# ============================================================

JOINT_LIMIT_MARGIN = math.radians(0.5)


# ============================================================
# CREATE BOX
# ============================================================

def create_box(
    half_extents,
    position,
    color,
    mass=0.0
):

    collision_shape = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=half_extents
    )

    visual_shape = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=half_extents,
        rgbaColor=color
    )

    return p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=position
    )


# ============================================================
# DISTANCE
# ============================================================

def configuration_distance(q1, q2):

    return float(
        np.linalg.norm(
            np.asarray(q2)
            -
            np.asarray(q1)
        )
    )


# ============================================================
# SMOOTHSTEP
# ============================================================

def smoothstep(t):

    return (
        3.0 * t * t
        -
        2.0 * t * t * t
    )


# ============================================================
# INTERPOLATION
# ============================================================

def interpolate_configuration(
    q1,
    q2,
    t
):

    q1 = np.asarray(
        q1,
        dtype=float
    )

    q2 = np.asarray(
        q2,
        dtype=float
    )

    return (
        q1
        +
        smoothstep(t)
        *
        (q2 - q1)
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
# RRT PLANNER
# ============================================================

class PandaRRTPlanner:

    def __init__(
        self,
        robot_id,
        table_id,
        obstacle_id
    ):

        self.robot_id = robot_id

        self.table_id = table_id

        self.obstacle_id = obstacle_id

        self.rng = random.Random(
            RANDOM_SEED
        )

        self.joint_lower = np.radians([
            -169.5,
            -104.5,
            -169.5,
            -179.5,
            -169.5,
            -4.5,
            -169.5
        ])

        self.joint_upper = np.radians([
            169.5,
            104.5,
            169.5,
            -0.5,
            169.5,
            218.5,
            169.5
        ])

    # ========================================================
    # JOINT LIMITS
    # ========================================================

    def within_joint_limits(self, q):

        q = np.asarray(
            q,
            dtype=float
        )

        lower = (
            self.joint_lower
            +
            JOINT_LIMIT_MARGIN
        )

        upper = (
            self.joint_upper
            -
            JOINT_LIMIT_MARGIN
        )

        return bool(
            np.all(q >= lower)
            and
            np.all(q <= upper)
        )

    # ========================================================
    # SET ROBOT CONFIGURATION
    # ========================================================

    def set_configuration(self, q):

        for i, joint in enumerate(
            ARM_JOINTS
        ):

            p.resetJointState(
                self.robot_id,
                joint,
                float(q[i])
            )

    # ========================================================
    # PANDA SELF COLLISION
    # ========================================================

    def self_collision(self):

        contacts = p.getClosestPoints(
            self.robot_id,
            self.robot_id,
            distance=0.0
        )

        for contact in contacts:

            link_a = contact[3]

            link_b = contact[4]

            # ------------------------------------------------
            # Ignore fixed base
            # ------------------------------------------------

            if (
                link_a == -1
                or
                link_b == -1
            ):
                continue

            # ------------------------------------------------
            # Ignore same-link contacts
            # ------------------------------------------------

            if link_a == link_b:
                continue

            # ------------------------------------------------
            # Adjacent links are connected by joints.
            # Their collision geometries can overlap.
            # ------------------------------------------------

            if abs(
                link_a - link_b
            ) <= 1:

                continue

            # ------------------------------------------------
            # Ignore Panda hand/finger structural contacts.
            #
            # Panda:
            #   8 = hand
            #   9 = left finger
            #   10 = right finger
            #   11 = grasp target
            #
            # These are mechanically connected.
            # ------------------------------------------------

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

            if (
                link_a,
                link_b
            ) in connected_pairs:

                continue

            # ------------------------------------------------
            # Any remaining non-adjacent contact is a real
            # self-collision.
            # ------------------------------------------------

            return True

        return False

    # ========================================================
    # TABLE COLLISION
    # ========================================================

    def table_collision(self):

        contacts = p.getClosestPoints(
            self.robot_id,
            self.table_id,
            distance=0.0
        )

        for contact in contacts:

            robot_link = contact[3]

            # Panda fixed base intentionally touches table.
            if robot_link == -1:

                continue

            # Actual arm/finger collision.
            return True

        return False

    # ========================================================
    # OBSTACLE COLLISION
    # ========================================================

    def obstacle_collision(self):

        contacts = p.getClosestPoints(
            self.robot_id,
            self.obstacle_id,
            distance=0.0
        )

        return len(contacts) > 0

    # ========================================================
    # ENVIRONMENT COLLISION
    # ========================================================

    def environment_collision(self):

        if self.table_collision():

            return True

        if self.obstacle_collision():

            return True

        return False

    # ========================================================
    # CONFIGURATION COLLISION
    # ========================================================

    def configuration_in_collision(self, q):

        q = np.asarray(
            q,
            dtype=float
        )

        if not self.within_joint_limits(q):

            return True

        self.set_configuration(q)

        p.performCollisionDetection()

        if self.self_collision():

            return True

        if self.environment_collision():

            return True

        return False

    # ========================================================
    # EDGE VALIDATION
    # ========================================================

    def edge_is_valid(
        self,
        q1,
        q2
    ):

        q1 = np.asarray(
            q1,
            dtype=float
        )

        q2 = np.asarray(
            q2,
            dtype=float
        )

        delta = q2 - q1

        max_change = float(
            np.max(
                np.abs(delta)
            )
        )

        steps = max(
            2,
            int(
                math.ceil(
                    max_change
                    /
                    EDGE_RESOLUTION
                )
            )
        )

        for i in range(
            steps + 1
        ):

            t = (
                i
                /
                float(steps)
            )

            q = (
                q1
                +
                t * delta
            )

            if self.configuration_in_collision(
                q
            ):

                return False

        return True

    # ========================================================
    # RANDOM CONFIGURATION
    # ========================================================

    def random_configuration(self):

        lower = (
            self.joint_lower
            +
            JOINT_LIMIT_MARGIN
        )

        upper = (
            self.joint_upper
            -
            JOINT_LIMIT_MARGIN
        )

        return np.array([
            self.rng.uniform(
                float(lower[i]),
                float(upper[i])
            )
            for i in range(7)
        ])

    # ========================================================
    # NEAREST NODE
    # ========================================================

    def nearest_node(
        self,
        nodes,
        q
    ):

        distances = [
            configuration_distance(
                node.q,
                q
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

    # ========================================================
    # STEER
    # ========================================================

    def steer(
        self,
        q_from,
        q_to
    ):

        q_from = np.asarray(
            q_from,
            dtype=float
        )

        q_to = np.asarray(
            q_to,
            dtype=float
        )

        delta = q_to - q_from

        norm = np.linalg.norm(
            delta
        )

        if norm <= RRT_STEP_SIZE:

            return q_to.copy()

        return (
            q_from
            +
            delta
            /
            norm
            *
            RRT_STEP_SIZE
        )

    # ========================================================
    # RECONSTRUCT PATH
    # ========================================================

    def reconstruct_path(
        self,
        node
    ):

        path = []

        current = node

        while current is not None:

            path.append(
                current.q.copy()
            )

            current = (
                current.parent
            )

        path.reverse()

        return path

    # ========================================================
    # DIRECT PATH
    # ========================================================

    def direct_path(
        self,
        start,
        goal
    ):

        print()
        print(
            "Checking direct joint-space path..."
        )

        if self.edge_is_valid(
            start,
            goal
        ):

            print(
                "Direct path: "
                "COLLISION-FREE"
            )

            return [
                np.asarray(
                    start,
                    dtype=float
                ),
                np.asarray(
                    goal,
                    dtype=float
                )
            ]

        print(
            "Direct path: BLOCKED"
        )

        return None

    # ========================================================
    # RRT PLANNING
    # ========================================================

    def plan(
        self,
        start,
        goal
    ):

        start = np.asarray(
            start,
            dtype=float
        )

        goal = np.asarray(
            goal,
            dtype=float
        )

        print()
        print(
            "=" * 70
        )

        print(
            "RRT PLANNING"
        )

        print(
            "=" * 70
        )

        print(
            "Start-goal distance:",
            f"{configuration_distance(start, goal):.4f} rad"
        )

        # ----------------------------------------------------
        # Validate start
        # ----------------------------------------------------

        if self.configuration_in_collision(
            start
        ):

            raise RuntimeError(
                "RRT start configuration is in collision."
            )

        print(
            "RRT start configuration: PASS"
        )

        # ----------------------------------------------------
        # Validate goal
        # ----------------------------------------------------

        if self.configuration_in_collision(
            goal
        ):

            raise RuntimeError(
                "RRT goal configuration is in collision."
            )

        print(
            "RRT goal configuration: PASS"
        )

        # ----------------------------------------------------
        # Try direct path
        # ----------------------------------------------------

        direct = self.direct_path(
            start,
            goal
        )

        if direct is not None:

            return direct

        # ----------------------------------------------------
        # RRT
        # ----------------------------------------------------

        print()
        print(
            "Starting joint-space RRT..."
        )

        nodes = [
            RRTNode(start)
        ]

        for iteration in range(
            RRT_MAX_ITERATIONS
        ):

            if (
                self.rng.random()
                <
                GOAL_BIAS
            ):

                q_random = goal.copy()

            else:

                q_random = (
                    self.random_configuration()
                )

            nearest = (
                self.nearest_node(
                    nodes,
                    q_random
                )
            )

            q_new = self.steer(
                nearest.q,
                q_random
            )

            if not self.edge_is_valid(
                nearest.q,
                q_new
            ):

                continue

            new_node = RRTNode(
                q_new,
                nearest
            )

            nodes.append(
                new_node
            )

            if (
                configuration_distance(
                    q_new,
                    goal
                )
                <=
                GOAL_TOLERANCE
            ):

                if self.edge_is_valid(
                    q_new,
                    goal
                ):

                    goal_node = RRTNode(
                        goal,
                        new_node
                    )

                    nodes.append(
                        goal_node
                    )

                    path = (
                        self.reconstruct_path(
                            goal_node
                        )
                    )

                    print()
                    print(
                        "RRT SUCCESS"
                    )

                    print(
                        "Iterations:",
                        iteration + 1
                    )

                    print(
                        "Nodes:",
                        len(nodes)
                    )

                    print(
                        "Raw waypoints:",
                        len(path)
                    )

                    return path

            if (
                (iteration + 1)
                %
                500
                ==
                0
            ):

                print(
                    f"Iteration "
                    f"{iteration + 1:5d} | "
                    f"Nodes "
                    f"{len(nodes):5d}"
                )

        raise RuntimeError(
            "RRT failed to find a "
            "collision-free path."
        )

    # ========================================================
    # PATH LENGTH
    # ========================================================

    def path_length(
        self,
        path
    ):

        total = 0.0

        for i in range(
            len(path) - 1
        ):

            total += (
                configuration_distance(
                    path[i],
                    path[i + 1]
                )
            )

        return total

    # ========================================================
    # SHORTCUT SMOOTHING
    # ========================================================

    def shortcut_path(
        self,
        path
    ):

        if len(path) <= 2:

            return path

        path = [
            np.asarray(
                q,
                dtype=float
            )
            for q in path
        ]

        initial_length = (
            self.path_length(path)
        )

        initial_waypoints = len(
            path
        )

        print()
        print(
            "Path smoothing..."
        )

        for _ in range(
            SHORTCUT_ATTEMPTS
        ):

            if len(path) <= 2:

                break

            i = self.rng.randint(
                0,
                len(path) - 2
            )

            j = self.rng.randint(
                i + 1,
                len(path) - 1
            )

            if j <= i + 1:

                continue

            if self.edge_is_valid(
                path[i],
                path[j]
            ):

                path = (
                    path[:i + 1]
                    +
                    path[j:]
                )

        final_length = (
            self.path_length(path)
        )

        print(
            "Initial waypoints:",
            initial_waypoints
        )

        print(
            "Final waypoints:",
            len(path)
        )

        print(
            "Initial path length:",
            f"{initial_length:.4f} rad"
        )

        print(
            "Final path length:",
            f"{final_length:.4f} rad"
        )

        return path

    # ========================================================
    # FINAL PATH VALIDATION
    # ========================================================

    def validate_path(
        self,
        path
    ):

        print()
        print(
            "=" * 70
        )

        print(
            "FINAL PATH VALIDATION"
        )

        print(
            "=" * 70
        )

        if len(path) < 2:

            raise RuntimeError(
                "Path contains fewer than two waypoints."
            )

        checked = 0

        collisions = 0

        for i in range(
            len(path) - 1
        ):

            q1 = np.asarray(
                path[i],
                dtype=float
            )

            q2 = np.asarray(
                path[i + 1],
                dtype=float
            )

            delta = q2 - q1

            steps = max(
                2,
                int(
                    math.ceil(
                        np.max(
                            np.abs(delta)
                        )
                        /
                        EDGE_RESOLUTION
                    )
                )
            )

            for k in range(
                steps + 1
            ):

                t = (
                    k
                    /
                    float(steps)
                )

                q = (
                    q1
                    +
                    t * delta
                )

                checked += 1

                if self.configuration_in_collision(
                    q
                ):

                    collisions += 1

        print(
            "Checked samples:",
            checked
        )

        print(
            "Collision samples:",
            collisions
        )

        if collisions > 0:

            raise RuntimeError(
                "FINAL PATH VALIDATION FAILED."
            )

        print(
            "FINAL PATH VALIDATION: PASS"
        )

        return True

    # ========================================================
    # OBSTACLE CLEARANCE
    # ========================================================

    def minimum_obstacle_clearance(
        self,
        path
    ):

        minimum = float(
            "inf"
        )

        for i in range(
            len(path) - 1
        ):

            q1 = np.asarray(
                path[i],
                dtype=float
            )

            q2 = np.asarray(
                path[i + 1],
                dtype=float
            )

            delta = q2 - q1

            steps = max(
                2,
                int(
                    math.ceil(
                        np.max(
                            np.abs(delta)
                        )
                        /
                        EDGE_RESOLUTION
                    )
                )
            )

            for k in range(
                steps + 1
            ):

                t = (
                    k
                    /
                    float(steps)
                )

                q = (
                    q1
                    +
                    t * delta
                )

                self.set_configuration(
                    q
                )

                p.performCollisionDetection()

                contacts = p.getClosestPoints(
                    self.robot_id,
                    self.obstacle_id,
                    distance=5.0
                )

                for contact in contacts:

                    d = contact[8]

                    if d < minimum:

                        minimum = d

        if minimum == float(
            "inf"
        ):

            return None

        return float(
            minimum
        )


# ============================================================
# ROBOT CONTROL
# ============================================================

def set_arm(
    robot_id,
    q,
    force=250.0
):

    for i, joint in enumerate(
        ARM_JOINTS
    ):

        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=joint,
            controlMode=p.POSITION_CONTROL,
            targetPosition=float(
                q[i]
            ),
            force=force,
            positionGain=0.35,
            velocityGain=1.0
        )


def get_arm_q(
    robot_id
):

    return np.array([
        p.getJointState(
            robot_id,
            joint
        )[0]
        for joint in ARM_JOINTS
    ])


# ============================================================
# SMOOTH EXECUTION
# ============================================================

def move_arm(
    robot_id,
    target_q,
    duration=3.0,
    label=""
):

    target_q = np.asarray(
        target_q,
        dtype=float
    )

    start_q = get_arm_q(
        robot_id
    )

    steps = max(
        1,
        int(
            duration / DT
        )
    )

    print()
    print(
        f"Executing: {label}"
    )

    print(
        f"Duration: {duration:.2f} s"
    )

    print(
        f"Samples: {steps}"
    )

    for step in range(
        steps + 1
    ):

        t = (
            step
            /
            float(steps)
        )

        q = interpolate_configuration(
            start_q,
            target_q,
            t
        )

        set_arm(
            robot_id,
            q
        )

        p.stepSimulation()

        time.sleep(DT)

    print(
        "Settling..."
    )

    for _ in range(
        180
    ):

        set_arm(
            robot_id,
            target_q
        )

        p.stepSimulation()

        time.sleep(DT)

    final_q = get_arm_q(
        robot_id
    )

    error = float(
        np.linalg.norm(
            final_q
            -
            target_q
        )
    )

    print(
        "Final joint convergence:",
        f"{error:.8f} rad"
    )

    if error > 0.01:

        raise RuntimeError(
            "Robot failed to converge."
        )

    return final_q


# ============================================================
# SCENE
# ============================================================

def create_scene():

    p.setAdditionalSearchPath(
        pybullet_data.getDataPath()
    )

    plane_id = p.loadURDF(
        "plane.urdf",
        [0, 0, -0.05]
    )

    table_id = create_box(
        TABLE_HALF_EXTENTS,
        TABLE_POSITION,
        [0.75, 0.75, 0.75, 1.0]
    )

    robot_id = p.loadURDF(
        ROBOT_URDF,
        [0, 0, 0],
        useFixedBase=True
    )

    obstacle_id = create_box(
        OBSTACLE_HALF_EXTENTS,
        OBSTACLE_POSITION,
        [1.0, 1.0, 1.0, 1.0]
    )

    plate_id = create_box(
        PLATE_HALF_EXTENTS,
        PLATE_POSITION,
        [0.02, 0.02, 0.02, 1.0]
    )

    return (
        plane_id,
        table_id,
        robot_id,
        obstacle_id,
        plate_id
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "=" * 70
    )

    print(
        "STAGE 9 - RRT MOTION PLANNING"
    )

    print(
        "=" * 70
    )

    print(
        "Planner: Joint-space RRT"
    )

    print(
        "Robot: Franka Panda"
    )

    print(
        "DOF: 7"
    )

    print(
        "Collision checking: ENABLED"
    )

    print(
        "Path smoothing: ENABLED"
    )

    print(
        "IK inside RRT: DISABLED"
    )

    print(
        "=" * 70
    )

    random.seed(
        RANDOM_SEED
    )

    np.random.seed(
        RANDOM_SEED
    )

    # --------------------------------------------------------
    # Connect
    # --------------------------------------------------------

    client = p.connect(
        p.GUI
    )

    if client < 0:

        raise RuntimeError(
            "Failed to connect to PyBullet."
        )

    p.setGravity(
        0,
        0,
        -9.81
    )

    p.setTimeStep(
        DT
    )

    (
        plane_id,
        table_id,
        robot_id,
        obstacle_id,
        plate_id
    ) = create_scene()

    # --------------------------------------------------------
    # Set HOME
    # --------------------------------------------------------

    for i, joint in enumerate(
        ARM_JOINTS
    ):

        p.resetJointState(
            robot_id,
            joint,
            float(
                HOME[i]
            )
        )

    p.performCollisionDetection()

    planner = PandaRRTPlanner(
        robot_id,
        table_id,
        obstacle_id
    )

    # ========================================================
    # HOME VALIDATION
    # ========================================================

    print()
    print(
        "Start configuration check..."
    )

    if planner.configuration_in_collision(
        HOME
    ):

        print(
            "ERROR: HOME is still "
            "classified as collision."
        )

        print()
        print(
            "This means there is a genuine "
            "non-ignored collision."
        )

        print()
        print(
            "TABLE CONTACTS:"
        )

        for c in p.getClosestPoints(
            robot_id,
            table_id,
            distance=0.01
        ):

            print(
                "link=",
                c[3],
                "distance=",
                c[8]
            )

        print()
        print(
            "OBSTACLE CONTACTS:"
        )

        for c in p.getClosestPoints(
            robot_id,
            obstacle_id,
            distance=0.01
        ):

            print(
                "link=",
                c[3],
                "distance=",
                c[8]
            )

        print()
        print(
            "SELF CONTACTS:"
        )

        for c in p.getClosestPoints(
            robot_id,
            robot_id,
            distance=0.0
        ):

            print(
                "linkA=",
                c[3],
                "linkB=",
                c[4],
                "distance=",
                c[8]
            )

        raise RuntimeError(
            "HOME configuration is in collision."
        )

    print(
        "HOME configuration: PASS"
    )

    # ========================================================
    # GOAL VALIDATION
    # ========================================================

    print()
    print(
        "Goal configuration check..."
    )

    if planner.configuration_in_collision(
        GREEN_LIFT
    ):

        raise RuntimeError(
            "GREEN_LIFT configuration is in collision."
        )

    print(
        "GREEN_LIFT configuration: PASS"
    )

    # ========================================================
    # GREEN FK
    # ========================================================

    planner.set_configuration(
        GREEN_LIFT
    )

    p.performCollisionDetection()

    green_ee = p.getLinkState(
        robot_id,
        EE_LINK,
        computeForwardKinematics=True
    )[4]

    print()
    print(
        "GREEN_LIFT EE position:",
        [
            round(
                value,
                6
            )
            for value in green_ee
        ]
    )

    # Return home
    planner.set_configuration(
        HOME
    )

    # ========================================================
    # RRT
    # ========================================================

    start_time = time.perf_counter()

    raw_path = planner.plan(
        HOME,
        GREEN_LIFT
    )

    planning_time = (
        time.perf_counter()
        -
        start_time
    )

    print()
    print(
        "Planning time:",
        f"{planning_time:.4f} s"
    )

    # ========================================================
    # SMOOTH
    # ========================================================

    smoothed_path = (
        planner.shortcut_path(
            raw_path
        )
    )

    # ========================================================
    # VALIDATE
    # ========================================================

    planner.validate_path(
        smoothed_path
    )

    # ========================================================
    # CLEARANCE
    # ========================================================

    clearance = (
        planner.minimum_obstacle_clearance(
            smoothed_path
        )
    )

    if clearance is None:

        clearance = 0.0

    clearance_mm = (
        clearance * 1000.0
    )

    print()
    print(
        "Minimum obstacle clearance:",
        f"{clearance_mm:.3f} mm"
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_dir = (
        "/mnt/e/Robotics_Assignment/"
        "src/vision_pick_place/results"
    )

    os.makedirs(
        results_dir,
        exist_ok=True
    )

    results_file = os.path.join(
        results_dir,
        "stage9_rrt_results.json"
    )

    results = {

        "stage":
            "Stage 9 - RRT Motion Planning",

        "robot":
            "Franka Panda",

        "dof":
            7,

        "planner":
            "Joint-space RRT",

        "collision_checking":
            True,

        "path_smoothing":
            True,

        "ik_inside_rrt":
            False,

        "random_seed":
            RANDOM_SEED,

        "rrt_parameters": {

            "max_iterations":
                RRT_MAX_ITERATIONS,

            "step_size_rad":
                RRT_STEP_SIZE,

            "edge_resolution_rad":
                EDGE_RESOLUTION,

            "goal_bias":
                GOAL_BIAS,

            "goal_tolerance_rad":
                GOAL_TOLERANCE,

            "shortcut_attempts":
                SHORTCUT_ATTEMPTS
        },

        "start_configuration_rad":
            HOME.tolist(),

        "goal_configuration_rad":
            GREEN_LIFT.tolist(),

        "raw_waypoints":
            len(raw_path),

        "smoothed_waypoints":
            len(smoothed_path),

        "raw_path_length_rad":
            planner.path_length(
                raw_path
            ),

        "smoothed_path_length_rad":
            planner.path_length(
                smoothed_path
            ),

        "planning_time_sec":
            planning_time,

        "minimum_obstacle_clearance_mm":
            clearance_mm,

        "home_validation":
            "PASS",

        "goal_validation":
            "PASS",

        "path_validation":
            "PASS"
    }

    with open(
        results_file,
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
        results_file
    )

    # ========================================================
    # EXECUTE
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "EXECUTING SMOOTH RRT PATH"
    )

    print(
        "=" * 70
    )

    planner.set_configuration(
        HOME
    )

    for i in range(
        len(smoothed_path) - 1
    ):

        move_arm(
            robot_id,
            smoothed_path[i + 1],
            duration=1.5,
            label=(
                f"RRT segment "
                f"{i + 1}/"
                f"{len(smoothed_path) - 1}"
            )
        )

    # ========================================================
    # FINAL CONVERGENCE
    # ========================================================

    final_q = get_arm_q(
        robot_id
    )

    final_error = float(
        np.linalg.norm(
            final_q
            -
            GREEN_LIFT
        )
    )

    print()
    print(
        "=" * 70
    )

    print(
        "STAGE 9 RESULT"
    )

    print(
        "=" * 70
    )

    print(
        "RRT planning: PASS"
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
        "Final joint error:",
        f"{final_error:.8f} rad"
    )

    if final_error > 0.01:

        raise RuntimeError(
            "Final joint convergence failed."
        )

    print()
    print(
        "STAGE 9 COMPLETE"
    )

    print(
        "=" * 70
    )

    while True:

        p.stepSimulation()

        time.sleep(DT)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()

