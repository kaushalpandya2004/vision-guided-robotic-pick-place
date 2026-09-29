#!/usr/bin/env python3

import math
import random
import threading
import time

import numpy as np
import pybullet as p
import pybullet_data

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point
from std_msgs.msg import String


DT = 1.0 / 240.0

ROBOT_URDF = "franka_panda/panda.urdf"

EE_LINK = 11

ARM_JOINTS = list(range(7))

LEFT_FINGER = 9
RIGHT_FINGER = 10


HOME = np.array([
    0.0,
    -0.785398163,
    0.0,
    -2.356194490,
    0.0,
    1.570796327,
    0.785398163,
])

# Verified IK solutions from the validated Stage-7/Stage-8 solution set.
# Used only as deterministic seed candidates for plate-approach IK.
VERIFIED_IK = {
    "A": np.deg2rad([
        89.129, -104.367, 60.777, -100.784,
        96.725, 44.546, 32.686
    ]),
    "B": np.deg2rad([
        37.575, -69.575, 68.747, -149.141,
        61.683, 97.200, 6.381
    ]),
    "C": np.deg2rad([
        127.420, -98.958, 67.210, -129.800,
        78.204, 52.208, 30.793
    ]),
}



OBJECTS = {
    "A": {
        "name": "RED",
        "position": np.array(
            [-0.45, -0.35, 0.04]
        ),
    },

    "B": {
        "name": "BLUE",
        "position": np.array(
            [-0.15, 0.35, 0.04]
        ),
    },

    "C": {
        "name": "GREEN",
        "position": np.array(
            [-0.25, -0.35, 0.04]
        ),
    },
}


# This obstacle was the successful obstacle from
# the independently verified Stage 9B RRT test.
OBSTACLE_POSITION = [
    0.28502324,
    -0.19519229,
    0.22,
]

OBSTACLE_HALF_EXTENTS = [
    0.08,
    0.08,
    0.22,
]


PLATE_POSITION = [
    0.55,
    0.40,
]

PLATE_APPROACH_Z = 0.25
PLATE_RELEASE_Z = 0.12


GRIPPER_OPEN = 0.040
GRIPPER_CLOSED = 0.000


RRT_MAX_ITERATIONS = 6000
RRT_STEP = 0.20
EDGE_RESOLUTION = 0.035
GOAL_BIAS = 0.15
GOAL_TOLERANCE = 0.12

RANDOM_SEED = 42


class MotionPlanningNode(Node):

    def __init__(self):

        super().__init__(
            "motion_planning_node"
        )

        self.selected_object = None
        self.latest_target = None
        self.execution_requested = False

        self.subscription = self.create_subscription(
            Point,
            "/vision/target_pose",
            self.target_callback,
            10,
        )

        self.mission_subscription = self.create_subscription(
            String,
            "/vision/mission_command",
            self.mission_callback,
            10,
        )

        self.status_publisher = self.create_publisher(
            String,
            "/motion/status",
            10,
        )

        self.robot_status_publisher = self.create_publisher(
            String,
            "/robot/status",
            10,
        )

        self.gripper_status_publisher = self.create_publisher(
            String,
            "/gripper/status",
            10,
        )

        self.busy = False
        self.pipeline_failed = False

        self.lock = threading.Lock()

        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        self.client = p.connect(p.GUI)

        if self.client < 0:
            raise RuntimeError(
                "PyBullet GUI connection failed."
            )

        p.setAdditionalSearchPath(
            pybullet_data.getDataPath()
        )

        p.setGravity(
            0,
            0,
            -9.81,
        )

        p.setTimeStep(DT)

        p.setPhysicsEngineParameter(
            fixedTimeStep=DT,
            numSolverIterations=150,
        )

        self.robot = None
        self.table = None
        self.obstacle = None
        self.plate = None
        self.object_ids = {}

        self.create_scene()

        self.set_arm(HOME)
        self.set_gripper(
            GRIPPER_OPEN
        )

        for _ in range(240):
            self.step()

        self.get_logger().info(
            "REAL PyBullet motion planning node started."
        )

        self.get_logger().info(
            "Target selection is operator-controlled: "
            "A=RED, B=BLUE, C=GREEN."
        )

    # ========================================================
    # SCENE
    # ========================================================

    def create_box(
        self,
        half_extents,
        position,
        color,
        mass=0.0,
    ):

        collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
        )

        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=color,
        )

        return p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=position,
        )

    def create_scene(self):

        self.table = self.create_box(
            [1.2, 1.0, 0.05],
            [0.0, 0.0, -0.05],
            [0.55, 0.55, 0.55, 1.0],
        )

        self.robot = p.loadURDF(
            ROBOT_URDF,
            [0.0, 0.0, 0.0],
            useFixedBase=True,
        )

        self.obstacle = self.create_box(
            OBSTACLE_HALF_EXTENTS,
            OBSTACLE_POSITION,
            [1.0, 1.0, 1.0, 1.0],
        )

        self.plate = self.create_box(
            [0.13, 0.13, 0.008],
            [0.55, 0.40, 0.008],
            [0.02, 0.02, 0.02, 1.0],
        )

        for key, data in OBJECTS.items():

            collision = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=0.04,
                height=0.08,
            )

            visual = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=0.04,
                length=0.08,
                rgbaColor={
                    "A": [1.0, 0.0, 0.0, 1.0],
                    "B": [0.0, 0.2, 1.0, 1.0],
                    "C": [0.0, 0.8, 0.1, 1.0],
                }[key],
            )

            object_id = p.createMultiBody(
                baseMass=0.05,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=data["position"],
            )

            self.object_ids[key] = object_id

            p.changeDynamics(
                object_id,
                -1,
                lateralFriction=2.0,
                spinningFriction=0.10,
                rollingFriction=0.02,
                restitution=0.0,
            )

        for link in [LEFT_FINGER, RIGHT_FINGER]:

            p.changeDynamics(
                self.robot,
                link,
                lateralFriction=2.0,
                spinningFriction=0.10,
                rollingFriction=0.02,
                restitution=0.0,
            )

    # ========================================================
    # ROBOT
    # ========================================================

    def step(self):

        p.stepSimulation()
        time.sleep(DT)

    def set_arm(self, q):

        for i, joint in enumerate(ARM_JOINTS):

            p.setJointMotorControl2(
                self.robot,
                joint,
                p.POSITION_CONTROL,
                targetPosition=float(q[i]),
                force=250.0,
                positionGain=0.35,
                velocityGain=1.0,
            )

    def set_gripper(self, width):

        p.setJointMotorControl2(
            self.robot,
            LEFT_FINGER,
            p.POSITION_CONTROL,
            targetPosition=float(width),
            force=100.0,
            positionGain=0.8,
            velocityGain=1.0,
        )

        p.setJointMotorControl2(
            self.robot,
            RIGHT_FINGER,
            p.POSITION_CONTROL,
            targetPosition=float(width),
            force=100.0,
            positionGain=0.8,
            velocityGain=1.0,
        )

    def get_arm_q(self):

        return np.array([
            p.getJointState(
                self.robot,
                joint,
            )[0]
            for joint in ARM_JOINTS
        ])

    def move_smooth(
        self,
        q_start,
        q_goal,
        duration,
        label,
    ):

        q_start = np.array(
            q_start,
            dtype=float,
        )

        q_goal = np.array(
            q_goal,
            dtype=float,
        )

        samples = max(
            60,
            int(duration / DT),
        )

        self.get_logger().info(
            f"Executing {label}: "
            f"{samples} samples"
        )

        for i in range(samples):

            t = i / float(
                samples - 1
            )

            s = (
                3.0 * t * t
                - 2.0 * t * t * t
            )

            q = (
                q_start
                + s * (
                    q_goal
                    - q_start
                )
            )

            self.set_arm(q)
            self.step()

        for _ in range(180):

            self.set_arm(q_goal)
            self.step()

        error = float(
            np.linalg.norm(
                self.get_arm_q()
                - q_goal
            )
        )

        self.get_logger().info(
            f"{label} final joint error: "
            f"{error:.8f} rad"
        )

        return error

    # ========================================================
    # JOINT LIMITS
    # ========================================================

    def joint_limits(self):

        lower = []
        upper = []

        margin = math.radians(
            0.5
        )

        for joint in ARM_JOINTS:

            info = p.getJointInfo(
                self.robot,
                joint,
            )

            lo = float(info[8])
            hi = float(info[9])

            lower.append(
                lo + margin
            )

            upper.append(
                hi - margin
            )

        return (
            np.array(lower),
            np.array(upper),
        )

    def within_limits(self, q):

        lower, upper = (
            self.joint_limits()
        )

        return bool(
            np.all(q >= lower)
            and np.all(q <= upper)
        )

    # ========================================================
    # FK / IK
    # ========================================================

    def fk_position(self, q):

        saved_q = self.get_arm_q().copy()

        for i, joint in enumerate(
            ARM_JOINTS
        ):

            p.resetJointState(
                self.robot,
                joint,
                float(q[i]),
            )

        p.performCollisionDetection()

        state = p.getLinkState(
            self.robot,
            EE_LINK,
            computeForwardKinematics=True,
        )

        position = np.array(
            state[4],
            dtype=float,
        )

        for i, joint in enumerate(
            ARM_JOINTS
        ):

            p.resetJointState(
                self.robot,
                joint,
                float(saved_q[i]),
            )

        p.performCollisionDetection()

        return position

    def solve_ik(
        self,
        target,
        seed=None,
    ):

        target = np.array(
            target,
            dtype=float,
        )

        lower, upper = (
            self.joint_limits()
        )

        ranges = (
            upper - lower
        )

        if seed is None:
            seed = HOME.copy()

        seeds = [
            np.array(seed),
            HOME.copy(),
        ]

        for _ in range(120):

            seeds.append(
                np.random.uniform(
                    lower,
                    upper,
                )
            )

        # Downward-facing gripper.
        target_orn = p.getQuaternionFromEuler(
            [math.pi, 0.0, 0.0]
        )

        candidates = []

        for rest in seeds:

            try:

                solution = p.calculateInverseKinematics(
                    self.robot,
                    EE_LINK,
                    target.tolist(),
                    targetOrientation=target_orn,
                    lowerLimits=lower.tolist(),
                    upperLimits=upper.tolist(),
                    jointRanges=ranges.tolist(),
                    restPoses=rest.tolist(),
                    maxNumIterations=250,
                    residualThreshold=1e-5,
                )

                q = np.array(
                    solution[:7],
                    dtype=float,
                )

                if not np.all(
                    np.isfinite(q)
                ):
                    continue

                if not self.within_limits(q):
                    continue

                actual = self.fk_position(q)

                position_error = float(
                    np.linalg.norm(
                        actual - target
                    )
                )

                if position_error > 0.003:
                    continue

                candidates.append(
                    (
                        float(
                            np.linalg.norm(
                                q - seed
                            )
                        ),
                        q,
                        position_error,
                    )
                )

            except Exception:
                continue

        if not candidates:

            return None

        candidates.sort(
            key=lambda item: item[0]
        )

        return candidates[0][1]

    def solve_ik_position_only(
        self,
        target,
        seed=None,
    ):

        target = np.array(
            target,
            dtype=float,
        )

        lower, upper = (
            self.joint_limits()
        )

        ranges = (
            upper - lower
        )

        if seed is None:

            seed = self.get_arm_q()

        seeds = [
            np.array(
                seed,
                dtype=float,
            ),
            HOME.copy(),
        ]

        for _ in range(250):

            seeds.append(
                np.random.uniform(
                    lower,
                    upper,
                )
            )

        candidates = []

        for rest in seeds:

            try:

                solution = (
                    p.calculateInverseKinematics(
                        self.robot,
                        EE_LINK,
                        target.tolist(),
                        lowerLimits=lower.tolist(),
                        upperLimits=upper.tolist(),
                        jointRanges=ranges.tolist(),
                        restPoses=rest.tolist(),
                        maxNumIterations=400,
                        residualThreshold=1e-5,
                    )
                )

                q = np.array(
                    solution[:7],
                    dtype=float,
                )

                if not np.all(
                    np.isfinite(q)
                ):

                    continue

                if not self.within_limits(q):

                    continue

                actual = self.fk_position(q)

                error = float(
                    np.linalg.norm(
                        actual - target
                    )
                )

                if error > 0.004:

                    continue

                if self.configuration_in_collision(q):

                    continue

                candidates.append(
                    (
                        float(
                            np.linalg.norm(
                                q - seed
                            )
                        ),
                        q,
                        error,
                    )
                )

            except Exception:

                continue

        if not candidates:

            return None

        candidates.sort(
            key=lambda item: item[0]
        )

        return candidates[0][1]

    # ========================================================
    # COLLISION CHECKING
    # ========================================================

    def set_configuration(self, q):

        for i, joint in enumerate(
            ARM_JOINTS
        ):

            p.resetJointState(
                self.robot,
                joint,
                float(q[i]),
            )

        p.performCollisionDetection()

    def self_collision(self):

        contacts = p.getClosestPoints(
            self.robot,
            self.robot,
            distance=0.0,
        )

        ignored_pairs = {
            (6, 8),
            (8, 6),
            (8, 9),
            (9, 8),
            (8, 10),
            (10, 8),
            (9, 10),
            (10, 9),
        }

        for contact in contacts:

            a = contact[3]
            b = contact[4]

            if a == -1 or b == -1:
                continue

            if a == b:
                continue

            if abs(a - b) <= 1:
                continue

            if (a, b) in ignored_pairs:
                continue

            return True

        return False

    def environment_collision(self):

        # Table.
        for link in range(
            -1,
            p.getNumJoints(
                self.robot
            ),
        ):

            contacts = p.getClosestPoints(
                self.robot,
                self.table,
                distance=0.0,
                linkIndexA=link,
                linkIndexB=-1,
            )

            if contacts:

                # Panda base touching the table
                # is intentional.
                if link == -1:
                    continue

                return True

        # White obstacle.
        contacts = p.getClosestPoints(
            self.robot,
            self.obstacle,
            distance=0.0,
        )

        if contacts:
            return True

        return False

    def configuration_in_collision(
        self,
        q,
    ):

        if not self.within_limits(q):
            return True

        saved_q = self.get_arm_q().copy()

        self.set_configuration(q)

        collision = (
            self.self_collision()
            or self.environment_collision()
        )

        self.set_configuration(saved_q)

        return collision

    # ========================================================
    # RRT
    # ========================================================

    def edge_valid(
        self,
        q1,
        q2,
    ):

        q1 = np.array(
            q1,
            dtype=float,
        )

        q2 = np.array(
            q2,
            dtype=float,
        )

        distance = float(
            np.linalg.norm(
                q2 - q1
            )
        )

        steps = max(
            2,
            int(
                math.ceil(
                    distance
                    / EDGE_RESOLUTION
                )
            ),
        )

        for i in range(
            steps + 1
        ):

            alpha = (
                i
                / float(steps)
            )

            q = (
                q1
                + alpha
                * (q2 - q1)
            )

            if self.configuration_in_collision(
                q
            ):
                return False

        return True

    def nearest_node(
        self,
        nodes,
        q,
    ):

        distances = [
            np.linalg.norm(
                node["q"] - q
            )
            for node in nodes
        ]

        return int(
            np.argmin(distances)
        )

    def steer(
        self,
        q_from,
        q_to,
    ):

        delta = (
            q_to - q_from
        )

        distance = float(
            np.linalg.norm(delta)
        )

        if distance <= RRT_STEP:
            return q_to.copy()

        return (
            q_from
            + delta / distance
            * RRT_STEP
        )

    def reconstruct(
        self,
        nodes,
        index,
    ):

        path = []

        while index >= 0:

            path.append(
                nodes[index]["q"]
            )

            index = nodes[index][
                "parent"
            ]

        path.reverse()

        return path

    def plan_rrt(
        self,
        start,
        goal,
        label,
    ):

        start = np.array(
            start,
            dtype=float,
        )

        goal = np.array(
            goal,
            dtype=float,
        )

        self.get_logger().info(
            f"RRT planning: {label}"
        )

        self.publish_status(
            f"RRT_START:{label}:"
            + ",".join(
                f"{v:.5f}"
                for v in start
            )
        )

        self.publish_status(
            f"RRT_GOAL:{label}:"
            + ",".join(
                f"{v:.5f}"
                for v in goal
            )
        )

        if self.configuration_in_collision(
            start
        ):
            raise RuntimeError(
                f"RRT start is invalid: {label}"
            )

        if self.configuration_in_collision(
            goal
        ):
            raise RuntimeError(
                f"RRT goal is invalid: {label}"
            )

        direct = self.edge_valid(
            start,
            goal,
        )

        if direct:

            self.get_logger().info(
                f"{label}: direct path collision-free."
            )

            self.publish_status(
                f"RRT_DIRECT_PASS:{label}"
            )

            self.publish_status(
                "RRT_WAYPOINTS:2"
            )

            self.publish_status(
                "RRT_NODES:2"
            )

            self.publish_status(
                "RRT_ITERATIONS:0"
            )

            return [
                start,
                goal,
            ]

        self.get_logger().info(
            f"{label}: direct path BLOCKED. "
            "Starting RRT."
        )

        self.publish_status(
            f"RRT_BLOCKED:{label}"
        )

        lower, upper = (
            self.joint_limits()
        )

        nodes = [
            {
                "q": start.copy(),
                "parent": -1,
            }
        ]

        for iteration in range(
            RRT_MAX_ITERATIONS
        ):

            if random.random() < GOAL_BIAS:

                sample = goal.copy()

            else:

                sample = np.random.uniform(
                    lower,
                    upper,
                )

            nearest = self.nearest_node(
                nodes,
                sample,
            )

            new_q = self.steer(
                nodes[nearest]["q"],
                sample,
            )

            if self.configuration_in_collision(
                new_q
            ):
                continue

            if not self.edge_valid(
                nodes[nearest]["q"],
                new_q,
            ):
                continue

            nodes.append(
                {
                    "q": new_q.copy(),
                    "parent": nearest,
                }
            )

            new_index = (
                len(nodes) - 1
            )

            if (
                np.linalg.norm(
                    new_q - goal
                )
                <= GOAL_TOLERANCE
            ):

                if self.edge_valid(
                    new_q,
                    goal,
                ):

                    nodes.append(
                        {
                            "q": goal.copy(),
                            "parent": new_index,
                        }
                    )

                    path = self.reconstruct(
                        nodes,
                        len(nodes) - 1,
                    )

                    self.get_logger().info(
                        f"{label}: RRT solution found "
                        f"at iteration {iteration + 1}, "
                        f"nodes={len(nodes)}"
                    )

                    self.publish_status(
                        f"RRT_SOLUTION:{label}"
                    )

                    self.publish_status(
                        f"RRT_ITERATIONS:"
                        f"{iteration + 1}"
                    )

                    self.publish_status(
                        f"RRT_NODES:{len(nodes)}"
                    )

                    self.publish_status(
                        f"RRT_WAYPOINTS:{len(path)}"
                    )

                    clearance_mm = (
                        self.path_min_clearance(
                            path
                        )
                    )

                    self.publish_status(
                        f"RRT_CLEARANCE:"
                        f"{clearance_mm:.3f}"
                    )

                    return path

        raise RuntimeError(
            f"RRT failed after "
            f"{RRT_MAX_ITERATIONS} iterations: "
            f"{label}"
        )

    def configuration_clearance_mm(
        self,
        q,
    ):

        saved_q = (
            self.get_arm_q().copy()
        )

        self.set_configuration(q)

        values = []

        obstacle_points = (
            p.getClosestPoints(
                self.robot,
                self.obstacle,
                distance=2.0,
            )
        )

        for contact in obstacle_points:

            values.append(
                max(
                    0.0,
                    float(contact[8]),
                )
            )

        table_points = (
            p.getClosestPoints(
                self.robot,
                self.table,
                distance=2.0,
            )
        )

        for contact in table_points:

            if contact[3] != -1:

                values.append(
                    max(
                        0.0,
                        float(contact[8]),
                    )
                )

        self.set_configuration(
            saved_q
        )

        if not values:

            return 2000.0

        return min(values) * 1000.0

    def path_min_clearance(
        self,
        path,
    ):

        minimum = 2000.0

        for q1, q2 in zip(
            path[:-1],
            path[1:],
        ):

            distance = float(
                np.linalg.norm(
                    np.array(q2)
                    - np.array(q1)
                )
            )

            steps = max(
                2,
                int(
                    math.ceil(
                        distance
                        / EDGE_RESOLUTION
                    )
                ),
            )

            for i in range(
                steps + 1
            ):

                alpha = (
                    i
                    / float(steps)
                )

                q = (
                    np.array(q1)
                    + alpha
                    * (
                        np.array(q2)
                        - np.array(q1)
                    )
                )

                minimum = min(
                    minimum,
                    self.configuration_clearance_mm(
                        q
                    ),
                )

        return minimum

    # ========================================================
    # PATH SMOOTHING
    # ========================================================

    def smooth_path(
        self,
        path,
    ):

        path = [
            np.array(q)
            for q in path
        ]

        for _ in range(300):

            if len(path) <= 2:
                break

            i = random.randint(
                0,
                len(path) - 2,
            )

            j = random.randint(
                i + 1,
                len(path) - 1,
            )

            if j <= i + 1:
                continue

            if self.edge_valid(
                path[i],
                path[j],
            ):

                path = (
                    path[:i + 1]
                    + path[j:]
                )

        return path

    # ========================================================
    # GRIPPER
    # ========================================================

    def open_gripper(self):

        self.publish_gripper_status(
            "OPENING"
        )

        for _ in range(180):

            self.set_gripper(
                GRIPPER_OPEN
            )

            self.step()

        self.publish_gripper_status(
            "OPEN"
        )

    def close_gripper(self):

        self.publish_gripper_status(
            "CLOSING"
        )

        for _ in range(480):

            self.set_gripper(
                GRIPPER_CLOSED
            )

            self.step()

        self.publish_gripper_status(
            "CLOSED"
        )

    def create_grasp_constraint(
        self,
        object_id,
    ):

        ee_state = p.getLinkState(
            self.robot,
            EE_LINK,
            computeForwardKinematics=True,
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
                ee_orn,
            )
        )

        relative_pos, relative_orn = (
            p.multiplyTransforms(
                inv_pos,
                inv_orn,
                obj_pos,
                obj_orn,
            )
        )

        constraint = p.createConstraint(
            parentBodyUniqueId=self.robot,
            parentLinkIndex=EE_LINK,
            childBodyUniqueId=object_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=relative_pos,
            childFramePosition=[0, 0, 0],
            parentFrameOrientation=relative_orn,
            childFrameOrientation=[
                0,
                0,
                0,
                1,
            ],
        )

        p.changeConstraint(
            constraint,
            maxForce=500.0,
        )

        return constraint

    def grasp(
        self,
        object_id,
    ):

        self.close_gripper()

        contacts = p.getContactPoints(
            bodyA=self.robot,
            bodyB=object_id,
        )

        finger_contacts = [
            c
            for c in contacts
            if c[3] in [
                LEFT_FINGER,
                RIGHT_FINGER,
            ]
        ]

        object_pos, _ = (
            p.getBasePositionAndOrientation(
                object_id
            )
        )

        ee_pos = p.getLinkState(
            self.robot,
            EE_LINK,
            computeForwardKinematics=True,
        )[4]

        distance = np.linalg.norm(
            np.array(object_pos)
            - np.array(ee_pos)
        )

        self.get_logger().info(
            f"Grasp contacts: "
            f"{len(finger_contacts)}, "
            f"EE-object distance: "
            f"{distance * 1000.0:.2f} mm"
        )

        if (
            len(finger_contacts) == 0
            and distance > 0.10
        ):

            return None

        constraint = (
            self.create_grasp_constraint(
                object_id
            )
        )

        return constraint

    # ========================================================
    # STATUS
    # ========================================================

    def publish_status(self, text):

        msg = String()
        msg.data = text

        self.status_publisher.publish(
            msg
        )

        self.get_logger().info(
            text
        )

    def publish_robot_status(
        self,
        text,
    ):

        msg = String()
        msg.data = text

        self.robot_status_publisher.publish(
            msg
        )

    def publish_gripper_status(
        self,
        text,
    ):

        msg = String()
        msg.data = text

        self.gripper_status_publisher.publish(
            msg
        )

    # ========================================================
    # TARGET PIPELINE
    # ========================================================

    def mission_callback(
        self,
        msg,
    ):

        command = msg.data.strip().upper()

        # ----------------------------------------------------
        # RESET
        # ----------------------------------------------------
        if command == "RESET":

            with self.lock:
                self.selected_object = None
                self.latest_target = None
                self.execution_requested = False
                self.busy = False
                self.pipeline_failed = False

            self.set_arm(HOME)
            self.open_gripper()

            self.publish_status(
                "MISSION_RESET"
            )

            self.publish_robot_status(
                "HOME"
            )

            return

        # ----------------------------------------------------
        # SELECT A / B / C
        # ----------------------------------------------------
        if command in OBJECTS:

            with self.lock:
                self.selected_object = command
                self.latest_target = None
                self.execution_requested = False
                self.pipeline_failed = False

            self.publish_status(
                f"TARGET_SELECTED_{command}_"
                f"{OBJECTS[command]['name']}"
            )

            return

        # ----------------------------------------------------
        # EXECUTE A / B / C
        #
        # IMPORTANT:
        # The selected command is authoritative.
        # Do NOT depend on the coordinate-transform node's
        # current target ID, because that node may still be
        # publishing the previous/default object.
        # ----------------------------------------------------
        if command.startswith("EXECUTE:"):

            selected = command.split(
                ":",
                1,
            )[1].strip().upper()

            if selected not in OBJECTS:

                self.publish_status(
                    "EXECUTION_REJECTED_INVALID_TARGET"
                )

                return

            with self.lock:

                if self.busy:

                    self.publish_status(
                        "EXECUTION_REJECTED_BUSY"
                    )

                    return

                if self.pipeline_failed:

                    self.publish_status(
                        "EXECUTION_REJECTED_AFTER_FAILURE"
                    )

                    return

                # EXECUTE also establishes the selected object.
                # This makes the system robust even if SELECT:A/B/C
                # was missed by the motion planner.
                self.selected_object = selected
                self.execution_requested = True

                # Use the known world position corresponding to
                # the requested object. This prevents a stale C
                # target from being used for A or B.
                target = OBJECTS[selected]["position"].copy()

                self.latest_target = target.copy()

            self.publish_status(
                f"EXECUTION_REQUESTED_{selected}"
            )

            self.publish_status(
                f"EXECUTION_TARGET_{selected}:"
                f"{target[0]:.4f},"
                f"{target[1]:.4f},"
                f"{target[2]:.4f}"
            )

            self.start_selected_pipeline(
                selected,
                target,
            )

            return

    def start_selected_pipeline(
        self,
        selected,
        target,
    ):

        with self.lock:

            if self.busy:
                return

            if self.pipeline_failed:
                return

            if self.selected_object != selected:
                return

            self.busy = True
            self.execution_requested = False

        thread = threading.Thread(
            target=self.execute_pipeline,
            args=(
                selected,
                target,
            ),
            daemon=True,
        )

        thread.start()

    def target_callback(
        self,
        msg,
    ):

        target = np.array([
            float(msg.x),
            float(msg.y),
            float(msg.z),
        ])

        if not np.all(
            np.isfinite(target)
        ):

            self.publish_status(
                "TARGET_REJECTED_INVALID"
            )

            return

        distance_map = {

            key: np.linalg.norm(
                target
                - data["position"]
            )

            for key, data
            in OBJECTS.items()
        }

        detected = min(
            distance_map,
            key=distance_map.get,
        )

        if distance_map[detected] > 0.06:

            self.publish_status(
                "TARGET_REJECTED_LOCALIZATION"
            )

            return

        with self.lock:

            self.latest_target = target.copy()

            selected = (
                self.selected_object
            )

            requested = (
                self.execution_requested
            )

            busy = self.busy

            failed = (
                self.pipeline_failed
            )

        self.publish_status(
            "TARGET_WORLD:"
            f"{target[0]:.4f},"
            f"{target[1]:.4f},"
            f"{target[2]:.4f}"
        )

        if selected is None:

            self.publish_status(
                "TARGET_WAITING_SELECTION_"
                f"DETECTED_{detected}"
            )

            return

        if detected != selected:

            self.publish_status(
                "TARGET_SELECTION_MISMATCH_"
                f"SELECTED_{selected}_"
                f"DETECTED_{detected}"
            )

            return

        if (
            requested
            and not busy
            and not failed
        ):

            self.start_selected_pipeline(
                selected,
                target,
            )

    def execute_pipeline(
        self,
        selected,
        detected_target,
    ):

        object_id = (
            self.object_ids[selected]
        )

        constraint = None

        try:

            self.publish_status(
                f"TARGET_ACCEPTED_{selected}"
            )

            self.publish_robot_status(
                "EXECUTION_STARTED"
            )

            # ------------------------------------------------
            # HOME
            # ------------------------------------------------

            self.set_arm(HOME)

            self.open_gripper()

            # ------------------------------------------------
            # APPROACH
            # ------------------------------------------------

            approach_target = np.array([
                detected_target[0],
                detected_target[1],
                0.18,
            ])

            approach_q = self.solve_ik(
                approach_target,
                HOME,
            )

            # Red has a verified object IK configuration but its
            # full-orientation pre-grasp IK can fail at Z=0.18.
            # Use position-only IK only as a Red-specific fallback.
            if approach_q is None and selected == "A":

                approach_q = self.solve_ik_position_only(
                    approach_target,
                    VERIFIED_IK["A"],
                )

                if approach_q is not None:
                    self.get_logger().info(
                        "RED APPROACH: position-only IK fallback PASS"
                    )

            if approach_q is None:

                raise RuntimeError(
                    "IK failure: approach"
                )

            self.publish_status(
                "IK_APPROACH_PASS"
            )

            path = self.plan_rrt(
                self.get_arm_q(),
                approach_q,
                "HOME -> PRE-GRASP",
            )

            path = self.smooth_path(
                path
            )

            self.publish_status(
                f"RRT_APPROACH_PASS_WAYPOINTS_{len(path)}"
            )

            current = self.get_arm_q()

            for i in range(
                len(path) - 1
            ):

                current = path[i]

                self.move_smooth(
                    current,
                    path[i + 1],
                    1.5,
                    f"PRE-GRASP {i + 1}",
                )

            # ------------------------------------------------
            # GRASP
            # ------------------------------------------------

            grasp_zs = (
                [0.12, 0.10, 0.09]
                if selected == "A"
                else [0.08, 0.09, 0.10]
            )

            grasp_q = None

            for grasp_z in grasp_zs:

                grasp_target = np.array([
                    detected_target[0],
                    detected_target[1],
                    grasp_z,
                ])

                candidate = self.solve_ik(
                    grasp_target,
                    self.get_arm_q(),
                )

                if candidate is not None:

                    grasp_q = candidate
                    break

            if grasp_q is None:

                raise RuntimeError(
                    "IK failure: grasp"
                )

            self.publish_status(
                "IK_GRASP_PASS"
            )

            path = self.plan_rrt(
                self.get_arm_q(),
                grasp_q,
                "PRE-GRASP -> GRASP",
            )

            path = self.smooth_path(
                path
            )

            for i in range(
                len(path) - 1
            ):

                self.move_smooth(
                    path[i],
                    path[i + 1],
                    1.2,
                    f"GRASP {i + 1}",
                )

            # ------------------------------------------------
            # CLOSE / ATTACH
            # ------------------------------------------------

            constraint = self.grasp(
                object_id
            )

            if constraint is None:

                raise RuntimeError(
                    "GRASP FAILED: "
                    "no valid object contact"
                )

            self.publish_status(
                "GRASP_PASS"
            )

            # ------------------------------------------------
            # LIFT
            # ------------------------------------------------

            lift_target = np.array([
                detected_target[0],
                detected_target[1],
                0.25,
            ])

            lift_q = self.solve_ik_position_only(
                lift_target,
                self.get_arm_q(),
            )

            if lift_q is None:

                raise RuntimeError(
                    "IK failure: lift"
                )

            self.publish_status(
                "IK_LIFT_PASS"
            )

            path = self.plan_rrt(
                self.get_arm_q(),
                lift_q,
                "GRASP -> LIFT",
            )

            path = self.smooth_path(
                path
            )

            for i in range(
                len(path) - 1
            ):

                self.move_smooth(
                    path[i],
                    path[i + 1],
                    1.5,
                    f"LIFT {i + 1}",
                )

            self.publish_status(
                "LIFT_PASS"
            )

            # ------------------------------------------------
            # MOVE AROUND OBSTACLE
            # ------------------------------------------------

            # Try several safe approach heights above the plate.
            # The fixed downward orientation may not have a valid
            # configuration at the nominal height, so use a small
            # deterministic candidate set.

            plate_q = None
            plate_approach = None

            current_q = self.get_arm_q().copy()

            if self.selected_object == "C":

                plate_approach_heights = [
                    PLATE_APPROACH_Z,
                    0.22,
                    0.20,
                    0.18,
                ]

                plate_seeds = [
                    ("CURRENT", current_q.copy()),
                    ("A", VERIFIED_IK["A"].copy()),
                    ("B", VERIFIED_IK["B"].copy()),
                    ("C", VERIFIED_IK["C"].copy()),
                    ("HOME", HOME.copy()),
                ]

                for plate_z in plate_approach_heights:

                    candidate_target = np.array([
                        PLATE_POSITION[0],
                        PLATE_POSITION[1],
                        plate_z,
                    ], dtype=float)

                    self.get_logger().info(
                        f"GREEN PLATE TEST Z={plate_z:.3f}"
                    )

                    for seed_name, plate_seed in plate_seeds:

                        candidate_q = self.solve_ik_position_only(
                            candidate_target,
                            plate_seed,
                        )

                        if candidate_q is None:
                            self.get_logger().info(
                                f"GREEN PLATE {seed_name} "
                                f"Z={plate_z:.3f}: IK_FAIL"
                            )
                            continue

                        collision = self.configuration_in_collision(
                            candidate_q
                        )

                        self.get_logger().info(
                            f"GREEN PLATE {seed_name} "
                            f"Z={plate_z:.3f}: "
                            f"IK_OK collision={collision}"
                        )

                        if collision:
                            continue

                        plate_q = candidate_q
                        plate_approach = candidate_target

                        self.get_logger().info(
                            f"GREEN PLATE SELECTED "
                            f"seed={seed_name} "
                            f"Z={plate_z:.3f}"
                        )

                        break

                    if plate_q is not None:
                        break

            else:

                plate_approach_heights = [
                    PLATE_APPROACH_Z,
                    0.22,
                    0.20,
                    0.18,
                ]

                plate_seeds = [
                    current_q.copy(),
                    HOME.copy(),
                ]

                for plate_z in plate_approach_heights:

                    candidate_target = np.array([
                        PLATE_POSITION[0],
                        PLATE_POSITION[1],
                        plate_z,
                    ], dtype=float)

                    for plate_seed in plate_seeds:

                        candidate_q = self.solve_ik(
                            candidate_target,
                            plate_seed,
                        )

                        if candidate_q is None:
                            candidate_q = (
                                self.solve_ik_position_only(
                                    candidate_target,
                                    plate_seed,
                                )
                            )

                        if candidate_q is None:
                            continue

                        if self.configuration_in_collision(
                            candidate_q
                        ):
                            continue

                        plate_q = candidate_q
                        plate_approach = candidate_target
                        break

                    if plate_q is not None:
                        break

            if plate_q is None:

                raise RuntimeError(
                    "IK failure: plate approach"
                )

            self.get_logger().info(
                f"Plate approach selected: "
                f"[{plate_approach[0]:.3f}, "
                f"{plate_approach[1]:.3f}, "
                f"{plate_approach[2]:.3f}]"
            )

            self.publish_status(
                "IK_PLATE_APPROACH_PASS"
            )

            path = self.plan_rrt(
                self.get_arm_q(),
                plate_q,
                "LIFT -> PLATE AROUND OBSTACLE",
            )

            path = self.smooth_path(
                path
            )

            self.publish_status(
                f"RRT_OBSTACLE_AVOIDANCE_PASS_WAYPOINTS_{len(path)}"
            )

            for i in range(
                len(path) - 1
            ):

                self.move_smooth(
                    path[i],
                    path[i + 1],
                    1.5,
                    f"RRT {i + 1}",
                )

            # ------------------------------------------------
            # PLACE
            # ------------------------------------------------

            place_q = None

            place_seeds = [
                self.get_arm_q().copy(),
                plate_q.copy(),
                HOME.copy(),
            ]

            if self.selected_object == "C":
                place_heights = [
                    PLATE_RELEASE_Z,
                    0.14,
                    0.16,
                    0.18,
                    0.20,
                ]
            else:
                place_heights = [
                    PLATE_RELEASE_Z,
                    0.14,
                    0.16,
                ]

            for z in place_heights:

                place_target = np.array([
                    PLATE_POSITION[0],
                    PLATE_POSITION[1],
                    z,
                ], dtype=float)

                for place_seed in place_seeds:

                    if self.selected_object == "C":
                        candidate = self.solve_ik_position_only(
                            place_target,
                            place_seed,
                        )
                    else:
                        candidate = self.solve_ik(
                            place_target,
                            place_seed,
                        )

                        if candidate is None:
                            candidate = (
                                self.solve_ik_position_only(
                                    place_target,
                                    place_seed,
                                )
                            )

                    if candidate is None:
                        continue

                    if self.configuration_in_collision(
                        candidate
                    ):
                        continue

                    place_q = candidate
                    break

                if place_q is not None:
                    break

            if place_q is None:

                raise RuntimeError(
                    "IK failure: place"
                )

            path = self.plan_rrt(
                self.get_arm_q(),
                place_q,
                "PLATE APPROACH -> PLACE",
            )

            path = self.smooth_path(
                path
            )

            for i in range(
                len(path) - 1
            ):

                self.move_smooth(
                    path[i],
                    path[i + 1],
                    1.2,
                    f"PLACE {i + 1}",
                )

            self.publish_status(
                "PLACE_POSITION_REACHED"
            )

            # ------------------------------------------------
            # RELEASE
            # ------------------------------------------------

            if constraint is not None:

                p.removeConstraint(
                    constraint
                )

                constraint = None

            self.open_gripper()

            for _ in range(360):
                self.step()

            object_position, _ = (
                p.getBasePositionAndOrientation(
                    object_id
                )
            )

            plate_error = math.hypot(
                object_position[0]
                - PLATE_POSITION[0],
                object_position[1]
                - PLATE_POSITION[1],
            )

            self.get_logger().info(
                f"Placed object XY error: "
                f"{plate_error * 1000.0:.2f} mm"
            )

            if plate_error > 0.10:

                raise RuntimeError(
                    "PLACE FAILED: "
                    "object outside plate tolerance"
                )

            self.publish_status(
                "RELEASE_PASS"
            )

            # ------------------------------------------------
            # RETURN HOME
            # ------------------------------------------------

            path = self.plan_rrt(
                self.get_arm_q(),
                HOME,
                "PLACE -> HOME",
            )

            path = self.smooth_path(
                path
            )

            for i in range(
                len(path) - 1
            ):

                self.move_smooth(
                    path[i],
                    path[i + 1],
                    1.5,
                    f"HOME {i + 1}",
                )

            final_error = float(
                np.linalg.norm(
                    self.get_arm_q()
                    - HOME
                )
            )

            self.publish_status(
                "PICK_PLACE_COMPLETE"
            )

            self.publish_robot_status(
                "EXECUTION_COMPLETE"
            )

            self.get_logger().info(
                "=" * 70
            )

            self.get_logger().info(
                "REAL ROS2 + PYBULLET PICK-AND-PLACE COMPLETE"
            )

            self.get_logger().info(
                f"Selected object: {selected}"
            )

            self.get_logger().info(
                f"Final home error: "
                f"{final_error:.8f} rad"
            )

            self.get_logger().info(
                "=" * 70
            )

        except Exception as exc:

            with self.lock:
                self.pipeline_failed = True

            self.publish_status(
                "PIPELINE_FAILED"
            )

            self.publish_robot_status(
                "SAFE_STOP"
            )

            self.get_logger().error(
                f"PIPELINE FAILURE: {exc}"
            )

            # Safe recovery.
            try:

                if constraint is not None:

                    p.removeConstraint(
                        constraint
                    )

                self.set_gripper(
                    GRIPPER_OPEN
                )

            except Exception:
                pass

        finally:

            with self.lock:
                self.busy = False


def main(args=None):

    rclpy.init(args=args)

    node = MotionPlanningNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:

        if node.client >= 0:

            p.disconnect(
                node.client
            )

        node.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":
    main()
