#!/usr/bin/env python3

import os
import json
import math
import numpy as np
import pybullet as p
import pybullet_data


# =============================================================================
# CONFIGURATION
# =============================================================================

POSITION_TOLERANCE_M = 0.001
ORIENTATION_TOLERANCE_DEG = 2.0
ORIENTATION_TOLERANCE_RAD = math.radians(
    ORIENTATION_TOLERANCE_DEG
)

# Search the complete tool yaw.
YAW_STEP_DEG = 5.0

# Search a small cone around vertical downward.
TILT_VALUES_DEG = [
    -15.0,
    -10.0,
    -5.0,
     0.0,
     5.0,
    10.0,
    15.0
]

REST_CONFIGURATIONS = 100

IK_ITERATIONS = 5000
IK_RESIDUAL = 1e-8

LIMIT_MARGIN_DEG = 0.5

TARGETS = {
    "RED": np.array(
        [-0.45, -0.35, 0.25],
        dtype=float
    ),

    "BLUE": np.array(
        [-0.15, 0.35, 0.25],
        dtype=float
    ),

    "GREEN": np.array(
        [-0.25, -0.35, 0.25],
        dtype=float
    )
}

HOME = np.array([
    0.0,
    -0.785398163,
    0.0,
    -2.356194490,
    0.0,
    1.570796327,
    0.785398163
], dtype=float)

RESULT_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "results",
        "stage7_ik_solutions.json"
    )
)


# =============================================================================
# ORIENTATION
# =============================================================================

def make_tool_orientation(
    yaw_deg,
    tilt_deg
):
    """
    Generate a gripper orientation that points generally downward.

    yaw_deg:
        Rotation around the vertical/tool approach axis.

    tilt_deg:
        Small deviation from the vertical downward approach.

    The resulting orientation is still suitable for a top-down grasp.
    """

    yaw = math.radians(yaw_deg)
    tilt = math.radians(tilt_deg)

    # Start with a downward-facing tool.
    base = p.getQuaternionFromEuler([
        math.pi,
        0.0,
        yaw
    ])

    # Additional pitch around the tool wrist.
    correction = p.getQuaternionFromEuler([
        0.0,
        tilt,
        0.0
    ])

    return np.array(
        p.multiplyTransforms(
            [0, 0, 0],
            base,
            [0, 0, 0],
            correction
        )[1],
        dtype=float
    )


# =============================================================================
# QUATERNION ERROR
# =============================================================================

def quaternion_error(
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

    q1 /= np.linalg.norm(q1)
    q2 /= np.linalg.norm(q2)

    dot = abs(
        np.dot(q1, q2)
    )

    dot = max(
        -1.0,
        min(1.0, dot)
    )

    return 2.0 * math.acos(dot)


# =============================================================================
# PANDA IK CLASS
# =============================================================================

class PandaIK:

    def __init__(self):

        self.client = p.connect(
            p.GUI
        )

        p.setAdditionalSearchPath(
            pybullet_data.getDataPath()
        )

        p.resetSimulation()

        p.setGravity(
            0,
            0,
            -9.81
        )

        self.robot = p.loadURDF(
            "franka_panda/panda.urdf",
            [0, 0, 0],
            useFixedBase=True
        )

        self.arm_joints = list(
            range(7)
        )

        self.ee_link = 11

        self.lower = []
        self.upper = []
        self.ranges = []

        for joint in self.arm_joints:

            info = p.getJointInfo(
                self.robot,
                joint
            )

            low = float(info[8])
            high = float(info[9])

            margin = math.radians(
                LIMIT_MARGIN_DEG
            )

            low += margin
            high -= margin

            self.lower.append(low)
            self.upper.append(high)
            self.ranges.append(
                high - low
            )

        self.lower = np.array(
            self.lower
        )

        self.upper = np.array(
            self.upper
        )

        self.ranges = np.array(
            self.ranges
        )

        self.print_limits()

        self.set_configuration(
            HOME
        )

    # -------------------------------------------------------------------------
    # Joint limits
    # -------------------------------------------------------------------------

    def print_limits(self):

        print()
        print("=" * 80)
        print("PANDA ARM JOINT LIMITS")
        print("=" * 80)

        for i in range(7):

            print(
                f"J{i+1}: "
                f"{math.degrees(self.lower[i]):8.3f}° "
                f"to "
                f"{math.degrees(self.upper[i]):8.3f}°"
            )

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    def set_configuration(
        self,
        q
    ):

        for i in range(7):

            p.resetJointState(
                self.robot,
                i,
                float(q[i])
            )

        p.stepSimulation()

    # -------------------------------------------------------------------------
    # FK
    # -------------------------------------------------------------------------

    def fk(
        self,
        q
    ):

        self.set_configuration(
            q
        )

        state = p.getLinkState(
            self.robot,
            self.ee_link,
            computeForwardKinematics=True
        )

        position = np.array(
            state[4],
            dtype=float
        )

        orientation = np.array(
            state[5],
            dtype=float
        )

        return position, orientation

    # -------------------------------------------------------------------------
    # Joint limit validation
    # -------------------------------------------------------------------------

    def valid_limits(
        self,
        q
    ):

        q = np.asarray(q)

        return bool(
            np.all(q >= self.lower)
            and
            np.all(q <= self.upper)
        )

    # -------------------------------------------------------------------------
    # FK verification
    # -------------------------------------------------------------------------

    def verify(
        self,
        q,
        target_position,
        target_orientation
    ):

        if not self.valid_limits(q):

            return (
                False,
                None,
                None
            )

        position, orientation = (
            self.fk(q)
        )

        position_error = np.linalg.norm(
            position - target_position
        )

        orientation_error = (
            quaternion_error(
                orientation,
                target_orientation
            )
        )

        valid = (
            position_error
            <= POSITION_TOLERANCE_M
            and
            orientation_error
            <= ORIENTATION_TOLERANCE_RAD
        )

        return (
            valid,
            position_error,
            orientation_error
        )

    # -------------------------------------------------------------------------
    # Rest configurations
    # -------------------------------------------------------------------------

    def generate_rest_configurations(
        self
    ):

        configs = []

        # Home
        configs.append(
            HOME.copy()
        )

        # Known useful Panda postures.
        configs.extend([
            np.array([
                0.0,
                -1.0,
                0.0,
                -2.0,
                0.0,
                1.5,
                0.7
            ]),

            np.array([
                0.0,
                -1.5,
                0.8,
                -2.5,
                0.0,
                1.6,
                0.7
            ]),

            np.array([
                0.0,
                -1.5,
                -0.8,
                -2.5,
                0.0,
                1.6,
                0.7
            ]),

            np.array([
                1.0,
                -1.2,
                1.0,
                -2.3,
                0.5,
                1.5,
                0.8
            ]),

            np.array([
                -1.0,
                -1.2,
                -1.0,
                -2.3,
                -0.5,
                1.5,
                -0.8
            ]),

            np.array([
                1.5,
                -1.5,
                1.2,
                -2.8,
                0.5,
                1.8,
                1.0
            ]),

            np.array([
                -1.5,
                -1.5,
                -1.2,
                -2.8,
                -0.5,
                1.8,
                -1.0
            ])
        ])

        rng = np.random.default_rng(
            20260928
        )

        for _ in range(
            REST_CONFIGURATIONS
        ):

            q = rng.uniform(
                self.lower,
                self.upper
            )

            configs.append(q)

        output = []

        for q in configs:

            q = np.clip(
                q,
                self.lower,
                self.upper
            )

            output.append(q)

        return output

    # -------------------------------------------------------------------------
    # Search
    # -------------------------------------------------------------------------

    def search(
        self,
        name,
        target
    ):

        print()
        print("=" * 80)
        print(
            f"SEARCHING DOWNWARD IK FOR {name}"
        )
        print("=" * 80)

        yaw_values = np.arange(
            0.0,
            360.0,
            YAW_STEP_DEG
        )

        rest_configs = (
            self.generate_rest_configurations()
        )

        print(
            f"Yaw candidates      : "
            f"{len(yaw_values)}"
        )

        print(
            f"Tilt candidates     : "
            f"{len(TILT_VALUES_DEG)}"
        )

        print(
            f"Rest configurations : "
            f"{len(rest_configs)}"
        )

        maximum = (
            len(yaw_values)
            *
            len(TILT_VALUES_DEG)
            *
            len(rest_configs)
        )

        print(
            f"Maximum attempts    : "
            f"{maximum}"
        )

        attempts = 0
        candidates = []

        for tilt in TILT_VALUES_DEG:

            for yaw in yaw_values:

                target_orientation = (
                    make_tool_orientation(
                        yaw,
                        tilt
                    )
                )

                for rest in rest_configs:

                    attempts += 1

                    try:

                        solution = (
                            p.calculateInverseKinematics(
                                self.robot,
                                self.ee_link,
                                target.tolist(),
                                targetOrientation=
                                    target_orientation.tolist(),
                                lowerLimits=
                                    self.lower.tolist(),
                                upperLimits=
                                    self.upper.tolist(),
                                jointRanges=
                                    self.ranges.tolist(),
                                restPoses=
                                    rest.tolist(),
                                maxNumIterations=
                                    IK_ITERATIONS,
                                residualThreshold=
                                    IK_RESIDUAL
                            )
                        )

                    except Exception:
                        continue

                    q = np.array(
                        solution[:7],
                        dtype=float
                    )

                    valid, pos_error, ori_error = (
                        self.verify(
                            q,
                            target,
                            target_orientation
                        )
                    )

                    if not valid:
                        continue

                    # Remove duplicate configurations.
                    duplicate = False

                    for candidate in candidates:

                        if np.linalg.norm(
                            q - candidate["q"]
                        ) < 1e-3:

                            duplicate = True
                            break

                    if duplicate:
                        continue

                    distance_home = np.linalg.norm(
                        q - HOME
                    )

                    candidates.append({
                        "q": q,
                        "yaw": float(yaw),
                        "tilt": float(tilt),
                        "position_error":
                            float(pos_error),
                        "orientation_error":
                            float(ori_error),
                        "distance_home":
                            float(distance_home)
                    })

        candidates.sort(
            key=lambda c:
                c["distance_home"]
        )

        return candidates, attempts

    # -------------------------------------------------------------------------
    # Print
    # -------------------------------------------------------------------------

    def print_solution(
        self,
        name,
        target,
        candidate,
        attempts,
        candidate_count
    ):

        q = candidate["q"]

        print()
        print("=" * 80)
        print(
            f"VALID IK SOLUTION: {name}"
        )
        print("=" * 80)

        print()
        print("TARGET")
        print("-" * 80)

        print(
            f"X = {target[0]: .6f} m"
        )

        print(
            f"Y = {target[1]: .6f} m"
        )

        print(
            f"Z = {target[2]: .6f} m"
        )

        print()
        print(
            f"Tool yaw  = "
            f"{candidate['yaw']:.1f}°"
        )

        print(
            f"Tool tilt = "
            f"{candidate['tilt']:.1f}°"
        )

        print()
        print("JOINT SOLUTION")
        print("-" * 80)

        for i in range(7):

            print(
                f"J{i+1} = "
                f"{q[i]: .9f} rad "
                f"({math.degrees(q[i]): .3f}°)"
            )

        print()
        print("JOINT LIMIT CHECK")
        print("-" * 80)

        for i in range(7):

            value = math.degrees(
                q[i]
            )

            low = math.degrees(
                self.lower[i]
            )

            high = math.degrees(
                self.upper[i]
            )

            print(
                f"J{i+1}: PASS | "
                f"{low:.2f}° <= "
                f"{value:.2f}° <= "
                f"{high:.2f}°"
            )

        position, orientation = (
            self.fk(q)
        )

        position_error = np.linalg.norm(
            position - target
        )

        orientation_error = (
            quaternion_error(
                orientation,
                make_tool_orientation(
                    candidate["yaw"],
                    candidate["tilt"]
                )
            )
        )

        print()
        print("FK VERIFICATION")
        print("-" * 80)

        print(
            f"FK X = {position[0]: .9f} m"
        )

        print(
            f"FK Y = {position[1]: .9f} m"
        )

        print(
            f"FK Z = {position[2]: .9f} m"
        )

        print()
        print(
            f"Position error    = "
            f"{position_error * 1000:.4f} mm"
        )

        print(
            f"Orientation error = "
            f"{math.degrees(orientation_error):.6f}°"
        )

        print()
        print("SEARCH STATISTICS")
        print("-" * 80)

        print(
            f"IK attempts          : "
            f"{attempts}"
        )

        print(
            f"Valid IK candidates  : "
            f"{candidate_count}"
        )

        print(
            f"Distance from home   : "
            f"{candidate['distance_home']:.6f} rad"
        )

        print()
        print(
            "IK STATUS: SUCCESS"
        )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    def save_results(
        self,
        results
    ):

        os.makedirs(
            os.path.dirname(
                RESULT_PATH
            ),
            exist_ok=True
        )

        output = {}

        for name, data in results.items():

            if data is None:

                output[name] = None

                continue

            q = data["q"]

            output[name] = {
                "target_position_m":
                    data["target"].tolist(),

                "tool_yaw_deg":
                    float(data["yaw"]),

                "tool_tilt_deg":
                    float(data["tilt"]),

                "joint_positions_rad":
                    q.tolist(),

                "joint_positions_deg":
                    [
                        float(
                            math.degrees(x)
                        )
                        for x in q
                    ],

                "position_error_mm":
                    float(
                        data["position_error"]
                        * 1000.0
                    ),

                "orientation_error_deg":
                    float(
                        math.degrees(
                            data["orientation_error"]
                        )
                    )
            }

        with open(
            RESULT_PATH,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                output,
                f,
                indent=4
            )

        print()
        print(
            "Saved verified IK results:"
        )

        print(
            RESULT_PATH
        )


# =============================================================================
# MAIN
# =============================================================================

def main():

    solver = PandaIK()

    results = {}

    for name, target in TARGETS.items():

        candidates, attempts = (
            solver.search(
                name,
                target
            )
        )

        if not candidates:

            print()
            print("=" * 80)
            print(
                f"IK FAILURE: {name}"
            )
            print("=" * 80)

            print(
                "No verified joint-limit-valid "
                "downward grasp pose was found."
            )

            print(
                f"Attempts: {attempts}"
            )

            results[name] = None

            continue

        best = candidates[0]

        results[name] = {
            "target": target,
            "q": best["q"],
            "yaw": best["yaw"],
            "tilt": best["tilt"],
            "position_error":
                best["position_error"],
            "orientation_error":
                best["orientation_error"]
        }

        solver.print_solution(
            name,
            target,
            best,
            attempts,
            len(candidates)
        )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("STAGE 7 FINAL SUMMARY")
    print("=" * 80)

    successful = 0
    errors = []

    for name in TARGETS:

        data = results[name]

        if data is None:

            print(
                f"{name:<8}: FAIL"
            )

        else:

            successful += 1

            error_mm = (
                data["position_error"]
                * 1000.0
            )

            errors.append(
                error_mm
            )

            print(
                f"{name:<8}: PASS | "
                f"Position = "
                f"{error_mm:.4f} mm | "
                f"Orientation = "
                f"{math.degrees(data['orientation_error']):.6f}°"
            )

    print("-" * 80)

    print(
        f"IK/FK successful : "
        f"{successful}/3"
    )

    if errors:

        print(
            f"Average position error : "
            f"{np.mean(errors):.4f} mm"
        )

        print(
            f"Maximum position error : "
            f"{np.max(errors):.4f} mm"
        )

    if successful == 3:

        print()
        print(
            "STAGE 7 STATUS: COMPLETE"
        )

    else:

        print()
        print(
            "STAGE 7 STATUS: INCOMPLETE"
        )

    solver.save_results(
        results
    )

    print()
    print(
        "Press Q or ESC to close."
    )

    while True:

        keys = p.getKeyboardEvents()

        if ord("q") in keys:
            break

        if 27 in keys:
            break

        p.stepSimulation()

    p.disconnect()


if __name__ == "__main__":
    main()
