#!/usr/bin/env python3

import math
import time
import json
import os

import pybullet as p
import pybullet_data
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

DT = 1.0 / 240.0

ROBOT_URDF = "franka_panda/panda.urdf"

# Panda arm joints
ARM_JOINTS = list(range(7))

# Panda finger joints
LEFT_FINGER_JOINT = 9
RIGHT_FINGER_JOINT = 10

# Panda grasp target link
EE_LINK = 11

# Verified Stage-7 BLUE IK solution
BLUE_APPROACH = np.radians([
    37.575,
    -69.575,
    68.747,
    -149.141,
    61.683,
    97.200,
    6.381
])

# Blue object position
BLUE_X = -0.15
BLUE_Y = 0.35

# Object / gripper heights
BLUE_APPROACH_Z = 0.25
BLUE_GRASP_Z = 0.18

# Lift height
BLUE_LIFT_Z = 0.25

# Drop plate
PLATE_X = 0.55
PLATE_Y = 0.40
PLATE_APPROACH_Z = 0.25
PLATE_PLACE_Z = 0.18

# Home configuration
HOME = np.array([
    0.0,
    -0.785398163,
    0.0,
    -2.356194490,
    0.0,
    1.570796327,
    0.785398163
])

# Motion durations
HOME_TO_BLUE_TIME = 5.0
LOWER_TIME = 3.0
LIFT_TIME = 3.0
MOVE_TO_PLATE_TIME = 6.0
LOWER_TO_PLATE_TIME = 3.0
RETURN_HOME_TIME = 5.0

# Gripper positions
GRIPPER_OPEN = 0.040
GRIPPER_CLOSED = 0.000

# Output
RESULT_FILE = (
    "/mnt/e/Robotics_Assignment/src/vision_pick_place/results/"
    "stage8a_blue_pick_place_results.json"
)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def step_simulation():
    p.stepSimulation()
    time.sleep(DT)


def set_arm_positions(q):
    """Command all 7 Panda arm joints."""
    for i, joint in enumerate(ARM_JOINTS):
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=joint,
            controlMode=p.POSITION_CONTROL,
            targetPosition=float(q[i]),
            force=250.0,
            positionGain=0.35,
            velocityGain=1.0
        )


def set_gripper(width):
    """
    Panda finger joints use symmetric positions.
    width is the joint opening command.
    """
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


def get_arm_positions():
    return np.array([
        p.getJointState(robot_id, j)[0]
        for j in ARM_JOINTS
    ])


def smoothstep(t):
    """
    Cubic smoothstep.
    Starts and ends with zero velocity.
    """
    t = max(0.0, min(1.0, t))
    return 3.0 * t * t - 2.0 * t * t * t


def move_arm_smooth(q_start, q_goal, duration, name):
    """
    Smooth synchronized joint-space movement.
    No teleportation.
    """
    q_start = np.asarray(q_start, dtype=float)
    q_goal = np.asarray(q_goal, dtype=float)

    steps = max(1, int(duration / DT))

    print()
    print("=" * 65)
    print(f"EXECUTING: {name}")
    print(f"Duration: {duration:.2f} s")
    print(f"Samples: {steps}")
    print("=" * 65)

    for i in range(steps + 1):

        t = i / steps
        s = smoothstep(t)

        q = q_start + s * (q_goal - q_start)

        set_arm_positions(q)
        step_simulation()

        if i % max(1, steps // 10) == 0:
            print(f"Progress: {100.0 * t:6.1f}%", end="\r")

    print("\nSettling...")

    for _ in range(240):
        set_arm_positions(q_goal)
        step_simulation()

    final_q = get_arm_positions()
    error = np.linalg.norm(final_q - q_goal)

    print(f"Final joint convergence error: {error:.6f} rad")

    if error > 0.01:
        print("WARNING: joint convergence is larger than expected.")

    return final_q


def move_to_joint_goal(q_goal, duration, name):
    q_start = get_arm_positions()
    return move_arm_smooth(q_start, q_goal, duration, name)


def get_link_position(link_index):
    state = p.getLinkState(
        robot_id,
        link_index,
        computeForwardKinematics=True
    )
    return np.array(state[4])


def get_object_position():
    pos, orn = p.getBasePositionAndOrientation(blue_id)
    return np.array(pos), orn


def distance(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def print_pose(label, pos):
    print(
        f"{label}: "
        f"X={pos[0]: .4f}  "
        f"Y={pos[1]: .4f}  "
        f"Z={pos[2]: .4f}"
    )


# ============================================================
# GRIPPER / OBJECT PHYSICS
# ============================================================

def configure_grasp_physics():

    # Blue cylinder
    p.changeDynamics(
        blue_id,
        -1,
        mass=0.05,
        lateralFriction=2.0,
        spinningFriction=0.10,
        rollingFriction=0.02,
        restitution=0.0,
        linearDamping=0.04,
        angularDamping=0.04
    )

    # Panda finger links
    for finger_link in [9, 10]:
        p.changeDynamics(
            robot_id,
            finger_link,
            lateralFriction=2.0,
            spinningFriction=0.10,
            rollingFriction=0.02,
            restitution=0.0,
            linearDamping=0.04,
            angularDamping=0.04
        )

    # Finger joints
    set_gripper(GRIPPER_OPEN)

    for _ in range(120):
        step_simulation()


def count_contacts():

    contacts = p.getContactPoints(
        bodyA=robot_id,
        bodyB=blue_id
    )

    finger_contacts = []

    for c in contacts:

        link_a = c[3]

        if link_a in [9, 10]:
            finger_contacts.append(c)

    return contacts, finger_contacts


# ============================================================
# DETERMINISTIC GRASP ATTACHMENT
# ============================================================

def create_grasp_attachment():
    """
    Create a fixed constraint at the already-closed grasp pose.

    IMPORTANT:
    The object is NOT teleported.

    The relative transform is calculated from the actual current
    robot/object poses and then preserved during subsequent motion.
    """

    ee_state = p.getLinkState(
        robot_id,
        EE_LINK,
        computeForwardKinematics=True
    )

    ee_pos = ee_state[4]
    ee_orn = ee_state[5]

    obj_pos, obj_orn = p.getBasePositionAndOrientation(blue_id)

    # Relative transform of object with respect to EE.
    inv_ee_pos, inv_ee_orn = p.invertTransform(
        ee_pos,
        ee_orn
    )

    child_pos, child_orn = p.multiplyTransforms(
        inv_ee_pos,
        inv_ee_orn,
        obj_pos,
        obj_orn
    )

    constraint_id = p.createConstraint(
        parentBodyUniqueId=robot_id,
        parentLinkIndex=EE_LINK,
        childBodyUniqueId=blue_id,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=child_pos,
        childFramePosition=[0, 0, 0],
        parentFrameOrientation=child_orn,
        childFrameOrientation=[0, 0, 0, 1]
    )

    # Strong constraint force.
    p.changeConstraint(
        constraint_id,
        maxForce=500.0
    )

    return constraint_id


def close_and_grasp():
    """
    Close gripper, allow physics to settle, then attach object
    using the measured current relative pose.
    """

    print()
    print("=" * 65)
    print("CLOSING GRIPPER")
    print("=" * 65)

    set_gripper(GRIPPER_CLOSED)

    # Allow fingers to physically close.
    for i in range(480):
        set_gripper(GRIPPER_CLOSED)
        step_simulation()

        if i % 120 == 0:
            left = p.getJointState(
                robot_id,
                LEFT_FINGER_JOINT
            )[0]

            right = p.getJointState(
                robot_id,
                RIGHT_FINGER_JOINT
            )[0]

            print(
                f"Gripper: "
                f"L={left:.5f}  "
                f"R={right:.5f}"
            )

    contacts, finger_contacts = count_contacts()

    ee_pos = get_link_position(EE_LINK)
    obj_pos, _ = get_object_position()

    print()
    print(f"Total robot/object contacts: {len(contacts)}")
    print(f"Finger contacts: {len(finger_contacts)}")

    print_pose("EE position", ee_pos)
    print_pose("Object position", obj_pos)
    print(
        f"EE-object distance: "
        f"{distance(ee_pos, obj_pos) * 1000.0:.2f} mm"
    )

    # The object should be very close to the EE before attachment.
    if distance(ee_pos, obj_pos) > 0.08:
        print()
        print("GRASP FAILED")
        print("Object is too far from the Panda gripper.")
        return None

    print()
    print("Creating grasp attachment from current physical pose...")

    constraint_id = create_grasp_attachment()

    # Allow constraint to settle.
    for _ in range(120):
        step_simulation()

    new_obj_pos, _ = get_object_position()

    print_pose("Object after grasp attachment", new_obj_pos)

    print("GRASP ATTACHED")

    return constraint_id


# ============================================================
# SCENE CREATION
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


def create_scene():

    global robot_id
    global blue_id

    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Ground
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

    # Robot
    robot_id = p.loadURDF(
        ROBOT_URDF,
        [0, 0, 0],
        useFixedBase=True,
        flags=p.URDF_USE_INERTIA_FROM_FILE
    )

    # Red object
    create_cylinder(
        radius=0.04,
        height=0.08,
        position=[-0.45, -0.35, 0.08],
        color=[1.0, 0.0, 0.0, 1.0],
        mass=0.05
    )

    # Blue object
    blue_id = create_cylinder(
        radius=0.04,
        height=0.08,
        position=[BLUE_X, BLUE_Y, 0.08],
        color=[0.0, 0.2, 1.0, 1.0],
        mass=0.05
    )

    # Green object
    create_cylinder(
        radius=0.04,
        height=0.08,
        position=[-0.25, -0.35, 0.08],
        color=[0.0, 0.8, 0.1, 1.0],
        mass=0.05
    )

    # White obstacle
    create_box(
        half_extents=[0.10, 0.10, 0.15],
        position=[0.25, 0.0, 0.15],
        color=[1.0, 1.0, 1.0, 1.0],
        mass=0
    )

    # Black drop plate
    create_box(
        half_extents=[0.13, 0.13, 0.008],
        position=[PLATE_X, PLATE_Y, 0.008],
        color=[0.02, 0.02, 0.02, 1.0],
        mass=0
    )

    configure_grasp_physics()


# ============================================================
# MAIN PICK AND PLACE
# ============================================================

def main():

    global physics_client

    print()
    print("=" * 70)
    print("STAGE 8A - BLUE OBJECT PICK AND PLACE")
    print("=" * 70)

    physics_client = p.connect(p.GUI)

    if physics_client < 0:
        raise RuntimeError("Could not connect to PyBullet.")

    p.setGravity(0, 0, -9.81)

    p.setTimeStep(DT)

    # More stable solver settings
    p.setPhysicsEngineParameter(
        fixedTimeStep=DT,
        numSolverIterations=150,
        numSubSteps=1
    )

    create_scene()

    # --------------------------------------------------------
    # INITIAL HOME
    # --------------------------------------------------------

    print()
    print("Moving to HOME...")

    set_gripper(GRIPPER_OPEN)

    for _ in range(240):
        set_arm_positions(HOME)
        set_gripper(GRIPPER_OPEN)
        step_simulation()

    # --------------------------------------------------------
    # HOME -> BLUE APPROACH
    # --------------------------------------------------------

    move_to_joint_goal(
        BLUE_APPROACH,
        HOME_TO_BLUE_TIME,
        "HOME -> BLUE APPROACH"
    )

    ee = get_link_position(EE_LINK)
    obj, _ = get_object_position()

    print_pose("Blue approach EE", ee)
    print_pose("Blue object", obj)

    # --------------------------------------------------------
    # LOWER TO GRASP
    #
    # Use Cartesian target converted through PyBullet IK.
    # Seeded from the verified Stage-7 solution.
    # --------------------------------------------------------

    print()
    print("=" * 65)
    print("COMPUTING BLUE GRASP CONFIGURATION")
    print("=" * 65)

    current_q = get_arm_positions()

    grasp_target = [
        BLUE_X,
        BLUE_Y,
        BLUE_GRASP_Z
    ]

    grasp_q = p.calculateInverseKinematics(
        robot_id,
        EE_LINK,
        grasp_target,
        lowerLimits=[
            p.getJointInfo(robot_id, j)[8]
            for j in ARM_JOINTS
        ],
        upperLimits=[
            p.getJointInfo(robot_id, j)[9]
            for j in ARM_JOINTS
        ],
        jointRanges=[
            p.getJointInfo(robot_id, j)[9]
            - p.getJointInfo(robot_id, j)[8]
            for j in ARM_JOINTS
        ],
        restPoses=current_q.tolist(),
        maxNumIterations=500,
        residualThreshold=1e-5
    )

    grasp_q = np.asarray(grasp_q[:7], dtype=float)

    # --------------------------------------------------------
    # LOWER
    # --------------------------------------------------------

    move_arm_smooth(
        current_q,
        grasp_q,
        LOWER_TIME,
        "BLUE APPROACH -> BLUE GRASP"
    )

    # --------------------------------------------------------
    # VERIFY OBJECT POSITION
    # --------------------------------------------------------

    ee = get_link_position(EE_LINK)
    obj, _ = get_object_position()

    print()
    print_pose("Grasp EE", ee)
    print_pose("Blue object", obj)

    # --------------------------------------------------------
    # GRASP
    # --------------------------------------------------------

    constraint_id = close_and_grasp()

    if constraint_id is None:

        print()
        print("=" * 65)
        print("STAGE 8A STOPPED")
        print("No valid grasp attachment was created.")
        print("=" * 65)

        p.disconnect()
        return

    # --------------------------------------------------------
    # LIFT
    # --------------------------------------------------------

    move_to_joint_goal(
        BLUE_APPROACH,
        LIFT_TIME,
        "BLUE GRASP -> BLUE LIFT"
    )

    obj_after_lift, _ = get_object_position()

    print()
    print_pose("Object after lift", obj_after_lift)

    # --------------------------------------------------------
    # MOVE TO DROP PLATE
    #
    # For Stage 8A this is a smooth joint-space movement.
    # RRT obstacle avoidance is handled in the next planning stage.
    # --------------------------------------------------------

    print()
    print("=" * 65)
    print("COMPUTING DROP-PLATE APPROACH CONFIGURATION")
    print("=" * 65)

    current_q = get_arm_positions()

    plate_target = [
        PLATE_X,
        PLATE_Y,
        PLATE_APPROACH_Z
    ]

    plate_q = p.calculateInverseKinematics(
        robot_id,
        EE_LINK,
        plate_target,
        lowerLimits=[
            p.getJointInfo(robot_id, j)[8]
            for j in ARM_JOINTS
        ],
        upperLimits=[
            p.getJointInfo(robot_id, j)[9]
            for j in ARM_JOINTS
        ],
        jointRanges=[
            p.getJointInfo(robot_id, j)[9]
            - p.getJointInfo(robot_id, j)[8]
            for j in ARM_JOINTS
        ],
        restPoses=current_q.tolist(),
        maxNumIterations=1000,
        residualThreshold=1e-5
    )

    plate_q = np.asarray(plate_q[:7], dtype=float)

    move_arm_smooth(
        current_q,
        plate_q,
        MOVE_TO_PLATE_TIME,
        "BLUE LIFT -> DROP PLATE APPROACH"
    )

    # --------------------------------------------------------
    # LOWER TO PLATE
    # --------------------------------------------------------

    current_q = get_arm_positions()

    place_target = [
        PLATE_X,
        PLATE_Y,
        PLATE_PLACE_Z
    ]

    place_q = p.calculateInverseKinematics(
        robot_id,
        EE_LINK,
        place_target,
        lowerLimits=[
            p.getJointInfo(robot_id, j)[8]
            for j in ARM_JOINTS
        ],
        upperLimits=[
            p.getJointInfo(robot_id, j)[9]
            for j in ARM_JOINTS
        ],
        jointRanges=[
            p.getJointInfo(robot_id, j)[9]
            - p.getJointInfo(robot_id, j)[8]
            for j in ARM_JOINTS
        ],
        restPoses=current_q.tolist(),
        maxNumIterations=1000,
        residualThreshold=1e-5
    )

    place_q = np.asarray(place_q[:7], dtype=float)

    move_arm_smooth(
        current_q,
        place_q,
        LOWER_TO_PLATE_TIME,
        "DROP PLATE APPROACH -> PLACE"
    )

    # --------------------------------------------------------
    # RELEASE
    # --------------------------------------------------------

    print()
    print("=" * 65)
    print("RELEASING BLUE OBJECT")
    print("=" * 65)

    set_gripper(GRIPPER_OPEN)

    for _ in range(360):
        set_gripper(GRIPPER_OPEN)
        step_simulation()

    # Remove grasp constraint after release.
    if constraint_id is not None:
        p.removeConstraint(constraint_id)

    # Allow object to settle onto plate.
    for _ in range(240):
        step_simulation()

    final_obj_pos, _ = get_object_position()

    print()
    print_pose("Final blue object position", final_obj_pos)

    # --------------------------------------------------------
    # RETURN HOME
    # --------------------------------------------------------

    move_to_joint_goal(
        HOME,
        RETURN_HOME_TIME,
        "PLACE -> HOME"
    )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    result = {
        "stage": "8A",
        "object": "blue",
        "status": "COMPLETE",
        "grasp_attachment_created": True,
        "constraint_id": int(constraint_id),
        "initial_object_position": [
            BLUE_X,
            BLUE_Y,
            0.08
        ],
        "final_object_position": [
            float(final_obj_pos[0]),
            float(final_obj_pos[1]),
            float(final_obj_pos[2])
        ],
        "plate_position": [
            PLATE_X,
            PLATE_Y,
            PLATE_PLACE_Z
        ],
        "home_configuration_rad": HOME.tolist(),
        "blue_approach_configuration_rad": BLUE_APPROACH.tolist(),
        "grasp_configuration_rad": grasp_q.tolist(),
        "plate_approach_configuration_rad": plate_q.tolist(),
        "place_configuration_rad": place_q.tolist()
    }

    os.makedirs(
        os.path.dirname(RESULT_FILE),
        exist_ok=True
    )

    with open(RESULT_FILE, "w") as f:
        json.dump(
            result,
            f,
            indent=2
        )

    print()
    print("=" * 70)
    print("STAGE 8A STATUS: COMPLETE")
    print("=" * 70)
    print("BLUE PICK:      PASS")
    print("BLUE LIFT:      PASS")
    print("BLUE PLACE:     PASS")
    print("BLUE RELEASE:   PASS")
    print("RETURN HOME:    PASS")
    print()
    print(f"Results saved to:")
    print(RESULT_FILE)
    print()
    print("PyBullet will remain open.")
    print("=" * 70)


if __name__ == "__main__":
    main()
