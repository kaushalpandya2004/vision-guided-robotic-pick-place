#!/usr/bin/env python3

import time
import pybullet as p
import pybullet_data


# ============================================================
# TABLE
# ============================================================

def create_table():

    half_x = 1.20
    half_y = 1.00
    half_z = 0.05

    collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[half_x, half_y, half_z]
    )

    visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[half_x, half_y, half_z],
        rgbaColor=[0.70, 0.70, 0.70, 1.0]
    )

    table_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=[0.0, 0.0, -0.05]
    )

    return table_id


# ============================================================
# CYLINDRICAL OBJECT
# ============================================================

def create_cylinder(position, color, radius=0.035, height=0.08):

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

    object_id = p.createMultiBody(
        baseMass=0.05,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position
    )

    return object_id


# ============================================================
# WHITE OBSTACLE
# ============================================================

def create_obstacle():

    half_x = 0.10
    half_y = 0.10
    half_z = 0.15

    collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[half_x, half_y, half_z]
    )

    visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[half_x, half_y, half_z],
        rgbaColor=[1.0, 1.0, 1.0, 1.0]
    )

    obstacle_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=[0.25, 0.0, 0.15]
    )

    return obstacle_id


# ============================================================
# BLACK DROP PLATE
# ============================================================

def create_drop_plate():

    half_x = 0.13
    half_y = 0.13
    half_z = 0.008

    collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[half_x, half_y, half_z]
    )

    visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[half_x, half_y, half_z],
        rgbaColor=[0.02, 0.02, 0.02, 1.0]
    )

    plate_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=[0.55, 0.40, 0.008]
    )

    return plate_id


# ============================================================
# PANDA
# ============================================================

def load_panda():

    panda_id = p.loadURDF(
        "franka_panda/panda.urdf",
        basePosition=[0.0, 0.0, 0.0],
        baseOrientation=p.getQuaternionFromEuler(
            [0.0, 0.0, 0.0]
        ),
        useFixedBase=True,
        flags=p.URDF_USE_INERTIA_FROM_FILE
    )

    return panda_id


# ============================================================
# PANDA HOME CONFIGURATION
# ============================================================

def set_home_configuration(panda_id):

    home = [
        0.0,
        -0.785398163,
        0.0,
        -2.356194490,
        0.0,
        1.570796327,
        0.785398163
    ]

    for i in range(7):

        p.resetJointState(
            panda_id,
            i,
            home[i]
        )


# ============================================================
# VISUAL LABELS
# ============================================================

def create_labels():

    p.addUserDebugText(
        "RED",
        [-0.45, -0.35, 0.12],
        textColorRGB=[1, 0, 0],
        textSize=1.2
    )

    p.addUserDebugText(
        "BLUE",
        [-0.15, 0.35, 0.12],
        textColorRGB=[0, 0, 1],
        textSize=1.2
    )

    p.addUserDebugText(
        "GREEN",
        [0.05, -0.35, 0.12],
        textColorRGB=[0, 1, 0],
        textSize=1.2
    )

    p.addUserDebugText(
        "OBSTACLE",
        [0.25, 0.0, 0.35],
        textColorRGB=[0, 0, 0],
        textSize=1.0
    )

    p.addUserDebugText(
        "DROP",
        [0.55, 0.40, 0.05],
        textColorRGB=[1, 1, 1],
        textSize=1.2
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("STAGE 3 - COMPLETE WORKSPACE")
    print("=" * 70)

    physics_client = p.connect(p.GUI)

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
    # TABLE
    # --------------------------------------------------------

    table_id = create_table()

    # --------------------------------------------------------
    # PANDA
    # --------------------------------------------------------

    panda_id = load_panda()

    set_home_configuration(
        panda_id
    )

    # --------------------------------------------------------
    # OBJECTS
    # --------------------------------------------------------

    red_id = create_cylinder(
        position=[-0.45, -0.35, 0.08],
        color=[1.0, 0.0, 0.0, 1.0]
    )

    blue_id = create_cylinder(
        position=[-0.15, 0.35, 0.08],
        color=[0.0, 0.0, 1.0, 1.0]
    )

    green_id = create_cylinder(
        position=[-0.25, -0.35, 0.08],
        color=[0.0, 1.0, 0.0, 1.0]
    )

    # --------------------------------------------------------
    # OBSTACLE
    # --------------------------------------------------------

    obstacle_id = create_obstacle()

    # --------------------------------------------------------
    # DROP PLATE
    # --------------------------------------------------------

    drop_plate_id = create_drop_plate()

    # --------------------------------------------------------
    # LABELS
    # --------------------------------------------------------

    create_labels()

    # --------------------------------------------------------
    # INFORMATION
    # --------------------------------------------------------

    print()
    print("TABLE ID       :", table_id)
    print("PANDA ID       :", panda_id)
    print("RED OBJECT ID  :", red_id)
    print("BLUE OBJECT ID :", blue_id)
    print("GREEN OBJECT ID:", green_id)
    print("OBSTACLE ID    :", obstacle_id)
    print("DROP PLATE ID  :", drop_plate_id)

    print()
    print("Workspace configuration:")
    print("  Red   : [-0.45, -0.35, 0.08]")
    print("  Blue  : [-0.15,  0.35, 0.08]")
    print("  Green : [ 0.05, -0.35, 0.08]")
    print("  Obstacle: [0.25, 0.00, 0.15]")
    print("  Drop    : [0.55, 0.40, 0.008]")

    print()
    print("STAGE 3 SUCCESSFUL")
    print("Close PyBullet to finish.")
    print()

    while p.isConnected():

        p.stepSimulation()

        time.sleep(
            1.0 / 240.0
        )

    print("PyBullet closed.")


if __name__ == "__main__":
    main()
