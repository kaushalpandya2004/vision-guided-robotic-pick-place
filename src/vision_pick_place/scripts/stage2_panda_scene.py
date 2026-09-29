#!/usr/bin/env python3

import time
import pybullet as p
import pybullet_data


def create_table():
    table_half_x = 1.20
    table_half_y = 1.00
    table_half_z = 0.05

    collision_shape = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[
            table_half_x,
            table_half_y,
            table_half_z
        ]
    )

    visual_shape = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[
            table_half_x,
            table_half_y,
            table_half_z
        ],
        rgbaColor=[0.70, 0.70, 0.70, 1.0]
    )

    table_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=[0.0, 0.0, -0.05]
    )

    return table_id


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


def set_home_configuration(panda_id):

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


def print_panda_information(panda_id):

    print()
    print("=" * 70)
    print("FRANKA PANDA INFORMATION")
    print("=" * 70)

    number_of_joints = p.getNumJoints(panda_id)

    print(f"Total PyBullet joints/links: {number_of_joints}")
    print()

    for joint_index in range(number_of_joints):

        joint_info = p.getJointInfo(
            panda_id,
            joint_index
        )

        joint_name = joint_info[1].decode("utf-8")
        joint_type = joint_info[2]
        link_name = joint_info[12].decode("utf-8")

        print(
            f"{joint_index:2d} | "
            f"Joint: {joint_name:20s} | "
            f"Type: {joint_type} | "
            f"Link: {link_name}"
        )

    print("=" * 70)


def main():

    print()
    print("=" * 70)
    print("STAGE 2 - FRANKA PANDA + TABLE")
    print("=" * 70)

    physics_client = p.connect(p.GUI)

    if physics_client < 0:
        raise RuntimeError(
            "Could not connect to PyBullet GUI."
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

    table_id = create_table()

    panda_id = load_panda()

    set_home_configuration(
        panda_id
    )

    print_panda_information(
        panda_id
    )

    print()
    print(f"Table ID : {table_id}")
    print(f"Panda ID : {panda_id}")
    print()
    print("STAGE 2 INITIALIZATION SUCCESSFUL")
    print("Close the PyBullet window to stop the test.")
    print()

    while p.isConnected():

        p.stepSimulation()

        time.sleep(
            1.0 / 240.0
        )

    print("PyBullet closed.")


if __name__ == "__main__":
    main()
