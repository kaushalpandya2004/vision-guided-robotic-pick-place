#!/usr/bin/env python3

import numpy as np

import rclpy
from rclpy.node import Node

from trajectory_msgs.msg import JointTrajectory
from std_msgs.msg import String


class RobotControllerNode(Node):

    def __init__(self):

        super().__init__(
            "robot_controller_node"
        )

        self.subscription = self.create_subscription(
            JointTrajectory,
            "/motion/validated_trajectory",
            self.trajectory_callback,
            10
        )

        self.status_publisher = (
            self.create_publisher(
                String,
                "/robot/status",
                10
            )
        )

        self.executing = False

        self.get_logger().info(
            "Robot controller node started."
        )

    def trajectory_callback(
        self,
        trajectory
    ):

        if self.executing:

            self.publish_status(
                "REJECTED_ALREADY_EXECUTING"
            )

            return

        if len(
            trajectory.points
        ) == 0:

            self.publish_status(
                "REJECTED_EMPTY_TRAJECTORY"
            )

            return

        # Validate before execution.
        for point in trajectory.points:

            values = np.asarray(
                point.positions,
                dtype=float
            )

            if (
                len(values) != 7
                or not np.all(
                    np.isfinite(values)
                )
            ):

                self.publish_status(
                    "REJECTED_INVALID_TRAJECTORY"
                )

                return

        self.executing = True

        self.publish_status(
            "EXECUTION_STARTED"
        )

        self.get_logger().info(
            f"Executing validated trajectory "
            f"with {len(trajectory.points)} points."
        )

        # Architecture-level execution acknowledgement.
        # Actual PyBullet execution remains in the
        # validated Stage 8/9 simulation.
        self.publish_status(
            "EXECUTION_COMPLETE"
        )

        self.executing = False

    def publish_status(
        self,
        status
    ):

        msg = String()

        msg.data = status

        self.status_publisher.publish(
            msg
        )

        self.get_logger().info(
            f"Robot status: {status}"
        )


def main(args=None):

    rclpy.init(
        args=args
    )

    node = RobotControllerNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
