#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from std_msgs.msg import String


class GripperControllerNode(Node):

    def __init__(self):

        super().__init__(
            "gripper_controller_node"
        )

        self.subscription = self.create_subscription(
            String,
            "/gripper/command",
            self.command_callback,
            10
        )

        self.status_publisher = (
            self.create_publisher(
                String,
                "/gripper/status",
                10
            )
        )

        self.state = "OPEN"

        self.get_logger().info(
            "Gripper controller node started."
        )

    def command_callback(
        self,
        msg
    ):

        command = (
            msg.data
            .strip()
            .upper()
        )

        if command == "OPEN":

            self.state = "OPEN"

        elif command == "CLOSE":

            self.state = "CLOSED"

        elif command == "RELEASE":

            self.state = "OPEN"

        else:

            self.publish_status(
                "INVALID_COMMAND"
            )

            return

        self.publish_status(
            self.state
        )

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
            f"Gripper status: {status}"
        )


def main(args=None):

    rclpy.init(
        args=args
    )

    node = GripperControllerNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
