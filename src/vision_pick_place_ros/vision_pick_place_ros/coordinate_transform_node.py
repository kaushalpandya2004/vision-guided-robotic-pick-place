#!/usr/bin/env python3

import json
import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point
from std_msgs.msg import String


IMAGE_WIDTH = 800
IMAGE_HEIGHT = 800

CAMERA_Z = 2.5
OBJECT_Z = 0.04

FOV_DEG = 60.0

FOCAL_LENGTH = (
    IMAGE_WIDTH / 2.0
    / math.tan(
        math.radians(FOV_DEG) / 2.0
    )
)


class CoordinateTransformNode(Node):

    def __init__(self):

        super().__init__(
            "coordinate_transform_node"
        )

        self.declare_parameter(
            "target_id",
            "C",
        )

        self.target_id = (
            self.get_parameter(
                "target_id"
            )
            .value
        )

        self.publisher = self.create_publisher(
            Point,
            "/vision/target_pose",
            10,
        )

        self.subscription = self.create_subscription(
            String,
            "/vision/detection",
            self.detection_callback,
            10,
        )

        self.get_logger().info(
            "Real camera-to-world transform node started."
        )

        self.get_logger().info(
            f"Selected target: {self.target_id}"
        )

    def pixel_to_world(
        self,
        px,
        py,
    ):

        cx = IMAGE_WIDTH / 2.0
        cy = IMAGE_HEIGHT / 2.0

        x_world = (
            (px - cx)
            * OBJECT_Z
            / FOCAL_LENGTH
            + 0.0
        )

        y_world = (
            (cy - py)
            * OBJECT_Z
            / FOCAL_LENGTH
            + 0.0
        )

        # Correct scale using actual camera height.
        # The ray-plane relationship uses camera height,
        # not object Z.
        x_world = (
            (px - cx)
            * CAMERA_Z
            / FOCAL_LENGTH
        )

        y_world = (
            (cy - py)
            * CAMERA_Z
            / FOCAL_LENGTH
        )

        return (
            x_world,
            y_world,
            OBJECT_Z,
        )

    def detection_callback(self, msg):

        try:

            detections = json.loads(
                msg.data
            )

            if self.target_id not in detections:

                self.get_logger().warning(
                    f"Target {self.target_id} not detected."
                )

                return

            detection = detections[
                self.target_id
            ]

            px, py = detection["pixel"]

            x, y, z = self.pixel_to_world(
                float(px),
                float(py),
            )

            point = Point()

            point.x = float(x)
            point.y = float(y)
            point.z = float(z)

            self.publisher.publish(point)

            self.get_logger().info(
                f"Target {self.target_id} -> "
                f"world "
                f"({x:.4f}, {y:.4f}, {z:.4f})"
            )

        except Exception as exc:

            self.get_logger().error(
                f"Transform error: {exc}"
            )


def main(args=None):

    rclpy.init(args=args)

    node = CoordinateTransformNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
