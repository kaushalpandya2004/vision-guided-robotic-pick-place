#!/usr/bin/env python3

import json

import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import String


class ObjectDetectionNode(Node):

    def __init__(self):

        super().__init__("object_detection_node")

        self.publisher = self.create_publisher(
            String,
            "/vision/detection",
            10,
        )

        self.subscription = self.create_subscription(
            Image,
            "/camera/image_raw",
            self.image_callback,
            10,
        )

        self.get_logger().info(
            "Real HSV object detection node started."
        )

    def detect_color(
        self,
        hsv,
        lower,
        upper,
    ):

        mask = cv2.inRange(
            hsv,
            np.array(lower),
            np.array(upper),
        )

        kernel = np.ones(
            (5, 5),
            np.uint8,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        best = None
        best_area = 0.0

        for contour in contours:

            area = cv2.contourArea(contour)

            if area < 100.0:
                continue

            if area > best_area:

                moments = cv2.moments(contour)

                if moments["m00"] == 0:
                    continue

                cx = (
                    moments["m10"]
                    / moments["m00"]
                )

                cy = (
                    moments["m01"]
                    / moments["m00"]
                )

                best = {
                    "pixel": [
                        float(cx),
                        float(cy),
                    ],
                    "area": float(area),
                }

                best_area = area

        return best

    def image_callback(self, msg):

        try:

            image = np.frombuffer(
                msg.data,
                dtype=np.uint8,
            ).reshape(
                msg.height,
                msg.width,
                3,
            )

            hsv = cv2.cvtColor(
                image,
                cv2.COLOR_RGB2HSV,
            )

            detections = {}

            red1 = self.detect_color(
                hsv,
                [0, 100, 80],
                [10, 255, 255],
            )

            red2 = self.detect_color(
                hsv,
                [170, 100, 80],
                [179, 255, 255],
            )

            if red1 is not None:
                detections["A"] = {
                    "name": "RED",
                    **red1,
                }

            elif red2 is not None:
                detections["A"] = {
                    "name": "RED",
                    **red2,
                }

            blue = self.detect_color(
                hsv,
                [100, 100, 80],
                [135, 255, 255],
            )

            if blue is not None:
                detections["B"] = {
                    "name": "BLUE",
                    **blue,
                }

            green = self.detect_color(
                hsv,
                [40, 70, 60],
                [85, 255, 255],
            )

            if green is not None:
                detections["C"] = {
                    "name": "GREEN",
                    **green,
                }

            output = String()
            output.data = json.dumps(detections)

            self.publisher.publish(output)

        except Exception as exc:

            self.get_logger().error(
                f"Detection error: {exc}"
            )


def main(args=None):

    rclpy.init(args=args)

    node = ObjectDetectionNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
