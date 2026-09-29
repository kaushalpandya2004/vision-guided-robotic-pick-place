#!/usr/bin/env python3

import cv2
import numpy as np
import pybullet as p
import pybullet_data

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image


IMAGE_WIDTH = 800
IMAGE_HEIGHT = 800

CAMERA_POSITION = [0.0, 0.0, 2.5]
CAMERA_TARGET = [0.0, 0.0, 0.0]
CAMERA_UP = [0.0, 1.0, 0.0]

FOV = 60.0
NEAR = 0.01
FAR = 5.0


OBJECTS = {
    "A": {
        "name": "RED",
        "position": [-0.45, -0.35, 0.04],
        "radius": 0.04,
        "height": 0.08,
        "color": [1.0, 0.0, 0.0, 1.0],
    },
    "B": {
        "name": "BLUE",
        "position": [-0.15, 0.35, 0.04],
        "radius": 0.04,
        "height": 0.08,
        "color": [0.0, 0.2, 1.0, 1.0],
    },
    "C": {
        "name": "GREEN",
        "position": [-0.25, -0.35, 0.04],
        "radius": 0.04,
        "height": 0.08,
        "color": [0.0, 0.8, 0.1, 1.0],
    },
}


class CameraNode(Node):

    def __init__(self):

        super().__init__("camera_node")

        self.publisher = self.create_publisher(
            Image,
            "/camera/image_raw",
            10,
        )

        self.client = p.connect(p.DIRECT)

        if self.client < 0:
            raise RuntimeError(
                "PyBullet camera connection failed."
            )

        p.setAdditionalSearchPath(
            pybullet_data.getDataPath()
        )

        p.setGravity(
            0,
            0,
            -9.81,
        )

        self.create_scene()

        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=CAMERA_POSITION,
            cameraTargetPosition=CAMERA_TARGET,
            cameraUpVector=CAMERA_UP,
        )

        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=FOV,
            aspect=float(IMAGE_WIDTH) / float(IMAGE_HEIGHT),
            nearVal=NEAR,
            farVal=FAR,
        )

        # Separate camera/detection UI.
        self.window_name = (
            "VISION CAMERA - OBJECT DETECTION"
        )

        cv2.namedWindow(
            self.window_name,
            cv2.WINDOW_NORMAL,
        )

        cv2.resizeWindow(
            self.window_name,
            900,
            900,
        )

        self.frame_count = 0

        self.timer = self.create_timer(
            0.10,
            self.publish_image,
        )

        self.get_logger().info(
            "Real PyBullet camera node started."
        )

        self.get_logger().info(
            "Camera detection UI opened."
        )

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

        # Ground/table.
        self.create_box(
            [1.2, 1.0, 0.05],
            [0.0, 0.0, -0.05],
            [0.55, 0.55, 0.55, 1.0],
        )

        # Colored objects.
        for data in OBJECTS.values():

            collision = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=data["radius"],
                height=data["height"],
            )

            visual = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=data["radius"],
                length=data["height"],
                rgbaColor=data["color"],
            )

            p.createMultiBody(
                baseMass=0.05,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=data["position"],
            )

        # White obstacle.
        self.create_box(
            [0.08, 0.08, 0.22],
            [0.28502324, -0.19519229, 0.22],
            [1.0, 1.0, 1.0, 1.0],
        )

        # Black drop plate.
        self.create_box(
            [0.13, 0.13, 0.008],
            [0.55, 0.40, 0.008],
            [0.02, 0.02, 0.02, 1.0],
        )

        # Panda for camera visual realism.
        p.loadURDF(
            "franka_panda/panda.urdf",
            [0.0, 0.0, 0.0],
            useFixedBase=True,
        )

        for _ in range(60):
            p.stepSimulation()

    def detect_objects(self, frame):

        hsv = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2HSV,
        )

        masks = {}

        # RED.
        red1 = cv2.inRange(
            hsv,
            np.array([0, 100, 80]),
            np.array([10, 255, 255]),
        )

        red2 = cv2.inRange(
            hsv,
            np.array([170, 100, 80]),
            np.array([179, 255, 255]),
        )

        masks["A"] = cv2.bitwise_or(
            red1,
            red2,
        )

        # BLUE.
        masks["B"] = cv2.inRange(
            hsv,
            np.array([100, 80, 50]),
            np.array([135, 255, 255]),
        )

        # GREEN.
        masks["C"] = cv2.inRange(
            hsv,
            np.array([40, 70, 50]),
            np.array([85, 255, 255]),
        )

        detections = {}

        kernel = np.ones(
            (5, 5),
            np.uint8,
        )

        for key, mask in masks.items():

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

                area = cv2.contourArea(
                    contour
                )

                if area < 150:
                    continue

                if area > best_area:
                    best_area = area
                    best = contour

            if best is None:
                continue

            x, y, w, h = cv2.boundingRect(
                best
            )

            moments = cv2.moments(
                best
            )

            if moments["m00"] == 0:
                continue

            cx = int(
                moments["m10"]
                / moments["m00"]
            )

            cy = int(
                moments["m01"]
                / moments["m00"]
            )

            detections[key] = {
                "name": OBJECTS[key]["name"],
                "bbox": (x, y, w, h),
                "center": (cx, cy),
                "area": best_area,
            }

        return detections

    def draw_detection_ui(
        self,
        frame,
        detections,
    ):

        display = frame.copy()

        # Camera title.
        cv2.rectangle(
            display,
            (0, 0),
            (IMAGE_WIDTH, 58),
            (20, 20, 20),
            -1,
        )

        cv2.putText(
            display,
            "VISION CAMERA | HSV OBJECT DETECTION",
            (20, 37),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Object colors in BGR for drawing.
        draw_colors = {
            "A": (0, 0, 255),
            "B": (255, 80, 0),
            "C": (0, 220, 0),
        }

        for key, data in detections.items():

            x, y, w, h = data["bbox"]
            cx, cy = data["center"]

            color = draw_colors[key]

            # Bounding box.
            cv2.rectangle(
                display,
                (x, y),
                (x + w, y + h),
                color,
                3,
            )

            # Center crosshair.
            cv2.line(
                display,
                (cx - 12, cy),
                (cx + 12, cy),
                color,
                2,
            )

            cv2.line(
                display,
                (cx, cy - 12),
                (cx, cy + 12),
                color,
                2,
            )

            # Label.
            label = (
                f"{key}: {data['name']} "
                f"({cx}, {cy})"
            )

            label_y = max(
                85,
                y - 10,
            )

            cv2.putText(
                display,
                label,
                (x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )

            # Area.
            area_text = (
                f"Area: {data['area']:.0f} px"
            )

            cv2.putText(
                display,
                area_text,
                (x, y + h + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        # Status panel.
        panel_x = 15
        panel_y = IMAGE_HEIGHT - 125

        cv2.rectangle(
            display,
            (
                panel_x,
                panel_y,
            ),
            (
                IMAGE_WIDTH - 15,
                IMAGE_HEIGHT - 15,
            ),
            (20, 20, 20),
            -1,
        )

        count = len(detections)

        status = (
            f"Objects detected: {count}/3"
        )

        cv2.putText(
            display,
            status,
            (30, panel_y + 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        detected_names = ", ".join(
            data["name"]
            for data in detections.values()
        )

        if not detected_names:
            detected_names = "NONE"

        cv2.putText(
            display,
            f"Detected: {detected_names}",
            (30, panel_y + 63),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            display,
            "Press Q or ESC to close camera UI",
            (30, panel_y + 94),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        return display

    def publish_image(self):

        result = p.getCameraImage(
            IMAGE_WIDTH,
            IMAGE_HEIGHT,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.projection_matrix,
            renderer=p.ER_TINY_RENDERER,
        )

        rgba = np.asarray(
            result[2],
            dtype=np.uint8,
        ).reshape(
            (IMAGE_HEIGHT, IMAGE_WIDTH, 4)
        )

        # PyBullet gives RGB.
        rgb = rgba[:, :, :3]

        # ROS image.
        msg = Image()

        msg.header.stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )

        msg.header.frame_id = (
            "camera_frame"
        )

        msg.height = IMAGE_HEIGHT
        msg.width = IMAGE_WIDTH
        msg.encoding = "rgb8"
        msg.is_bigendian = False
        msg.step = IMAGE_WIDTH * 3
        msg.data = rgb.tobytes()

        self.publisher.publish(msg)

        # OpenCV uses BGR.
        frame = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2BGR,
        )

        detections = self.detect_objects(
            frame
        )

        display = self.draw_detection_ui(
            frame,
            detections,
        )

        cv2.imshow(
            self.window_name,
            display,
        )

        key = cv2.waitKey(1) & 0xFF

        if key in (
            ord("q"),
            27,
        ):

            self.get_logger().info(
                "Camera UI closed by user."
            )

            self.destroy_node()

            return

    def destroy_node(self):

        try:
            cv2.destroyWindow(
                self.window_name
            )

            cv2.destroyAllWindows()

        except Exception:
            pass

        if self.client >= 0:
            p.disconnect(
                self.client
            )

        super().destroy_node()


def main(args=None):

    rclpy.init(
        args=args
    )

    node = CameraNode()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:

        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
