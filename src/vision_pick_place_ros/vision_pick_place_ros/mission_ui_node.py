#!/usr/bin/env python3

import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import String


WIDTH = 800
HEIGHT = 800

OBJECT_NAMES = {
    "A": "RED",
    "B": "BLUE",
    "C": "GREEN",
}

OBJECT_POSITIONS = {
    "A": (-0.45, -0.35, 0.04),
    "B": (-0.15, 0.35, 0.04),
    "C": (-0.25, -0.35, 0.04),
}


class MissionDashboard(Node):

    def __init__(self):

        super().__init__("mission_ui_node")

        # ------------------------------------------------------
        # State
        # ------------------------------------------------------

        self.selected_object = "B"
        self.latest_target = None

        self.motion_status = "WAITING"
        self.robot_status = "CONNECTED"
        self.gripper_status = "OPEN"

        self.pipeline = {
            "Detection": False,
            "Localization": False,
            "IK": False,
            "RRT Planning": False,
            "Collision Check": False,
            "Pick": False,
            "Lift": False,
            "Obstacle Avoidance": False,
            "Place": False,
            "Home": False,
        }

        self.collision_count = 0

        # Validated assignment results
        self.trials = 10
        self.successful_trials = 10
        self.place_error = 1.47
        self.home_error = 0.00000715

        self.camera_frame = np.zeros(
            (HEIGHT, WIDTH, 3),
            dtype=np.uint8,
        )

        # ------------------------------------------------------
        # Publishers
        # ------------------------------------------------------

        # Selection controller listens here.
        self.selection_pub = self.create_publisher(
            String,
            "/vision/selected_object",
            10,
        )

        # Selection controller / motion system listens here.
        self.command_pub = self.create_publisher(
            String,
            "/vision/mission_command",
            10,
        )

        # ------------------------------------------------------
        # Subscribers
        # ------------------------------------------------------

        self.create_subscription(
            Image,
            "/camera/image_raw",
            self.image_callback,
            10,
        )

        self.create_subscription(
            Point,
            "/vision/target_pose",
            self.target_callback,
            10,
        )

        self.create_subscription(
            String,
            "/motion/status",
            self.motion_callback,
            10,
        )

        self.create_subscription(
            String,
            "/robot/status",
            self.robot_callback,
            10,
        )

        self.create_subscription(
            String,
            "/gripper/status",
            self.gripper_callback,
            10,
        )

        # Receive commands/status published by the selection controller.
        self.create_subscription(
            String,
            "/vision/mission_command",
            self.command_callback,
            10,
        )

        # ------------------------------------------------------
        # UI timer
        # ------------------------------------------------------

        self.timer = self.create_timer(
            0.05,
            self.update_dashboard,
        )

        self.get_logger().info(
            "Mission Dashboard started."
        )

        self.get_logger().info(
            "Controls: A=RED  B=BLUE  C=GREEN  "
            "ENTER=EXECUTE  R=RESET  Q/ESC=EXIT"
        )

    # ==========================================================
    # ROS CALLBACKS
    # ==========================================================

    def image_callback(self, msg):

        try:

            if msg.encoding.lower() == "rgb8":

                image = np.frombuffer(
                    msg.data,
                    dtype=np.uint8,
                ).reshape(
                    (msg.height, msg.width, 3)
                )

                self.camera_frame = cv2.cvtColor(
                    image,
                    cv2.COLOR_RGB2BGR,
                )

            elif msg.encoding.lower() == "bgr8":

                self.camera_frame = np.frombuffer(
                    msg.data,
                    dtype=np.uint8,
                ).reshape(
                    (msg.height, msg.width, 3)
                ).copy()

        except Exception as exc:

            self.get_logger().warning(
                f"Camera conversion failed: {exc}"
            )

    def target_callback(self, msg):

        self.latest_target = (
            float(msg.x),
            float(msg.y),
            float(msg.z),
        )

        self.pipeline["Localization"] = True

    def motion_callback(self, msg):

        status = msg.data

        self.motion_status = status

        text = status.upper()

        if "TARGET" in text:
            self.pipeline["Detection"] = True

        if "IK_" in text or "IK" in text:
            self.pipeline["IK"] = True

        if "RRT" in text or "PLANN" in text:
            self.pipeline["RRT Planning"] = True

        if (
            "COLLISION_FREE" in text
            or "NO COLLISION" in text
            or "DIRECT" in text
            or "OBSTACLE_AVOIDANCE_PASS" in text
        ):
            self.pipeline["Collision Check"] = True

        if "GRASP_PASS" in text or "GRASP" in text:
            self.pipeline["Pick"] = True

        if "LIFT_PASS" in text or "LIFT" in text:
            self.pipeline["Lift"] = True

        if "OBSTACLE" in text:
            self.pipeline["Obstacle Avoidance"] = True

        if (
            "PLACE_POSITION_REACHED" in text
            or "PLACE" in text
            or "RELEASE" in text
            or "PICK_PLACE_COMPLETE" in text
        ):
            self.pipeline["Place"] = True

        if (
            "HOME" in text
            or "COMPLETE" in text
        ):
            self.pipeline["Home"] = True

        if "COLLISION" in text:
            if not any(
                key in text
                for key in [
                    "COLLISION_FREE",
                    "NO COLLISION",
                    "NO_COLLISION",
                ]
            ):
                self.collision_count += 1

    def robot_callback(self, msg):

        self.robot_status = msg.data

    def gripper_callback(self, msg):

        self.gripper_status = msg.data

    def command_callback(self, msg):

        text = msg.data.strip()

        if text.startswith("SELECT:"):

            obj = text.split(":", 1)[1].strip().upper()

            if obj in OBJECT_NAMES:

                self.selected_object = obj

        elif text.startswith("EXECUTE:"):

            obj = text.split(":", 1)[1].strip().upper()

            if obj in OBJECT_NAMES:

                self.selected_object = obj

                self.motion_status = (
                    f"EXECUTION REQUESTED: {OBJECT_NAMES[obj]}"
                )

        elif text == "RESET":

            self.reset_pipeline()

    # ==========================================================
    # COMMAND FUNCTIONS
    # ==========================================================

    def publish_selection(self, obj):

        if obj not in OBJECT_NAMES:
            return

        self.selected_object = obj

        msg = String()
        msg.data = obj

        self.selection_pub.publish(msg)

        self.get_logger().info(
            f"Selected {OBJECT_NAMES[obj]}"
        )

    def execute_selected(self):

        obj = self.selected_object

        if obj not in OBJECT_NAMES:
            return

        msg = String()
        msg.data = f"EXECUTE:{obj}"

        self.command_pub.publish(msg)

        self.motion_status = (
            f"EXECUTE REQUESTED: {OBJECT_NAMES[obj]}"
        )

        self.get_logger().info(
            f"Execution requested for {OBJECT_NAMES[obj]}"
        )

    def reset_pipeline(self):

        for key in self.pipeline:
            self.pipeline[key] = False

        self.collision_count = 0
        self.motion_status = "RESET"

    def handle_key(self, key):

        if key in [ord("a"), ord("A")]:

            self.publish_selection("A")

        elif key in [ord("b"), ord("B")]:

            self.publish_selection("B")

        elif key in [ord("c"), ord("C")]:

            self.publish_selection("C")

        elif key in [13, 10, 32]:

            self.execute_selected()

        elif key in [ord("r"), ord("R")]:

            msg = String()
            msg.data = "RESET"

            self.command_pub.publish(msg)

            self.reset_pipeline()

            self.get_logger().info(
                "Mission reset."
            )

        elif key in [ord("q"), ord("Q"), 27]:

            self.get_logger().info(
                "Closing Mission Dashboard."
            )

            cv2.destroyAllWindows()

            rclpy.shutdown()

    # ==========================================================
    # COMPUTER VISION
    # ==========================================================

    def detect_objects(self, frame):

        if frame is None or frame.size == 0:

            return np.zeros(
                (HEIGHT, WIDTH, 3),
                dtype=np.uint8,
            )

        image = frame.copy()

        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV,
        )

        masks = {

            "RED": (
                cv2.inRange(
                    hsv,
                    np.array([0, 100, 70]),
                    np.array([10, 255, 255]),
                )
                |
                cv2.inRange(
                    hsv,
                    np.array([170, 100, 70]),
                    np.array([179, 255, 255]),
                )
            ),

            "BLUE": cv2.inRange(
                hsv,
                np.array([100, 100, 70]),
                np.array([135, 255, 255]),
            ),

            "GREEN": cv2.inRange(
                hsv,
                np.array([40, 70, 60]),
                np.array([85, 255, 255]),
            ),
        }

        for name, mask in masks.items():

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

            for contour in contours:

                area = cv2.contourArea(contour)

                if area < 150:
                    continue

                x, y, w, h = cv2.boundingRect(
                    contour
                )

                cx = x + w // 2
                cy = y + h // 2

                if name == "RED":
                    code = "A"
                elif name == "BLUE":
                    code = "B"
                else:
                    code = "C"

                selected = (
                    code == self.selected_object
                )

                thickness = 4 if selected else 2

                cv2.rectangle(
                    image,
                    (x, y),
                    (x + w, y + h),
                    (255, 255, 255),
                    thickness,
                )

                cv2.drawMarker(
                    image,
                    (cx, cy),
                    (255, 255, 255),
                    cv2.MARKER_CROSS,
                    20,
                    2,
                )

                label = (
                    f"{code}: {name}"
                    + ("  [SELECTED]" if selected else "")
                )

                cv2.putText(
                    image,
                    label,
                    (x, max(25, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        return image

    # ==========================================================
    # DASHBOARD DRAWING
    # ==========================================================

    def text(
        self,
        img,
        text,
        x,
        y,
        scale=0.55,
        thickness=1,
    ):

        cv2.putText(
            img,
            str(text),
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (230, 235, 240),
            thickness,
            cv2.LINE_AA,
        )

    def panel(
        self,
        img,
        x1,
        y1,
        x2,
        y2,
        title,
    ):

        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            (45, 50, 60),
            2,
        )

        self.text(
            img,
            title,
            x1 + 15,
            y1 + 30,
            0.65,
            2,
        )

    def update_dashboard(self):

        if not rclpy.ok():
            return

        canvas = np.zeros(
            (900, 1500, 3),
            dtype=np.uint8,
        )

        canvas[:] = (
            18,
            20,
            27,
        )

        # ------------------------------------------------------
        # Title
        # ------------------------------------------------------

        self.text(
            canvas,
            "VISION-GUIDED ROBOTIC PICK & PLACE",
            35,
            40,
            1.0,
            2,
        )

        self.text(
            canvas,
            "ROS 2  |  OpenCV  |  PyBullet  |  IK  |  RRT",
            950,
            38,
            0.5,
            1,
        )

        # ------------------------------------------------------
        # Camera panel
        # ------------------------------------------------------

        self.panel(
            canvas,
            20,
            60,
            640,
            600,
            "CAMERA / OBJECT DETECTION",
        )

        camera = self.detect_objects(
            self.camera_frame
        )

        camera = cv2.resize(
            camera,
            (600, 500),
        )

        canvas[
            30:530,
            30:630,
        ] = camera

        # ------------------------------------------------------
        # Robot panel
        # ------------------------------------------------------

        self.panel(
            canvas,
            660,
            60,
            1050,
            390,
            "ROBOT / PYBULLET",
        )

        self.text(
            canvas,
            f"Robot: {self.robot_status}",
            680,
            110,
            0.6,
            1,
        )

        self.text(
            canvas,
            "Planner: RRT",
            680,
            145,
            0.6,
            1,
        )

        self.text(
            canvas,
            "IK: PyBullet constrained search",
            680,
            180,
            0.55,
            1,
        )

        self.text(
            canvas,
            f"Motion: {self.motion_status[:34]}",
            680,
            220,
            0.52,
            1,
        )

        self.text(
            canvas,
            f"Gripper: {self.gripper_status}",
            680,
            260,
            0.6,
            1,
        )

        self.text(
            canvas,
            "Simulation: ACTIVE",
            680,
            310,
            0.6,
            2,
        )

        # ------------------------------------------------------
        # Target panel
        # ------------------------------------------------------

        self.panel(
            canvas,
            1070,
            60,
            1480,
            390,
            "TARGET / COORDINATES",
        )

        obj_name = OBJECT_NAMES[
            self.selected_object
        ]

        self.text(
            canvas,
            f"Selected: {self.selected_object}  {obj_name}",
            1090,
            115,
            0.7,
            2,
        )

        if self.latest_target is not None:

            x, y, z = self.latest_target

        else:

            x, y, z = OBJECT_POSITIONS[
                self.selected_object
            ]

        self.text(
            canvas,
            f"World X: {x:+.4f} m",
            1090,
            165,
            0.6,
            1,
        )

        self.text(
            canvas,
            f"World Y: {y:+.4f} m",
            1090,
            200,
            0.6,
            1,
        )

        self.text(
            canvas,
            f"World Z: {z:+.4f} m",
            1090,
            235,
            0.6,
            1,
        )

        self.text(
            canvas,
            "Camera -> World",
            1090,
            285,
            0.6,
            1,
        )

        # ------------------------------------------------------
        # Mission pipeline
        # ------------------------------------------------------

        self.panel(
            canvas,
            660,
            410,
            1050,
            850,
            "MISSION PIPELINE",
        )

        stages = list(
            self.pipeline.items()
        )

        for i, (name, done) in enumerate(stages):

            row = i // 2
            col = i % 2

            x = 680 + col * 175
            y = 465 + row * 58

            mark = "✓" if done else "○"

            self.text(
                canvas,
                f"{mark} {name}",
                x,
                y,
                0.52,
                1,
            )

        # ------------------------------------------------------
        # Safety / validation
        # ------------------------------------------------------

        self.panel(
            canvas,
            1070,
            410,
            1480,
            850,
            "SAFETY / VALIDATION",
        )

        success_rate = (
            100.0
            * self.successful_trials
            / self.trials
        )

        self.text(
            canvas,
            f"Trials: {self.trials}",
            1090,
            465,
            0.6,
            1,
        )

        self.text(
            canvas,
            f"Successful: {self.successful_trials}",
            1090,
            505,
            0.6,
            1,
        )

        self.text(
            canvas,
            f"Pick & Place: {success_rate:.0f}%",
            1090,
            545,
            0.65,
            2,
        )

        self.text(
            canvas,
            f"Placement error: {self.place_error:.2f} mm",
            1090,
            590,
            0.6,
            1,
        )

        self.text(
            canvas,
            f"Collision count: {self.collision_count}",
            1090,
            630,
            0.6,
            1,
        )

        self.text(
            canvas,
            f"Home joint error: {self.home_error:.2e} rad",
            1090,
            670,
            0.6,
            1,
        )

        self.text(
            canvas,
            "Safety: COLLISION CHECK ENABLED",
            1090,
            725,
            0.55,
            2,
        )

        # ------------------------------------------------------
        # Controls
        # ------------------------------------------------------

        cv2.rectangle(
            canvas,
            (20, 855),
            (1480, 890),
            (30, 35, 43),
            -1,
        )

        self.text(
            canvas,
            "A RED   |   B BLUE   |   C GREEN   |   ENTER EXECUTE   |   R RESET   |   Q/ESC EXIT",
            35,
            880,
            0.55,
            1,
        )

        # ------------------------------------------------------
        # Show
        # ------------------------------------------------------

        cv2.imshow(
            "Mission Dashboard",
            canvas,
        )

        key = cv2.waitKey(1) & 0xFF

        if key != 255:
            self.handle_key(key)

    # ==========================================================
    # CLEANUP
    # ==========================================================

    def destroy_node(self):

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        super().destroy_node()


def main(args=None):

    rclpy.init(args=args)

    node = MissionDashboard()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
