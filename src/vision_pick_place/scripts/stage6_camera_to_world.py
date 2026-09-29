import math
import time

import cv2
import numpy as np
import pybullet as p
import pybullet_data


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_WIDTH = 800
IMAGE_HEIGHT = 800

CAMERA_POSITION = np.array(
    [0.0, 0.0, 2.5],
    dtype=np.float64
)

CAMERA_TARGET = np.array(
    [0.0, 0.0, 0.0],
    dtype=np.float64
)

CAMERA_UP = np.array(
    [0.0, 1.0, 0.0],
    dtype=np.float64
)

FOV = 60.0
NEAR = 0.01
FAR = 5.0

# ------------------------------------------------------------
# IMPORTANT
#
# Cylinder:
#   center position Z = 0.08 m
#   height            = 0.16 m
#
# The detected image centroid corresponds approximately to
# the projected center of the cylinder.
# Therefore we back-project to Z = 0.08 m.
# ------------------------------------------------------------

OBJECT_CENTER_Z = 0.08

TABLE_Z = 0.0


# ============================================================
# GROUND TRUTH
# ============================================================

GROUND_TRUTH = {

    "RED": np.array(
        [-0.45, -0.35, 0.08],
        dtype=np.float64
    ),

    "BLUE": np.array(
        [-0.15, 0.35, 0.08],
        dtype=np.float64
    ),

    "GREEN": np.array(
        [0.05, -0.35, 0.08],
        dtype=np.float64
    ),
}


# ============================================================
# HSV RANGES
# ============================================================

HSV_RANGES = {

    "RED": [
        (
            np.array([0, 100, 80]),
            np.array([10, 255, 255])
        ),
        (
            np.array([170, 100, 80]),
            np.array([179, 255, 255])
        ),
    ],

    "BLUE": [
        (
            np.array([100, 100, 80]),
            np.array([135, 255, 255])
        ),
    ],

    "GREEN": [
        (
            np.array([40, 70, 60]),
            np.array([85, 255, 255])
        ),
    ],
}


# ============================================================
# HOME POSITION
# ============================================================

HOME_POSITION = [
    0.0,
    -0.785398163,
    0.0,
    -2.356194490,
    0.0,
    1.570796327,
    0.785398163,
]


# ============================================================
# MATRIX CONVERSION
# ============================================================

def bullet_matrix_to_numpy(matrix):

    return np.array(
        matrix,
        dtype=np.float64
    ).reshape(
        (4, 4),
        order="F"
    )


# ============================================================
# CAMERA MATRICES
# ============================================================

def get_camera_matrices():

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=CAMERA_POSITION.tolist(),
        cameraTargetPosition=CAMERA_TARGET.tolist(),
        cameraUpVector=CAMERA_UP.tolist()
    )

    projection_matrix = p.computeProjectionMatrixFOV(
        fov=FOV,
        aspect=IMAGE_WIDTH / IMAGE_HEIGHT,
        nearVal=NEAR,
        farVal=FAR
    )

    return view_matrix, projection_matrix


# ============================================================
# WORLD → PIXEL
# ============================================================

def world_to_pixel(
    world_point,
    view_matrix,
    projection_matrix
):

    view = bullet_matrix_to_numpy(
        view_matrix
    )

    projection = bullet_matrix_to_numpy(
        projection_matrix
    )

    world_h = np.array(
        [
            world_point[0],
            world_point[1],
            world_point[2],
            1.0
        ],
        dtype=np.float64
    )

    camera_h = view @ world_h

    clip_h = projection @ camera_h

    if abs(clip_h[3]) < 1e-12:
        raise RuntimeError(
            "Invalid projection."
        )

    ndc = clip_h / clip_h[3]

    pixel_x = (
        (ndc[0] + 1.0)
        * 0.5
        * (IMAGE_WIDTH - 1)
    )

    pixel_y = (
        (1.0 - ndc[1])
        * 0.5
        * (IMAGE_HEIGHT - 1)
    )

    return np.array(
        [pixel_x, pixel_y],
        dtype=np.float64
    )


# ============================================================
# PIXEL → WORLD RAY
# ============================================================

def pixel_to_world_ray(
    pixel_x,
    pixel_y,
    view_matrix,
    projection_matrix
):

    # Pixel → NDC
    ndc_x = (
        2.0
        * pixel_x
        / (IMAGE_WIDTH - 1)
    ) - 1.0

    ndc_y = 1.0 - (
        2.0
        * pixel_y
        / (IMAGE_HEIGHT - 1)
    )

    near_ndc = np.array(
        [
            ndc_x,
            ndc_y,
            -1.0,
            1.0
        ],
        dtype=np.float64
    )

    far_ndc = np.array(
        [
            ndc_x,
            ndc_y,
            1.0,
            1.0
        ],
        dtype=np.float64
    )

    projection = bullet_matrix_to_numpy(
        projection_matrix
    )

    view = bullet_matrix_to_numpy(
        view_matrix
    )

    inverse_projection = np.linalg.inv(
        projection
    )

    inverse_view = np.linalg.inv(
        view
    )

    near_camera = (
        inverse_projection
        @ near_ndc
    )

    far_camera = (
        inverse_projection
        @ far_ndc
    )

    near_camera /= near_camera[3]
    far_camera /= far_camera[3]

    near_world = (
        inverse_view
        @ near_camera
    )

    far_world = (
        inverse_view
        @ far_camera
    )

    near_world /= near_world[3]
    far_world /= far_world[3]

    origin = near_world[:3]

    direction = (
        far_world[:3]
        - near_world[:3]
    )

    direction /= np.linalg.norm(
        direction
    )

    return origin, direction


# ============================================================
# RAY → HORIZONTAL PLANE
# ============================================================

def ray_plane_intersection(
    origin,
    direction,
    plane_z
):

    denominator = direction[2]

    if abs(denominator) < 1e-12:

        raise RuntimeError(
            "Ray is parallel to plane."
        )

    t = (
        plane_z - origin[2]
    ) / denominator

    if t <= 0:

        raise RuntimeError(
            "Invalid intersection."
        )

    point = (
        origin
        + t * direction
    )

    return point


# ============================================================
# PIXEL → OBJECT WORLD COORDINATE
# ============================================================

def pixel_to_object_world(
    pixel_x,
    pixel_y,
    view_matrix,
    projection_matrix
):

    origin, direction = pixel_to_world_ray(
        pixel_x,
        pixel_y,
        view_matrix,
        projection_matrix
    )

    return ray_plane_intersection(
        origin,
        direction,
        OBJECT_CENTER_Z
    )


# ============================================================
# WORLD → CAMERA
# ============================================================

def world_to_camera(
    world_point,
    view_matrix
):

    view = bullet_matrix_to_numpy(
        view_matrix
    )

    world_h = np.array(
        [
            world_point[0],
            world_point[1],
            world_point[2],
            1.0
        ],
        dtype=np.float64
    )

    camera_h = view @ world_h

    return camera_h[:3]


# ============================================================
# OBJECT DETECTION
# ============================================================

def detect_objects(frame):

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_RGB2HSV
    )

    detections = {}

    kernel = np.ones(
        (5, 5),
        dtype=np.uint8
    )

    for name, ranges in HSV_RANGES.items():

        mask = np.zeros(
            hsv.shape[:2],
            dtype=np.uint8
        )

        for lower, upper in ranges:

            current_mask = cv2.inRange(
                hsv,
                lower,
                upper
            )

            mask = cv2.bitwise_or(
                mask,
                current_mask
            )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        best_contour = None
        best_area = 0.0

        for contour in contours:

            area = cv2.contourArea(
                contour
            )

            if (
                area > 150
                and area > best_area
            ):

                best_area = area
                best_contour = contour

        if best_contour is None:
            continue

        # ----------------------------------------------------
        # Use enclosing-circle center AND contour centroid.
        #
        # The cylinder is circular/symmetric, so the enclosing
        # circle center is less affected by small segmentation
        # irregularities.
        # ----------------------------------------------------

        (circle_x, circle_y), radius = (
            cv2.minEnclosingCircle(
                best_contour
            )
        )

        moments = cv2.moments(
            best_contour
        )

        if moments["m00"] <= 0:
            continue

        centroid_x = (
            moments["m10"]
            / moments["m00"]
        )

        centroid_y = (
            moments["m01"]
            / moments["m00"]
        )

        # Average the two independent center estimates.
        pixel_x = (
            circle_x + centroid_x
        ) / 2.0

        pixel_y = (
            circle_y + centroid_y
        ) / 2.0

        detections[name] = {

            "pixel": np.array(
                [
                    pixel_x,
                    pixel_y
                ],
                dtype=np.float64
            ),

            "contour": best_contour,

            "area": best_area,

            "radius": radius
        }

    return detections


# ============================================================
# SCENE
# ============================================================

def create_scene():

    client = p.connect(
        p.GUI
    )

    p.setAdditionalSearchPath(
        pybullet_data.getDataPath()
    )

    p.setGravity(
        0,
        0,
        -9.81
    )

    # --------------------------------------------------------
    # TABLE
    # --------------------------------------------------------

    table_half = [
        1.2,
        1.0,
        0.05
    ]

    table_collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=table_half
    )

    table_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=table_half,
        rgbaColor=[
            0.75,
            0.75,
            0.75,
            1.0
        ]
    )

    p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=table_collision,
        baseVisualShapeIndex=table_visual,
        basePosition=[
            0,
            0,
            -0.05
        ]
    )

    # --------------------------------------------------------
    # PANDA
    # --------------------------------------------------------

    panda_id = p.loadURDF(
        "franka_panda/panda.urdf",
        basePosition=[
            0,
            0,
            0
        ],
        useFixedBase=True
    )

    for i, position in enumerate(
        HOME_POSITION
    ):

        p.resetJointState(
            panda_id,
            i,
            position
        )

    # --------------------------------------------------------
    # COLORED CYLINDERS
    # --------------------------------------------------------

    objects = {

        "RED": (
            [-0.45, -0.35, 0.08],
            [0.9, 0.1, 0.1, 1.0]
        ),

        "BLUE": (
            [-0.15, 0.35, 0.08],
            [0.1, 0.2, 0.95, 1.0]
        ),

        "GREEN": (
            [0.05, -0.35, 0.08],
            [0.1, 0.8, 0.1, 1.0]
        )
    }

    for position, color in objects.values():

        collision = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=0.055,
            height=0.16
        )

        visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.055,
            length=0.16,
            rgbaColor=color
        )

        p.createMultiBody(
            baseMass=0.05,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=position
        )

    # --------------------------------------------------------
    # WHITE OBSTACLE
    # --------------------------------------------------------

    obstacle_half = [
        0.10,
        0.10,
        0.15
    ]

    obstacle_collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=obstacle_half
    )

    obstacle_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=obstacle_half,
        rgbaColor=[
            1.0,
            1.0,
            1.0,
            1.0
        ]
    )

    p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=obstacle_collision,
        baseVisualShapeIndex=obstacle_visual,
        basePosition=[
            0.25,
            0.0,
            0.15
        ]
    )

    # --------------------------------------------------------
    # BLACK DROP PLATE
    # --------------------------------------------------------

    plate_collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[
            0.13,
            0.13,
            0.008
        ]
    )

    plate_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[
            0.13,
            0.13,
            0.008
        ],
        rgbaColor=[
            0.01,
            0.01,
            0.01,
            1.0
        ]
    )

    p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=plate_collision,
        baseVisualShapeIndex=plate_visual,
        basePosition=[
            0.55,
            0.40,
            0.008
        ]
    )

    return client, panda_id


# ============================================================
# CAMERA IMAGE
# ============================================================

def capture_camera(
    view_matrix,
    projection_matrix
):

    result = p.getCameraImage(
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )

    rgba = np.asarray(
        result[2],
        dtype=np.uint8
    ).reshape(
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        4
    )

    return rgba[:, :, :3]


# ============================================================
# ERROR
# ============================================================

def calculate_error(
    estimated,
    truth
):

    difference = (
        estimated - truth
    )

    dx = abs(difference[0])
    dy = abs(difference[1])
    dz = abs(difference[2])

    xy_error = math.sqrt(
        difference[0] ** 2
        + difference[1] ** 2
    )

    error_3d = np.linalg.norm(
        difference
    )

    return (
        dx,
        dy,
        dz,
        xy_error,
        error_3d
    )


# ============================================================
# MAIN
# ============================================================

def main():

    _, panda_id = create_scene()

    view_matrix, projection_matrix = (
        get_camera_matrices()
    )

    frame = capture_camera(
        view_matrix,
        projection_matrix
    )

    detections = detect_objects(
        frame
    )

    display = cv2.cvtColor(
        frame,
        cv2.COLOR_RGB2BGR
    )

    print()
    print("=" * 82)
    print("       HIGH ACCURACY CAMERA → WORLD LOCALIZATION")
    print("=" * 82)

    print()
    print("CAMERA")
    print("-" * 82)

    print(
        f"Position : "
        f"X={CAMERA_POSITION[0]:.6f}  "
        f"Y={CAMERA_POSITION[1]:.6f}  "
        f"Z={CAMERA_POSITION[2]:.6f} m"
    )

    print(
        f"Target   : "
        f"X={CAMERA_TARGET[0]:.6f}  "
        f"Y={CAMERA_TARGET[1]:.6f}  "
        f"Z={CAMERA_TARGET[2]:.6f} m"
    )

    print(
        f"Resolution : "
        f"{IMAGE_WIDTH} × {IMAGE_HEIGHT}"
    )

    print(
        f"FOV        : {FOV:.2f}°"
    )

    print(
        f"Object center plane : "
        f"Z={OBJECT_CENTER_Z:.4f} m"
    )

    print()
    print(
        "TRANSFORMATION"
    )
    print("-" * 82)

    print(
        "Pixel → Camera Ray → Object Center Plane "
        f"(Z={OBJECT_CENTER_Z:.2f} m) → World"
    )

    print()

    if not detections:

        print(
            "NO OBJECTS DETECTED."
        )

        p.disconnect()
        return

    xy_errors = []
    errors_3d = []

    for name in [
        "RED",
        "BLUE",
        "GREEN"
    ]:

        if name not in detections:

            print(
                f"{name}: NOT DETECTED"
            )

            continue

        pixel = detections[name]["pixel"]
        contour = detections[name]["contour"]

        pixel_x = pixel[0]
        pixel_y = pixel[1]

        # ----------------------------------------------------
        # CAMERA → WORLD
        # ----------------------------------------------------

        world_point = pixel_to_object_world(
            pixel_x,
            pixel_y,
            view_matrix,
            projection_matrix
        )

        # ----------------------------------------------------
        # CAMERA COORDINATE
        # ----------------------------------------------------

        camera_point = world_to_camera(
            world_point,
            view_matrix
        )

        # ----------------------------------------------------
        # TABLE CONTACT POINT
        #
        # Same X/Y as object center, Z=0.
        # ----------------------------------------------------

        table_point = np.array(
            [
                world_point[0],
                world_point[1],
                TABLE_Z
            ],
            dtype=np.float64
        )

        truth = GROUND_TRUTH[name]

        (
            dx,
            dy,
            dz,
            xy_error,
            error_3d
        ) = calculate_error(
            world_point,
            truth
        )

        xy_errors.append(
            xy_error
        )

        errors_3d.append(
            error_3d
        )

        # ----------------------------------------------------
        # TERMINAL REPORT
        # ----------------------------------------------------

        print("=" * 82)
        print(
            f"OBJECT: {name}"
        )
        print("=" * 82)

        print(
            f"Pixel centroid : "
            f"u={pixel_x:.4f}, "
            f"v={pixel_y:.4f} px"
        )

        print()
        print(
            "CAMERA FRAME"
        )

        print(
            f"Xc = {camera_point[0]: .9f} m"
        )

        print(
            f"Yc = {camera_point[1]: .9f} m"
        )

        print(
            f"Zc = {camera_point[2]: .9f} m"
        )

        print()
        print(
            "WORLD OBJECT CENTER"
        )

        print(
            f"Xw = {world_point[0]: .9f} m"
        )

        print(
            f"Yw = {world_point[1]: .9f} m"
        )

        print(
            f"Zw = {world_point[2]: .9f} m"
        )

        print()
        print(
            "TABLE CONTACT POINT"
        )

        print(
            f"X = {table_point[0]: .9f} m"
        )

        print(
            f"Y = {table_point[1]: .9f} m"
        )

        print(
            f"Z = {table_point[2]: .9f} m"
        )

        print()
        print(
            "GROUND TRUTH"
        )

        print(
            f"Xgt = {truth[0]: .9f} m"
        )

        print(
            f"Ygt = {truth[1]: .9f} m"
        )

        print(
            f"Zgt = {truth[2]: .9f} m"
        )

        print()
        print(
            "LOCALIZATION ERROR"
        )

        print(
            f"ΔX = {dx * 1000:.4f} mm"
        )

        print(
            f"ΔY = {dy * 1000:.4f} mm"
        )

        print(
            f"ΔZ = {dz * 1000:.4f} mm"
        )

        print(
            f"XY = {xy_error * 1000:.4f} mm"
        )

        print(
            f"3D = {error_3d * 1000:.4f} mm"
        )

        # ----------------------------------------------------
        # IMAGE DISPLAY
        # ----------------------------------------------------

        cv2.drawContours(
            display,
            [contour],
            -1,
            (255, 255, 255),
            2
        )

        cx = int(
            round(pixel_x)
        )

        cy = int(
            round(pixel_y)
        )

        cv2.drawMarker(
            display,
            (cx, cy),
            (255, 255, 255),
            cv2.MARKER_CROSS,
            24,
            2
        )

        tx = max(
            10,
            min(
                cx - 120,
                IMAGE_WIDTH - 390
            )
        )

        ty = max(
            35,
            cy - 55
        )

        cv2.putText(
            display,
            name,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            display,
            (
                f"X={world_point[0]:.3f} "
                f"Y={world_point[1]:.3f}"
            ),
            (tx, ty + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            display,
            (
                f"Error={xy_error * 1000:.2f} mm"
            ),
            (tx, ty + 47),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 82)
    print("LOCALIZATION SUMMARY")
    print("=" * 82)

    print(
        f"Objects detected : "
        f"{len(xy_errors)}/3"
    )

    if xy_errors:

        print(
            f"Average XY error : "
            f"{np.mean(xy_errors) * 1000:.6f} mm"
        )

        print(
            f"Maximum XY error : "
            f"{np.max(xy_errors) * 1000:.6f} mm"
        )

        print(
            f"Average 3D error : "
            f"{np.mean(errors_3d) * 1000:.6f} mm"
        )

        print(
            f"Maximum 3D error : "
            f"{np.max(errors_3d) * 1000:.6f} mm"
        )

    print("=" * 82)

    cv2.imshow(
        "Stage 6 - High Accuracy Localization",
        display
    )

    print()
    print(
        "Press Q or ESC to close."
    )

    while True:

        key = cv2.waitKey(1) & 0xFF

        if key in [
            ord("q"),
            ord("Q"),
            27
        ]:
            break

        time.sleep(0.01)

    cv2.destroyAllWindows()
    p.disconnect()


if __name__ == "__main__":
    main()
