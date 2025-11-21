#!/usr/bin/env python3

import math
import threading
import heapq
import sys

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from std_msgs.msg import Float32MultiArray
from pynput.keyboard import Key, Listener
import tf2_ros

# =========================
# ROBOT + CONTROL CONSTANTS
# =========================

ROBOT_r = 0.0508    # wheel radius [m]
ROBOT_LENGTH = 0.127  # wheel separation [m]

DIST_ERROR_MARGIN = 0.01    # [m] distance threshold to consider waypoint reached
ORIENT_ERROR_MARGIN = 0.001  # [rad] (not heavily used here)
DT = 0.1   # control update step [s]

KP_OMEGA = 4.0        # P gain for heading
KD_OMEGA = 0.0        # D gain for heading
KP_V = 0.5            # P gain for linear velocity

LAST_THETA_ERROR = 0.0

# =========================
# WORLD / MAP GEOMETRY
# =========================
# All coordinates are in meters.
#
# Your landmarks and points were given in centimeters, so:
#   117 cm -> 1.17 m, etc.
#
# World bounds: choose slightly larger than furthest tag position.
# Max magnitude ≈ 1.17 m, so use ±1.2 m.

WORLD_X_MIN = -1.2
WORLD_X_MAX =  1.2
WORLD_Y_MIN = -1.2
WORLD_Y_MAX =  1.2

WORLD_W = WORLD_X_MAX - WORLD_X_MIN
WORLD_H = WORLD_Y_MAX - WORLD_Y_MIN

GRID_RES = 0.05  # 5 cm per cell

# Obstacle: 35 x 35 cm centered at (0, 0) => 0.35 x 0.35 m
# Rectangle: [-0.175, 0.175] x [-0.175, 0.175]
OBSTACLES = [
    (-0.175, -0.175, 0.175, 0.175)  # (xmin, ymin, xmax, ymax)
]

ROBOT_SAFETY_MARGIN = 0.10  # additional clearance around robot [m]

# Safety weighting for cost-map (larger => stronger push away from obstacle)
SAFETY_WEIGHT = 1.0

# =========================
# LANDMARK DEFINITIONS
# (Not directly used in planner, but here for completeness)
# Positions in meters, headings in degrees
# =========================

LANDMARKS = {
    # Obstacle landmarks (tags 8-11), centered around obstacle at (0,0)
    8: {"x": -0.17, "y":  0.00, "theta_deg": 180.0},
    9: {"x":  0.00, "y": -0.17, "theta_deg": -90.0},
   10: {"x":  0.17, "y":  0.00, "theta_deg":   0.0},
   11: {"x":  0.00, "y":  0.17, "theta_deg":  90.0},

    # Outer landmarks (tags 0-7)
    0: {"x":  1.17, "y":  0.85, "theta_deg":  90.0},
    1: {"x": -0.85, "y":  1.17, "theta_deg": -90.0},
    2: {"x":  0.85, "y": -1.17, "theta_deg":  90.0},
    3: {"x":  1.17, "y": -0.85, "theta_deg": 180.0},
    4: {"x": -0.85, "y": -1.17, "theta_deg":  90.0},
    5: {"x": -1.17, "y": -0.85, "theta_deg":   0.0},
    6: {"x": -1.17, "y":  0.85, "theta_deg":   0.0},
    7: {"x":  0.85, "y":  1.17, "theta_deg": -90.0},
}

# Start and goal points in meters
START_WORLD = (-0.85, -0.85)  # (-85 cm, -85 cm)
GOAL_WORLD  = ( 0.85,  0.85)  # (85 cm, 85 cm)

# =========================
# UTILITY FUNCTIONS
# =========================

def normalize_angle(angle):
    """Normalize angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def get_wheel_speeds(v_req, omega_req):
    """Differential drive equations for each side's angular velocity."""
    phi_dot_R = (v_req + (omega_req * ROBOT_LENGTH) / 2.0) / ROBOT_r
    phi_dot_L = (v_req - (omega_req * ROBOT_LENGTH) / 2.0) / ROBOT_r
    return phi_dot_L, phi_dot_R


def get_control_signals(robot_x, robot_y, robot_theta, target_x, target_y, dt):
    """Compute (v, omega) to drive the robot towards a target (x, y)."""
    global LAST_THETA_ERROR

    dx = target_x - robot_x
    dy = target_y - robot_y
    dist_error = math.hypot(dx, dy)

    v_req = KP_V * dist_error
    v_req = min(v_req, 0.5)  # conservative speed cap

    theta_to_target = math.atan2(dy, dx)
    theta_error = normalize_angle(theta_to_target - robot_theta)

    P_term = KP_OMEGA * theta_error
    derivative_error = (theta_error - LAST_THETA_ERROR) / dt
    D_term = KD_OMEGA * derivative_error

    omega_req = P_term + D_term
    LAST_THETA_ERROR = theta_error

    return v_req, omega_req, dist_error


def cap_angular_speed(w):
    """Cap the maximum absolute speed to 1 rad/s and minimum absolute speed to 0.01 rad/s."""
    return min(1.0, max(abs(w), 0.01))


def quat_to_yaw(qx, qy, qz, qw):
    """Convert quaternion to yaw (rotation around Z)."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def calibrator(L, R, lateral_offset=0.0):
    """
    Calibrate raw wheel speeds and apply pose-based lateral (cross-track) correction.

    `lateral_offset` is the signed distance [m] from the robot to the current path
    segment. Positive offset means the robot is on one side and should yaw towards
    the negative side to come back to the line.
    """
    # Setup constants for each wheel set's angular velocity to scale according
    # to the environment
    K_L = 0.09
    K_R = 0.10
    M = 1.1

    # Pose-based lateral correction for straight-line deviation
    KP_CORRECTION = 15.0  # tune as needed
    omega_correction = -KP_CORRECTION * lateral_offset

    if L > 0 and R > 0:
        # Moving forward - apply straightness correction
        L += omega_correction / 2.0
        R -= omega_correction / 2.0
        if L < M and R < M:
            pass
        elif abs(R - L) < 0.01:
            K_L = 1.15 * K_L
    else:
        # Turning in place or reverse: keep original style
        K_L = 1.7
        K_R = 1.7
        if R > 0 and L < 0:
            L = -cap_angular_speed(L)
            R = cap_angular_speed(R)
        elif R < 0 and L > 0:
            L = cap_angular_speed(L)
            R = -cap_angular_speed(R)

    L *= K_L
    R *= K_R
    print("lateral_offset:", lateral_offset)
    print("L:", L, "R:", R)
    return L, R

# =========================
# CONFIGURATION SPACE / GRID HELPERS
# =========================

def world_to_grid(x, y):
    """Convert world coordinates (m) to integer grid indices (i, j)."""
    i = int((x - WORLD_X_MIN) / GRID_RES)
    j = int((y - WORLD_Y_MIN) / GRID_RES)
    return i, j


def grid_to_world(i, j):
    """Convert grid indices (i, j) back to world coordinates (m)."""
    x = WORLD_X_MIN + (i + 0.5) * GRID_RES
    y = WORLD_Y_MIN + (j + 0.5) * GRID_RES
    return x, y


def point_to_rect_distance(x, y, rect):
    """Distance from a point to an axis-aligned rectangle (xmin, ymin, xmax, ymax)."""
    xmin, ymin, xmax, ymax = rect
    dx = max(xmin - x, 0.0, x - xmax)
    dy = max(ymin - y, 0.0, y - ymax)
    return math.hypot(dx, dy)


def build_cspace_grid():
    """
    Build a configuration-space occupancy grid with inflated obstacles.

    Returns:
        grid: 2D list [W][H] where True = occupied, False = free.
        dist_map: 2D list of distance to nearest obstacle (for safety cost).
    """
    grid_w = int(WORLD_W / GRID_RES)
    grid_h = int(WORLD_H / GRID_RES)

    grid = [[False for _ in range(grid_h)] for _ in range(grid_w)]
    dist_map = [[float("inf") for _ in range(grid_h)] for _ in range(grid_w)]

    inflate_radius = ROBOT_r + ROBOT_SAFETY_MARGIN

    for i in range(grid_w):
        for j in range(grid_h):
            x, y = grid_to_world(i, j)

            # Bounds check (should be inside by construction, but keep safe)
            if not (WORLD_X_MIN <= x <= WORLD_X_MAX and WORLD_Y_MIN <= y <= WORLD_Y_MAX):
                grid[i][j] = True
                dist_map[i][j] = 0.0
                continue

            # Compute distance to obstacles
            min_d = float("inf")
            occupied = False
            for rect in OBSTACLES:
                d = point_to_rect_distance(x, y, rect)
                if d <= inflate_radius:
                    occupied = True
                if d < min_d:
                    min_d = d

            grid[i][j] = occupied
            dist_map[i][j] = min_d if not occupied else 0.0

    return grid, dist_map


def astar_search(grid, dist_map, start_world, goal_world):
    """
    Run A* on the inflated C-space grid with a safety-aware cost map.

    Args:
        grid: occupancy grid from build_cspace_grid()
        dist_map: distance to nearest obstacle per cell
        start_world: (x, y) in meters
        goal_world: (x, y) in meters

    Returns:
        path_world: list of (x, y) waypoints in world coordinates.
    """
    grid_w = len(grid)
    grid_h = len(grid[0])

    start_i, start_j = world_to_grid(*start_world)
    goal_i, goal_j = world_to_grid(*goal_world)

    def in_bounds(i, j):
        return 0 <= i < grid_w and 0 <= j < grid_h

    def is_free(i, j):
        return in_bounds(i, j) and not grid[i][j]

    # 8-connected neighbors
    neighbors_8 = [
        (-1, 0), (1, 0),  (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    ]

    open_heap = []
    heapq.heappush(open_heap, (0.0, (start_i, start_j)))
    came_from = {}
    g_score = {(start_i, start_j): 0.0}

    def heuristic(i, j):
        # Euclidean distance in world units
        x, y = grid_to_world(i, j)
        gx, gy = grid_to_world(goal_i, goal_j)
        return math.hypot(x - gx, y - gy)

    while open_heap:
        _, (ci, cj) = heapq.heappop(open_heap)

        if (ci, cj) == (goal_i, goal_j):
            # Reconstruct path
            path_cells = [(ci, cj)]
            while (ci, cj) in came_from:
                ci, cj = came_from[(ci, cj)]
                path_cells.append((ci, cj))
            path_cells.reverse()
            return [grid_to_world(i, j) for (i, j) in path_cells]

        for di, dj in neighbors_8:
            ni, nj = ci + di, cj + dj
            if not is_free(ni, nj):
                continue

            step_dist = math.hypot(di, dj) * GRID_RES
            base_cost = step_dist

            # Safety cost: penalize being close to obstacle
            d_to_obs = dist_map[ni][nj]
            safety_cost = 0.0
            if d_to_obs > 0.0:
                safety_cost = SAFETY_WEIGHT / (d_to_obs + 1e-3)

            step_cost = base_cost * (1.0 + safety_cost)
            tentative_g = g_score[(ci, cj)] + step_cost

            if (ni, nj) not in g_score or tentative_g < g_score[(ni, nj)]:
                g_score[(ni, nj)] = tentative_g
                f_score = tentative_g + heuristic(ni, nj)
                heapq.heappush(open_heap, (f_score, (ni, nj)))
                came_from[(ni, nj)] = (ci, cj)

    raise RuntimeError("A* failed to find a path in the current C-space grid.")


def plan_safe_path():
    """Plan a safe path using A* C-space planner.

    Uses START_WORLD and GOAL_WORLD.
    Returns:
        waypoints: list of (x, y, theta) along the path.
    """
    grid, dist_map = build_cspace_grid()
    path_xy = astar_search(grid, dist_map, START_WORLD, GOAL_WORLD)

    waypoints = []
    for idx, (x, y) in enumerate(path_xy):
        if idx < len(path_xy) - 1:
            nx, ny = path_xy[idx + 1]
            theta = math.atan2(ny - y, nx - x)
        else:
            # For final waypoint, keep same heading as previous if possible
            if len(path_xy) >= 2:
                px, py = path_xy[-2]
                theta = math.atan2(y - py, x - px)
            else:
                theta = 0.0
        waypoints.append((x, y, theta))

    # Optional: downsample to reduce number of waypoints
    DOWNSAMPLE = 2
    if len(waypoints) > 2:
        waypoints = waypoints[::DOWNSAMPLE]
        # ensure final goal is included
        gx, gy = path_xy[-1]
        if len(path_xy) >= 2:
            px, py = path_xy[-2]
            gtheta = math.atan2(gy - py, gx - px)
        else:
            gtheta = 0.0
        if (gx, gy) != (waypoints[-1][0], waypoints[-1][1]):
            waypoints.append((gx, gy, gtheta))

    return waypoints


# =========================
# KEYBOARD + PATH-FOLLOWING NODE
# =========================

class KeyboardControllerNode(Node):
    """
    ROS2 node that follows a safe A* path using pose-based feedback.

    - Uses configuration-space A* planner (plan_safe_path) to compute waypoints.
    - At runtime, reads robot pose from TF (odom -> base_link).
    - Computes (v, omega) using get_control_signals and converts to wheel speeds.
    - Applies pose-based lateral correction via `calibrator`.
    - Publishes [L, R] wheel commands on topic 'motor_commands'.
    """

    def __init__(self):
        super().__init__('keyboard_controller_node')

        # Publisher for wheel angular velocities [L, R]
        self.publisher = self.create_publisher(
            Float32MultiArray,
            'motor_commands',
            10
        )

        # TF2 listener for robot pose (expects an odom -> base_link transform)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.last_robot_pose = None  # (x, y, theta)
        self.tf_warned_once = False

        # Plan a safe global path using A* in configuration space
        try:
            self.waypoints = plan_safe_path()  # list of (x, y, theta)
            self.get_logger().info(
                f"Planned {len(self.waypoints)} waypoints using A* C-space planner."
            )
        except Exception as e:
            self.get_logger().error(f"Failed to plan A* path: {e}")
            self.waypoints = []
        self.current_wp_idx = 0

        # Keyboard control / state
        self.pressed_keys = set()
        self.current_L = 0.0
        self.current_R = 0.0
        self.running = True

        # Start keyboard listener (ESC to quit, X to stop)
        self.start_keyboard_listener()
        self.print_instructions()

        # Timer to publish commands at 20 Hz
        self.timer = self.create_timer(0.05, self.publish_motor_commands)

    # -----------------------------
    # Pose + path helpers
    # -----------------------------
    def get_robot_pose_from_tf(self):
        """Get robot pose (x, y, theta) from TF (odom -> base_link).

        If your TF tree uses different frames, change 'odom' and 'base_link' here.
        """
        try:
            transform = self.tf_buffer.lookup_transform(
                'odom',        # target frame (world-like)
                'base_link',   # source frame (robot)
                rclpy.time.Time(),
                timeout=Duration(seconds=0.1)
            )
            t = transform.transform.translation
            q = transform.transform.rotation
            x = t.x
            y = t.y
            theta = quat_to_yaw(q.x, q.y, q.z, q.w)
            self.last_robot_pose = (x, y, theta)
            return x, y, theta
        except Exception as e:
            if self.last_robot_pose is not None:
                if not self.tf_warned_once:
                    self.get_logger().warn(
                        f"TF lookup failed, using last pose once: {e}"
                    )
                    self.tf_warned_once = True
                return self.last_robot_pose
            else:
                if not self.tf_warned_once:
                    self.get_logger().warn(
                        f"TF lookup failed and no last pose available. "
                        f"Is your TF publisher (EKF/Apriltag node) running?"
                    )
                    self.tf_warned_once = True
                return None

    def compute_lateral_offset(self, robot_x, robot_y):
        """
        Compute pose-based cross-track error to the current path segment.

        Lateral offset is the signed distance from the robot
        to the line segment joining the previous and current waypoints.
        """
        if not self.waypoints or self.current_wp_idx == 0:
            return 0.0

        # Previous and current waypoint
        px, py, _ = self.waypoints[self.current_wp_idx - 1]
        tx, ty, _ = self.waypoints[self.current_wp_idx]

        dx = tx - px
        dy = ty - py
        seg_len = math.hypot(dx, dy)
        if seg_len < 1e-6:
            return 0.0

        # Vector from previous waypoint to robot
        rx = robot_x - px
        ry = robot_y - py

        # 2D cross product scaled by segment length gives signed distance
        cross = dx * ry - dy * rx
        lateral_offset = cross / seg_len
        return lateral_offset

    # -----------------------------
    # Keyboard handling
    # -----------------------------
    def start_keyboard_listener(self):
        """Start keyboard listener in a separate thread."""
        def listener_thread():
            with Listener(on_press=self.on_press,
                          on_release=self.on_release,
                          suppress=True) as listener:
                listener.join()

        self.listener_thread = threading.Thread(
            target=listener_thread,
            daemon=True
        )
        self.listener_thread.start()
        self.get_logger().info('Keyboard listener started')

    def print_instructions(self):
        instructions = f"""
========================================
   🤖  Safe Path Follower (A* + C-space)
========================================
- Uses configuration-space A* (inflated 35x35 cm obstacle at (0,0))
- Start: {START_WORLD} m
- Goal : {GOAL_WORLD} m
- Tags 8-11 surround the obstacle and assist localization.

Keyboard:
   X : Stop (send zero wheel speeds, keep node alive)
  ESC: Quit node (force exit)

Publishing motor commands on topic: motor_commands
========================================
"""
        self.get_logger().info(instructions)

    def on_press(self, key):
        """Keyboard press callback."""
        try:
            if hasattr(key, "char") and key.char:
                c = key.char.lower()
                self.pressed_keys.add(c)
                if c == 'x':
                    # Emergency stop (but node keeps running)
                    self.current_L = 0.0
                    self.current_R = 0.0
                    self.running = False
                    self.get_logger().info("X pressed: stopping robot (node still running).")
        except AttributeError:
            # Special keys
            if key == Key.esc:
                # Forcefully exit so the shell is released even if ROS shutdown hangs.
                self.get_logger().info("ESC pressed: exiting program.")
                sys.exit(0)

    def on_release(self, key):
        """Keyboard release callback (unused, but kept for completeness)."""
        try:
            if hasattr(key, "char") and key.char:
                c = key.char.lower()
                if c in self.pressed_keys:
                    self.pressed_keys.remove(c)
        except AttributeError:
            pass

    # -----------------------------
    # Core control loop
    # -----------------------------
    def publish_motor_commands(self):
        """Compute and publish wheel commands towards the next waypoint."""
        if not rclpy.ok():
            return

        # If globally stopped by 'x', keep publishing zeros
        if not self.running:
            msg = Float32MultiArray()
            msg.data = [0.0, 0.0]
            self.publisher.publish(msg)
            return

        if not self.waypoints or self.current_wp_idx >= len(self.waypoints):
            # Path finished or missing: send zero command
            msg = Float32MultiArray()
            msg.data = [0.0, 0.0]
            self.publisher.publish(msg)
            return

        pose = self.get_robot_pose_from_tf()
        if pose is None:
            # Cannot control without any pose estimate
            msg = Float32MultiArray()
            msg.data = [0.0, 0.0]
            self.publisher.publish(msg)
            return

        robot_x, robot_y, robot_theta = pose

        # Current target waypoint
        target_x, target_y, _ = self.waypoints[self.current_wp_idx]

        # Compute control based on pose error
        v_req, omega_req, dist_error = get_control_signals(
            robot_x, robot_y, robot_theta,
            target_x, target_y,
            DT
        )

        # If close enough, move to the next waypoint
        if dist_error < DIST_ERROR_MARGIN:
            self.current_wp_idx += 1
            if self.current_wp_idx >= len(self.waypoints):
                self.get_logger().info("Reached final waypoint. Stopping robot.")
                msg = Float32MultiArray()
                msg.data = [0.0, 0.0]
                self.publisher.publish(msg)
                self.running = False
                return
            else:
                return  # new target will be used in the next tick

        # Convert body velocity to wheel speeds
        L, R = get_wheel_speeds(v_req, omega_req)

        # Pose-based lateral correction relative to path segment
        lateral_offset = self.compute_lateral_offset(robot_x, robot_y)
        L, R = calibrator(L, R, lateral_offset)

        # Publish command
        msg = Float32MultiArray()
        msg.data = [L, R]
        self.current_L = L
        self.current_R = R
        self.publisher.publish(msg)

        # Optional logging
        self.get_logger().info(
            f"WP {self.current_wp_idx}/{len(self.waypoints)-1} | "
            f"pose=({robot_x:.2f}, {robot_y:.2f}, {robot_theta:.2f}) | "
            f"target=({target_x:.2f}, {target_y:.2f}) | "
            f"v={v_req:.2f}, w={omega_req:.2f}, "
            f"L={L:.2f}, R={R:.2f}, lateral={lateral_offset:.3f}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # We don't explicitly destroy node / shutdown because ESC uses sys.exit.
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass
        return 0


if __name__ == '__main__':
    main()

