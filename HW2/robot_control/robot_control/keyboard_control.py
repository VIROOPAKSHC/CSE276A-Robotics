import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from pynput.keyboard import Key, Listener
import threading
import tf2_ros
import matplotlib.pyplot as plt

ROBOT_r = 0.0508    # radius of the wheel in meters
ROBOT_LENGTH = 0.127  # distance between the wheels in meters
DIST_ERROR_MARGIN = 0.01  # distance error margin to consider waypoint reached
ORIENT_ERROR_MARGIN = 0.001  # ORIENTATION error margin to consider orientation reached  
DT = 0.1   # time step for each sub-simulation or the sub-segment simulation in seconds              

KP_OMEGA = 4 # constant for Proportionality for the BODY's angular velocity     
KD_OMEGA = 0.4 # constant for Derivative method for the BODY's angular velocity     
KP_V = 0.3  # constant for Proportionality for the BODY's linear velocity

LAST_THETA_ERROR = 0.0  # global variable to keep track of the theta error across functions


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))

def get_wheel_speeds(v_req, omega_req):
    # differential drive equations for each side's angular velocity
    phi_dot_R = (v_req + (omega_req * ROBOT_LENGTH) / 2.0) / ROBOT_r
    phi_dot_L = (v_req - (omega_req * ROBOT_LENGTH) / 2.0) / ROBOT_r
    return phi_dot_L, phi_dot_R

def get_control_signals(robot_x, robot_y, robot_theta, target_x, target_y, dt):
    global LAST_THETA_ERROR
    
    dist_error = math.sqrt( (target_x - robot_x)**2 + (target_y - robot_y)**2 )
    v_req = KP_V * dist_error
    v_req = min(v_req, 1.0)             # capping the speed to 1 m/s 
    
    theta_to_target = math.atan2(target_y - robot_y, target_x - robot_x)
    theta_error = normalize_angle(theta_to_target - robot_theta)

    P_term = KP_OMEGA * theta_error
    derivative_error = (theta_error - LAST_THETA_ERROR) / dt
    D_term = KD_OMEGA * derivative_error
    
    omega_req = P_term + D_term             # for now no Integral term 
    LAST_THETA_ERROR = theta_error
    
    return v_req, omega_req, dist_error


def run_pose_tracking(waypoints_with_pose):
    # Movement along the poses by decoupling rotation and translation
    
    def angle_diff(a, b):
        return math.atan2(math.sin(a - b), math.cos(a - b))

    ORIENT_ERR_UPPER = math.radians(60)   # Upper bound orientation error, when reached needs maximum angular speed
    ORIENT_ERR_LOWER = math.radians(15)    # Lower bound orientation error, when below it, has to creep to reach the desired orientation

    orientA = 0.01                 # constant to enhance omega - slope
    orientB = 0.25                 # constant to change omega - intercept
    omega_max   = 0.60                 # maximum omega needed if the orientation difference is so big
    omega_min = 0.15                 # minimum omega needed if the orientation difference is so small

    ORIENT_DT  = 0.3           # Time step size while rotation
    MAX_STEPS_ROT = 80           # total number of steps needed for the rotation
    ORIENT_OFFSET  = math.radians(15)  # offset introduced in the rotation due to the environment of the robot

    def rotation_output_from_pd(err_now, err_prev, dt):
        
        omega_pd = KP_OMEGA * err_now + KD_OMEGA * ((err_now - err_prev) / dt)
        e = abs(err_now)
        sign = 1.0 if omega_pd >= 0.0 else -1.0 

        if e > ORIENT_ERR_UPPER:
            omega = omega_max
        elif e > ORIENT_ERR_LOWER:
            omega = orientA + orientB * abs(omega_pd)
            omega = max(orientA, min(omega, omega_max))
        else:
            omega = omega_min
            omega = max(orientA, min(omega, omega_max))

        L = -omega if sign > 0 else +omega
        R = +omega if sign > 0 else -omega
        return L, R

    def rotate_to(target_theta, robot_theta, robot_x, robot_y, control_cmds, t):
        path_x_local, path_y_local = [], []
        err       = angle_diff(target_theta, robot_theta)
        prev_err  = err
        goal_mag  = abs(err)                   
        dir_sign  = 1.0 if err >= 0 else -1.0  

        turned_mag = 0.0   
        steps = 0

        while abs(err) > ORIENT_ERROR_MARGIN and steps < MAX_STEPS_ROT:

            L_cmd, R_cmd = rotation_output_from_pd(err, prev_err, ORIENT_DT)
            control_cmds.append((L_cmd, R_cmd))

            omega_actual = ROBOT_r * (R_cmd - L_cmd) / ROBOT_LENGTH

            prev_theta = robot_theta
            robot_theta = normalize_angle(robot_theta + omega_actual * ORIENT_DT)

            dtheta_signed = angle_diff(robot_theta, prev_theta)   
            progress = dir_sign * dtheta_signed                   
            if progress > 0:
                turned_mag += progress

            if turned_mag - goal_mag > ORIENT_OFFSET :
                #  stop rotation if rotation goes over the goal rotation
                robot_theta = normalize_angle(target_theta)
                break

            t += ORIENT_DT
            path_x_local.append(robot_x)
            path_y_local.append(robot_y)

            prev_err = err
            err = angle_diff(target_theta, robot_theta)
            steps += 1

        return robot_theta, robot_x, robot_y, t, path_x_local, path_y_local

    robot_path_x, robot_path_y, control_cmds = [], [], []
    robot_x, robot_y, robot_theta = waypoints_with_pose[0]
    current_waypoint_index = 1
    t = 0.0

    print(f"Starting at: ({robot_x:.2f}, {robot_y:.2f}), theta={robot_theta:.2f} rad")

    while True:
        
        robot_path_x.append(robot_x)
        robot_path_y.append(robot_y)

        if current_waypoint_index >= len(waypoints_with_pose):
            print(f"All waypoints processed at t={t:.2f}s. Stopping.")
            break

        target_x, target_y, target_theta = waypoints_with_pose[current_waypoint_index]

        # Step - 1:  ROTATE toward the next waypoint
        dx, dy = target_x - robot_x, target_y - robot_y
        theta_to_target = math.atan2(dy, dx)

        robot_theta, robot_x, robot_y, t, px, py = rotate_to(theta_to_target, robot_theta, robot_x, robot_y, control_cmds, t)
        robot_path_x.extend(px)
        robot_path_y.extend(py)

        # Step - 2: DRIVE STRAIGHT toward waypoint 
        dx, dy = target_x - robot_x, target_y - robot_y
        dist_error = math.hypot(dx, dy)

        v_req = KP_V * dist_error
        v_req = min(v_req, 1.0)
        omega_req = 0.0

        if dist_error < DIST_ERROR_MARGIN:
            v_req = 0.0

        phi_dot_L, phi_dot_R = get_wheel_speeds(v_req, omega_req)
        control_cmds.append((phi_dot_L, phi_dot_R))

        v_actual     = ROBOT_r * (phi_dot_R + phi_dot_L) / 2.0
        omega_actual = ROBOT_r * (phi_dot_R - phi_dot_L) / ROBOT_LENGTH

        robot_theta = normalize_angle(robot_theta + omega_actual * DT)
        robot_x += v_actual * math.cos(robot_theta) * DT
        robot_y += v_actual * math.sin(robot_theta) * DT
        t += DT

        # Step - 3: FINAL POSE ROTATION at the waypoint to the desired orientation
        dx, dy = target_x - robot_x, target_y - robot_y
        if math.hypot(dx, dy) < DIST_ERROR_MARGIN:
            robot_theta, robot_x, robot_y, t, px2, py2 = rotate_to(
                target_theta, robot_theta, robot_x, robot_y, control_cmds, t
            )
            robot_path_x.extend(px2); robot_path_y.extend(py2)
            current_waypoint_index += 1

    return robot_path_x, robot_path_y, waypoints_with_pose, control_cmds

def cap_angular_speed(w):
    # capping the maximum absolute speed to 1 rad/s and minimum absolute speed to 0.01 rad/s
    return min(1, max(abs(w), 0.01))

def calibrator(L, R, lateral_offset=0.0):
    # Setup constants for each wheel set's angular velocity to scale according to the environment
    K_L = 0.09
    K_R = 0.10
    M = 1.1
    
    # Correction for straight-line deviation using AprilTag
    # Positive offset = robot drifting right, need to turn left
    KP_CORRECTION = 15.0  # tune this value
    omega_correction = -KP_CORRECTION * lateral_offset
    if L > 0 and R > 0:
        # Moving forward - apply straightness correction
        L += omega_correction / 2
        R -= omega_correction / 2
        if L < M and R < M:
            pass
        elif (abs(R - L) < 0.01):
            K_L = 1.15 * K_L
    else:
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
    print("lateral_offset", lateral_offset)
    print("L:", L, "R:", R)
    return L, R

waypts_file = open("robot_control/robot_control/waypoints.txt").readlines()
WAYPOINTS_WITH_POSE = [[float(v) for v in line.split(",")] for line in waypts_file]


class KeyboardControllerNode(Node):
    def __init__(self):
        super().__init__('keyboard_controller_node')
        
        # create publisher for motor commands
        self.publisher = self.create_publisher(
            Float32MultiArray,
            'motor_commands',
            10
        )

        # TF2 listener for AprilTag
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tag_lateral_offset = 0.0  # lateral deviation from centerline

        # pulling generated trajectory from the robot.py file for more calibration locally
        self.cmd_idx = 0
        _, _, _, self.trajectory = run_pose_tracking(WAYPOINTS_WITH_POSE)
        
        # keyboard state
        self.pressed_keys = set()
        self.current_L = 0.0
        self.current_R = 0.0
        self.running = True
        
        # start keyboard listener
        self.start_keyboard_listener()
        
        # setup timer to publish commands at 20Hz
        self.timer = self.create_timer(0.05, self.publish_motor_commands)  # 20Hz
        
        self.get_logger().info('Keyboard Controller Node started')
        self.print_instructions()

    def get_apriltag_offset(self):
        """Get lateral offset from AprilTag via TF"""
        try:
            for i in range(2):
                # Get transform from camera to tag
                trans = self.tf_buffer.lookup_transform(
                    'camera_frame',  # adjust to your camera frame name
                    f'tag_{i}',  # adjust to your tag frame name
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.1)
                )
                # Y-axis offset represents lateral deviation
                self.tag_lateral_offset = -trans.transform.translation.x
                return self.tag_lateral_offset
        except Exception as e:
            # Tag not visible or TF not available
            return self.tag_lateral_offset  # use last known value
        
    def print_instructions(self):
        """print control instructions"""
        instructions = """
========================================
   🚗  Robot Keyboard Controller
----------------------------------------
   W : Forward  (L=0.3, R=0.3)
   S : Backward (L=-0.3, R=-0.3)
   A : Turn Left (L=-0.5, R=0.5)
   D : Turn Right (L=0.5, R=-0.5)
   X : Stop (L=0.0, R=0.0)
  ESC: Quit
========================================
Publishing motor commands on topic: motor_commands
"""
        self.get_logger().info(instructions)
    
    def on_press(self, key):
        """press key callback"""
        try:
            if hasattr(key, "char") and key.char:
                c = key.char.lower()
                self.pressed_keys.add(c)
                
                if c == 'w':
                    self.current_L = 0.3
                    self.current_R = 0.3
                elif c == 's':
                    self.current_L = -0.3
                    self.current_R = -0.3
                elif c == 'a':
                    self.current_L = -0.5
                    self.current_R = 0.5
                elif c == 'd':
                    self.current_L = 0.5
                    self.current_R = -0.5
                elif c == 'x':
                    self.current_L = 0.0
                    self.current_R = 0.0
                    self.pressed_keys.clear()
                    
        except Exception as e:
            self.get_logger().error(f'Error in on_press: {str(e)}')
    
    def on_release(self, key):
        """release key callback"""
        try:
            if key == Key.esc:
                self.get_logger().info('ESC pressed, shutting down...')
                self.running = False
                rclpy.shutdown()
                return False 
            
            if hasattr(key, "char") and key.char:
                c = key.char.lower()
                self.pressed_keys.discard(c)
                
                # stop if no direction keys are pressed
                if not any(k in self.pressed_keys for k in 'wasd'):
                    self.current_L = 0.0
                    self.current_R = 0.0
                    self.get_logger().debug('All direction keys released, stopping')
                    
        except Exception as e:
            self.get_logger().error(f'Error in on_release: {str(e)}')
    
    def start_keyboard_listener(self):
        """Start keyboard listener in a separate thread"""
        def listener_thread():
            with Listener(on_press=self.on_press, on_release=self.on_release, suppress=True) as listener:
                listener.join()
        
        self.listener_thread = threading.Thread(target=listener_thread, daemon=True)
        self.listener_thread.start()
        self.get_logger().info('Keyboard listener started')

    def publish_motor_commands(self):
        """Publish motor commands to ROS topic"""
        if self.running and rclpy.ok():
            try:
                msg = Float32MultiArray()
                
                # extracting angular velocities and get AprilTag offset
                L, R = self.trajectory[self.cmd_idx][0], self.trajectory[self.cmd_idx][1]
                offset = self.get_apriltag_offset()
                L, R = calibrator(L, R, offset)
                self.get_logger().info(f'Lateral offset: {offset:.3f}m')

                msg.data = [L, R]
                self.cmd_idx += 1
                print(self.cmd_idx, msg.data)
                self.publisher.publish(msg)
                
                # Log only when commands change to reduce console spam
                if hasattr(self, '_last_L') and hasattr(self, '_last_R'):
                    if self._last_L != self.current_L or self._last_R != self.current_R:
                        self.get_logger().info(f'Motor command: L={self.current_L:.2f}, R={self.current_R:.2f}')
                else:
                    self.get_logger().info(f'Motor command: L={self.current_L:.2f}, R={self.current_R:.2f}')
                
                self._last_L = self.current_L
                self._last_R = self.current_R

                if self.cmd_idx >= len(self.trajectory):
                    exit(0)
            except Exception as e:
                # Context may be shutting down, ignore errors
                if self.running:
                    self.get_logger().error(f'Error publishing command: {str(e)}')
    
    def destroy_node(self):
        """Cleanup operations when node is destroyed"""
        self.running = False
        
        # Send stop command before shutdown
        msg = Float32MultiArray()
        msg.data = [0.0, 0.0]
        self.publisher.publish(msg)
        self.get_logger().info('Stop command sent')
        
        super().destroy_node()


def plot_path(path_x, path_y, waypoints):
    waypoint_x = [p[0] for p in waypoints]
    waypoint_y = [p[1] for p in waypoints]
    
    plt.figure(figsize=(10, 8))
    plt.plot(waypoint_x, waypoint_y, 'ro', label='Target Waypoints')


    for i, (x, y, theta) in enumerate(waypoints):
        plt.text(x, y + 0.1, f'P{i}', fontsize=10, ha='center')
        if i > 0:
            plt.arrow(x, y, 0.5 * math.cos(theta), 0.5 * math.sin(theta), 
                      head_width=0.05, head_length=0.1, fc='red', ec='red', 
                      label='Desired theta' if i == 1 else '')
    
    plt.plot(path_x, path_y, 'b-', label='Robot Trajectory')
    plt.plot(path_x[0], path_y[0], 'go', markersize=8, label='Start Position')
    plt.plot(path_x[-1], path_y[-1], 'gx', markersize=10, label='End Position')

    plt.title('Robot Pose Tracking Simulation')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # If timer already destroyed the node, skip
        if not getattr(node, "_already_destroyed", False):
            try:
                node.destroy_node()
            except Exception:
                pass

        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass
        return 0

if __name__ == '__main__':
    # Standalone test - generate trajectory
    path_x, path_y, wps, control_cmds = run_pose_tracking(WAYPOINTS_WITH_POSE)
    plot_path(path_x, path_y, wps)
    for i, (phi_dot_L, phi_dot_R) in enumerate(control_cmds):
        print(f"Step {i}: Wheel Speeds -> Left: {phi_dot_L:.2f} rad/s, Right: {phi_dot_R:.2f} rad/s")