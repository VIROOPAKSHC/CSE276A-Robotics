'''
This file implements a complete EKF SLAM system from scratch.
It uses a "stop-turn-go" waypoint navigator to control the robot based on the 
EKF's estimated pose, which is more robust for real-world hardware.
'''
import rclpy
from rclpy.node import Node
import numpy as np
import math
from std_msgs.msg import Float32MultiArray
import tf2_ros
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid, Path
from tf2_ros import TransformBroadcaster
import tf_transformations
import yaml
import time
from .path_planning import AStarPlanner
import os
from ament_index_python.packages import get_package_share_directory
from scipy.spatial.transform import Rotation


ROBOT_WHEEL_RADIUS = 0.0508
ROBOT_WHEEL_BASE = 0.127
MAX_CNT = 10

class EkfSlamNode(Node):
    '''A ROS 2 Node to implement and run EKF SLAM from first principles.'''
    def __init__(self):
        super().__init__('ekf_slam_node')

        self.motor_pub = self.create_publisher(Float32MultiArray, 'motor_commands', 10)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.pose_pub = self.create_publisher(PoseStamped, 'ekf_pose', 10)
        self.grid_pub = self.create_publisher(OccupancyGrid, 'occupancy_grid', 10)
        self.path_pub = self.create_publisher(Path, 'planned_path', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.x_est = np.array([0.0, 0.0, 0.0])
        self.last_control = np.array([0.0, 0.0])
        self.last_command_time = time.time()
        
        self.tag_positions = {}
        self.last_tag_detection_time = 0.0
        self.using_tag_localization = False
        self.pose_updated_this_cycle = False
        self.cnt = MAX_CNT

        # Error-correction control parameters
        self.pose_error = np.array([0.0, 0.0, 0.0]) # [dx, dy, d_theta] in world frame
        self.correction_strength_lat = 0.5 # Tune this for lateral error
        self.correction_strength_angle = 0.2 # Tune this for angle error
        # Occupancy Grid parameters
        self.grid_resolution = 0.1  # meters per cell
        self.grid_width = int(3.0 / self.grid_resolution)  # 10 ft = 3.048 m
        self.grid_height = int(3.0 / self.grid_resolution)
        self.grid_origin = [-1.5, -1.5]  # Center the grid at (0,0)
        self.occupancy_grid = np.full((self.grid_height, self.grid_width), -1, dtype=np.int8)
        
        # Obstacle parameters
        self.obstacle_center = [0.0, 0.0]  # meters
        self.obstacle_size = [0.5, 0.5]   # meters (2x2 ft)

        self.create_occupancy_grid()

        # A* Planner
        self.planner = AStarPlanner(self.occupancy_grid, self.grid_resolution, self.grid_origin)
        start_point = [-0.85, -0.85]
        goal_point = [0.85, 0.85]
        self.x_est = np.array([start_point[0], start_point[1], 0.0])
        self.waypoints = np.array(self.planner.plan(start_point, goal_point))

        if len(self.waypoints)==0:
            self.get_logger().error("Could not find a path.")
            return

        self.iterations = 1
        self.current_waypoint_index = 1
        self.goal_reached = False
        self.navigation_state = 'TURNING'
        
        # Track when state changes to allow settling
        self.state_change_time = time.time()
        self.STATE_SETTLE_TIME = 1  # Wait 0.5s after state change before moving

        self.DIST_THRESHOLD = 0.1
        self.ANGLE_THRESHOLD = np.deg2rad(15)
        self.DRIVE_SPEED = 0.19  # The true physical speed (m/s) the robot achieves
        self.TURN_SPEED = 1.5    # The true physical angular speed (rad/s)

        # Calibrated based on final experimental data
        self.K_L = 1
        self.K_R = 1

        # Control loop runs at 10Hz
        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info('EKF SLAM Node Initialized with Stop-Turn-Go Controller.')
        self.load_tag_configurations()

    def load_tag_configurations(self):
        """Load AprilTag positions and orientations from YAML file"""
        try:
            package_share_dir = get_package_share_directory('robot_control')
            yaml_path = os.path.join(package_share_dir, 'config', 'apriltags_position.yaml')
            
            with open(yaml_path, 'r') as file:
                data = yaml.safe_load(file)
            
            tags_data = data.get('apriltags', [])
            
            for tag in tags_data:
                tag_id = tag.get('id')
                if tag_id is None:
                    continue
                
                self.tag_positions[tag_id] = {
                    'x': float(tag['x']),
                    'y': float(tag['y']),
                    'z': float(tag['z']),
                    'qx': float(tag['qx']),
                    'qy': float(tag['qy']),
                    'qz': float(tag['qz']),
                    'qw': float(tag['qw'])
                }
                
                self.get_logger().info(
                    f'Loaded tag {tag_id}: pos=({tag["x"]:.2f}, {tag["y"]:.2f}, {tag["z"]:.2f})'
                )
                
        except Exception as e:
            self.get_logger().error(f'Failed to load tag configurations: {str(e)}')

    def localization_update(self):
        """Main localization update - tries AprilTag first, then dead reckoning"""
        current_time = time.time()
        self.pose_updated_this_cycle = False
        closest_tag_id = None
        closest_observation = None
        closest_distance = float('inf')
        
        for tag_id in self.tag_positions.keys():
            try:
                tag_frame = f'tag_{tag_id}'
                observation = self.tf_buffer.lookup_transform('camera_frame', tag_frame, rclpy.time.Time())
                transform_time = rclpy.time.Time.from_msg(observation.header.stamp)
                time_diff = (self.get_clock().now() - transform_time).nanoseconds / 1e9
                if time_diff > 0.25:
                    continue
                dx = observation.transform.translation.x
                dy = observation.transform.translation.y
                dz = observation.transform.translation.z
                distance = np.sqrt(dx*dx + dy*dy + dz*dz)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_tag_id = tag_id
                    closest_observation = observation
            except Exception as e:
                continue
        
        if closest_observation is not None:
            self.get_logger().info(f"Tag {closest_tag_id} detected at distance {closest_distance:.2f}m.")
            self.compute_and_publish_robot_pose_from_tag(closest_tag_id, closest_observation)
            self.last_tag_detection_time = current_time
            self.using_tag_localization = True
            self.pose_updated_this_cycle = True
        else:
            time_since_last_tag = current_time - self.last_tag_detection_time
            if time_since_last_tag > 1.0:  # 1 second timeout
                self.using_tag_localization = False

    def compute_and_publish_robot_pose_from_tag(self, tag_id, tag_observation):
        """Compute robot pose from tag observation and calculate the error."""
        
        tag_map = self.tag_positions[tag_id]
        tag_map_pos = np.array([tag_map['x'], tag_map['y'], tag_map['z']])
        tag_map_rot = Rotation.from_quat([tag_map['qx'], tag_map['qy'], tag_map['qz'], tag_map['qw']])
        
        obs_pos = np.array([
            tag_observation.transform.translation.x,
            0.0, # Project onto horizontal plane
            tag_observation.transform.translation.z
        ])
        obs_rot = Rotation.from_quat([
            tag_observation.transform.rotation.x,
            tag_observation.transform.rotation.y,
            tag_observation.transform.rotation.z,
            tag_observation.transform.rotation.w
        ])
        
        tag_to_camera_rot = obs_rot.inv()
        tag_to_camera_pos = -tag_to_camera_rot.apply(obs_pos)
        
        camera_map_rot = tag_map_rot * tag_to_camera_rot
        camera_map_pos = tag_map_pos + tag_map_rot.apply(tag_to_camera_pos)
        
        camera_forward_in_map = camera_map_rot.apply([0, 0, 1])
        yaw = np.arctan2(camera_forward_in_map[1], camera_forward_in_map[0])
        
        measured_pose = np.array([camera_map_pos[0], camera_map_pos[1], yaw])
        
        # Calculate the error between the measurement and the dead-reckoned pose
        self.pose_error = measured_pose - self.x_est
        # Normalize the angle error
        self.pose_error[2] = self._normalize_angle(self.pose_error[2])

        self.get_logger().info(
            f'Tag {tag_id} detected. Pose error: '
            f'dx={self.pose_error[0]:.3f}, dy={self.pose_error[1]:.3f}, dθ={math.degrees(self.pose_error[2]):.1f}°'
        )

    def create_occupancy_grid(self):
        """Initializes the occupancy grid, inflates the central obstacle."""
        self.occupancy_grid.fill(0)  # Initialize all cells as free space
        ox, oy = self.obstacle_center
        sw, sh = self.obstacle_size
        
        # Define robot radius for inflation
        robot_radius = 0.1  # meters
        inflation_radius_cells = int(np.ceil(robot_radius / self.grid_resolution))

        # Create a temporary grid to store the raw obstacle
        raw_obstacle_grid = np.zeros_like(self.occupancy_grid)

        for i in range(self.grid_height):
            for j in range(self.grid_width):
                map_x = self.grid_origin[0] + (j + 0.5) * self.grid_resolution
                map_y = self.grid_origin[1] + (i + 0.5) * self.grid_resolution
                if (map_x > ox - sw / 2 and map_x < ox + sw / 2 and
                    map_y > oy - sh / 2 and map_y < oy + sh / 2):
                    raw_obstacle_grid[i, j] = 100  # Mark raw obstacle

        # Inflate the obstacle
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                if raw_obstacle_grid[i, j] == 100:
                    # This cell is an obstacle, so inflate around it
                    for di in range(-inflation_radius_cells, inflation_radius_cells + 1):
                        for dj in range(-inflation_radius_cells, inflation_radius_cells + 1):
                            ni, nj = i + di, j + dj
                            if 0 <= ni < self.grid_height and 0 <= nj < self.grid_width:
                                self.occupancy_grid[ni, nj] = 100


    def publish_occupancy_grid(self):
        """Publishes the occupancy grid for visualization."""
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.info.resolution = self.grid_resolution
        msg.info.width = self.grid_width
        msg.info.height = self.grid_height
        msg.info.origin.position.x = self.grid_origin[0]
        msg.info.origin.position.y = self.grid_origin[1]
        msg.data = self.occupancy_grid.flatten().tolist()
        self.grid_pub.publish(msg)

    def publish_path(self):
        """Publishes the planned path for visualization."""
        if not len(self.waypoints):
            return
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        for waypoint in self.waypoints:
            pose = PoseStamped()
            pose.header.stamp = msg.header.stamp
            pose.header.frame_id = 'odom'
            pose.pose.position.x = waypoint[0]
            pose.pose.position.y = waypoint[1]
            msg.poses.append(pose)
        self.path_pub.publish(msg)

    def update_dead_reckoning(self):
        """Update robot pose using dead reckoning based on last command"""
        linear_vel, angular_vel = self.last_control
        dt = time.time() - self.last_command_time
        
        # Clamp dt to reasonable values
        if dt > 0.5:
            dt = 0.1  # Default timer period
        
        self.x_est[0] += linear_vel * np.cos(self.x_est[2]) * dt
        self.x_est[1] += linear_vel * np.sin(self.x_est[2]) * dt
        self.x_est[2] += angular_vel * dt
        self.x_est[2] = (self.x_est[2] + np.pi) % (2 * np.pi) - np.pi

    def _publish_pose(self):
        pose = self.x_est[:3]
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.pose.position.x = pose[0]
        msg.pose.position.y = pose[1]
        q = tf_transformations.quaternion_from_euler(0, 0, pose[2])
        msg.pose.orientation.x = q[0]
        msg.pose.orientation.y = q[1]
        msg.pose.orientation.z = q[2]
        msg.pose.orientation.w = q[3]
        self.pose_pub.publish(msg)
        
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'ekf_base_link'
        t.transform.translation.x = pose[0]
        t.transform.translation.y = pose[1]
        t.transform.translation.z = 0.0
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        self.tf_broadcaster.sendTransform(t)

    def _normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def control_loop(self):
        """Main control loop: Predict, Measure, Act"""
        # Predict step: update pose based on last command
        self.update_dead_reckoning()

        # Measure step: get a fresh observation from the camera
        if self.cnt == 0:
            self.localization_update()
            self.cnt = MAX_CNT
        self.cnt -= 1
        
        # Act step: decide and execute the next move based on the dead-reckoned pose
        # The navigation logic will internally use self.pose_error to correct the course
        self.navigate()
        
        # Visualization
        self.publish_occupancy_grid()
        self.publish_path()
        self._publish_pose()

    def navigate(self):
        """Stop-turn-go navigation with settling time after state changes"""
        v_req, omega_req = 0.0, 0.0
        current_time = time.time()

        self.get_logger().info(
            f"Pose: x={self.x_est[0]:.3f}, y={self.x_est[1]:.3f}, θ={math.degrees(self.x_est[2]):.1f}° | "
            f"State: {self.navigation_state} | Tag: {self.using_tag_localization}"
        )

        if self.current_waypoint_index >= len(self.waypoints):
            self.get_logger().info("Iteration Done !!", once=True)
            self.current_waypoint_index = 1
            self.iterations -= 1
            if self.iterations <= 0:
                self.goal_reached = True
                self.send_motor_commands(0.0, 0.0)
                exit(0)
            return

        current_pose = self.x_est[:3]
        target = self.waypoints[self.current_waypoint_index]
        dist_error = np.linalg.norm(target - current_pose[:2])
        angle_to_target = math.atan2(target[1] - current_pose[1], target[0] - current_pose[0])
        angle_error = self._normalize_angle(angle_to_target - current_pose[2])

        # Check if we need to settle after a state change
        time_since_state_change = current_time - self.state_change_time
        if time_since_state_change < self.STATE_SETTLE_TIME:
            # Still settling - stop and wait for tag detection
            self.send_motor_commands(0.0, 0.0)
            # self.get_logger().info(f"Settling... {self.STATE_SETTLE_TIME - time_since_state_change:.2f}s remaining")
            return

        if self.navigation_state == 'TURNING':
            self.K_L = 0.24
            self.K_R = 0.24

            self.state_change_time = current_time

            if abs(angle_error) > self.ANGLE_THRESHOLD:
                omega_req = self.TURN_SPEED * np.sign(angle_error)
            else:
                # Switch to driving
                self.navigation_state = 'DRIVING'
                self.get_logger().info("Switching to DRIVING state")
                omega_req = 0.0

        elif self.navigation_state == 'DRIVING':
            self.K_L = 0.04
            self.K_R = self.K_L * 1.1
            
            # Check if we've drifted off course
            if abs(angle_error) > self.ANGLE_THRESHOLD * 1.5:
                self.navigation_state = 'TURNING'
                self.get_logger().info("Off course - switching to TURNING")
                self.send_motor_commands(0.0, 0.0)
                return
            
            if dist_error > self.DIST_THRESHOLD:
                v_req = self.DRIVE_SPEED
            else:
                # Waypoint reached
                self.get_logger().info(f"Waypoint {self.current_waypoint_index} reached!")
                self.current_waypoint_index += 1
                self.navigation_state = 'TURNING'
                v_req = 0.0

        self.send_motor_commands(v_req, omega_req)

    def send_motor_commands(self, v_req, omega_req):
        """Calculate and send motor commands, applying camera-based correction."""
        
        omega_corr = 0.0
        # If there is a fresh error from the camera, calculate a correction
        if np.any(self.pose_error):
            dx, dy, d_theta = self.pose_error
            theta = self.x_est[2]
            
            # Transform world error to robot's local frame
            # lateral_error is the distance the robot is to the left of the measured pose
            lateral_error = -dx * np.sin(theta) + dy * np.cos(theta)
            
            # Calculate corrective angular velocity
            omega_corr = (self.correction_strength_lat * lateral_error) + \
                         (self.correction_strength_angle * d_theta)
            
            self.get_logger().info(
                f"Correction: lat_err={lateral_error:.3f}, ang_err={math.degrees(d_theta):.1f}° "
                f"-> ω_corr={omega_corr:.3f}"
            )
            
            # Reset the error so the correction is only applied once
            self.pose_error.fill(0.0)

        # Add correction to the navigation command
        omega_final = omega_req + omega_corr

        l_wheel, r_wheel = self._get_wheel_speeds(v_req, omega_final)
        l_wheel_cmd = l_wheel * self.K_L
        r_wheel_cmd = r_wheel * self.K_R
        
        self.get_logger().info(
            f"Commands: v={v_req:.3f} m/s, ω_nav={omega_req:.3f}, ω_corr={omega_corr:.3f}, ω_final={omega_final:.3f} rad/s | "
            f"Motors: L={l_wheel_cmd:.3f}, R={r_wheel_cmd:.3f}"
        )
        
        msg = Float32MultiArray()
        msg.data = [float(l_wheel_cmd), float(r_wheel_cmd)]
        self.motor_pub.publish(msg)
        
        # Update tracking
        self.last_control = np.array([v_req, omega_req]) # Use the un-corrected velocities for dead reckoning
        self.last_command_time = time.time()

    def _get_wheel_speeds(self, v_req, omega_req):
        phi_dot_R = (v_req + (omega_req * ROBOT_WHEEL_BASE) / 2.0) / ROBOT_WHEEL_RADIUS
        phi_dot_L = (v_req - (omega_req * ROBOT_WHEEL_BASE) / 2.0) / ROBOT_WHEEL_RADIUS
        return phi_dot_L, phi_dot_R

def main(args=None):
    rclpy.init(args=args)
    node = EkfSlamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node and not getattr(node, "_already_destroyed", False):
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()