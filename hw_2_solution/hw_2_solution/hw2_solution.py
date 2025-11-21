#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TransformStamped
from tf2_ros import Buffer, TransformListener, TransformBroadcaster
import numpy as np
import os 
import yaml
import time
import math
from ament_index_python.packages import get_package_share_directory
from scipy.spatial.transform import Rotation

"""
The class of the pid controller for differential drive robot.
"""
class PIDcontroller:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.target = None
        self.I = np.array([0.0, 0.0])
        self.lastError = np.array([0.0, 0.0])
        self.timestep = 0.1
        self.maximumValue = 0.2

    def setTarget(self, state):
        """
        set the target pose.
        """
        self.I = np.array([0.0, 0.0]) 
        self.lastError = np.array([0.0, 0.0])
        self.target = np.array(state)

    def getError(self, currentState, targetState, drive_backwards):
        """
        return the error between current and target state
        for differential drive: distance error and heading error
        """
        delta_x = targetState[0] - currentState[0]
        delta_y = targetState[1] - currentState[1]
        
        distance = np.sqrt(delta_x**2 + delta_y**2)
        
        angle_to_target = np.arctan2(delta_y, delta_x)
        
        if drive_backwards:
            desired_heading = angle_to_target + np.pi
            desired_heading = (desired_heading + np.pi) % (2 * np.pi) - np.pi
            distance = -distance
        else:
            desired_heading = angle_to_target
        
        heading_error = desired_heading - currentState[2]
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
        
        if abs(distance) < 0.05:
            heading_error = targetState[2] - currentState[2]
            heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
            distance = 0.0
        
        return np.array([distance, heading_error])

    def setMaximumUpdate(self, mv):
        """
        set maximum velocity for stability.
        """
        self.maximumValue = mv

    def update(self, currentState, drive_backwards):
        """
        calculate the update value based on PID control
        Returns: [linear_velocity, angular_velocity]
        """
        e = self.getError(currentState, self.target, drive_backwards)

        P = self.Kp * e
        self.I = self.I + self.Ki * e * self.timestep 
        I = self.I
        D = self.Kd * (e - self.lastError)
        result = P + I + D

        self.lastError = e

        if abs(result[0]) > self.maximumValue:
            result[0] = np.sign(result[0]) * self.maximumValue
            
        max_angular = 1.5 # rad/s
        if abs(result[1]) > max_angular:
            result[1] = np.sign(result[1]) * max_angular
        
        if abs(e[0]) < 0.05:
            result[0] = 0.0
        return result

class Hw2SolutionNode(Node):
    def __init__(self):
        super().__init__('hw2_solution_node')
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        self.odom_frame = 'odom'
        self.base_frame = 'base_link'
        
        self.tag_positions = {}
        
        self.waypoints = np.array([
            [0.0, 0.0, 0.0], 
            [1.0, 0.0, 0.0],
            [1.0, 2.0, np.pi],
            [0.0, 0.0, 0.0]
        ])
        
        self.pid = PIDcontroller(0.8, 0.01, 0.005)
        
        self.current_state = np.array([0.0, 0.0, 0.0])
        self.obs_current_state = np.array([0.0, 0.0, 0.0])
        
        self.current_waypoint_idx = 0
        self.waypoint_reached = False
        self.tolerance = 0.15
        self.angle_tolerance = 0.1
        
        self.last_tag_detection_time = 0.0
        self.using_tag_localization = False
        self.tag_initialized = False
        
        self.load_tag_configurations()
        
        self.drive_backwards = False
        self.dt = 0.1
        self.control_timer = self.create_timer(self.dt, self.control_loop)
        self.localization_timer = self.create_timer(0.1, self.localization_update) 
        
        self.stage = 'rotate_to_goal'
        self.stage_pid = PIDcontroller(0.8, 0.01, 0.005)
        self.fixed_rotation_vel = 0.785
        
    def load_tag_configurations(self):
        """Load AprilTag positions and orientations from YAML file"""
        try:
            package_share_dir = get_package_share_directory('hw_2_solution')
            yaml_path = os.path.join(package_share_dir, 'configs', 'apriltags_position.yaml')
            
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
        
    def update_dead_reckoning(self, linear_vel, angular_vel):
        """
        Update robot pose using dead reckoning
        """
        self.current_state[0] += linear_vel * np.cos(self.current_state[2]) * self.dt
        self.current_state[1] += linear_vel * np.sin(self.current_state[2]) * self.dt
        self.current_state[2] += angular_vel * self.dt
        self.current_state[2] = (self.current_state[2] + np.pi) % (2 * np.pi) - np.pi
        
    def broadcast_tf(self):
        """
        Broadcast TF transform from odom to base_link
        """
        current_time = self.get_clock().now()
        
        t = TransformStamped()
        t.header.stamp = current_time.to_msg()
        t.header.frame_id = self.odom_frame
        t.child_frame_id = self.base_frame
        
        t.transform.translation.x = self.current_state[0]
        t.transform.translation.y = self.current_state[1]
        t.transform.translation.z = 0.0
        
        qx, qy, qz, qw = self.euler_to_quaternion(0, 0, self.current_state[2])
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        
        self.tf_broadcaster.sendTransform(t)
        
    def euler_to_quaternion(self, roll, pitch, yaw):
        """
        Convert Euler angles to quaternion
        """
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        
        return qx, qy, qz, qw
    
    def should_drive_backwards(self, current_wp):
        """
        Determine if robot should drive backwards considering both:
        1. Angle to target position
        2. Target orientation at waypoint
        
        Returns True if backwards is better, False for forward
        """
        # delta_x = current_wp[0] - self.current_state[0]
        # delta_y = current_wp[1] - self.current_state[1]
        # angle_to_target = np.arctan2(delta_y, delta_x)
        
        # angle_diff_to_pos = angle_to_target - self.current_state[2]
        # angle_diff_to_pos = (angle_diff_to_pos + np.pi) % (2 * np.pi) - np.pi
        
        # angle_diff_to_orient = current_wp[2] - self.current_state[2]
        # angle_diff_to_orient = (angle_diff_to_orient + np.pi) % (2 * np.pi) - np.pi
        
        # forward_rotation = abs(angle_diff_to_pos) + abs(current_wp[2] - angle_to_target)
        
        # backward_angle_to_pos = angle_to_target + np.pi
        # backward_angle_to_pos = (backward_angle_to_pos + np.pi) % (2 * np.pi) - np.pi
        # backward_initial_rotation = backward_angle_to_pos - self.current_state[2]
        # backward_initial_rotation = (backward_initial_rotation + np.pi) % (2 * np.pi) - np.pi
        
        # backward_rotation = abs(backward_initial_rotation) + abs(current_wp[2] - backward_angle_to_pos)
        
        # drive_backwards = backward_rotation < forward_rotation
        return False
    
    def get_desired_heading_to_goal(self, current_wp, drive_backwards):
        """
        Get the desired heading to face towards (or away from) the goal
        """
        delta_x = current_wp[0] - self.current_state[0]
        delta_y = current_wp[1] - self.current_state[1]
        angle_to_target = np.arctan2(delta_y, delta_x)
        
        if drive_backwards:
            desired_heading = angle_to_target + np.pi
            desired_heading = (desired_heading + np.pi) % (2 * np.pi) - np.pi
        else:
            desired_heading = angle_to_target
        
        return desired_heading
    
    def get_rotation_direction(self, heading_error):
        """
        Determine rotation direction based on heading error.
        Returns: angular velocity with fixed magnitude but correct direction
        """
        if heading_error > 0:
            return self.fixed_rotation_vel
        else:
            return -self.fixed_rotation_vel
        
    def compute_and_publish_robot_pose_from_tag(self, tag_id, tag_observation):
        """Compute robot pose from tag observation and publish it"""
        tag_map = self.tag_positions[tag_id]
        
        tag_map_pos = np.array([tag_map['x'], tag_map['y'], tag_map['z']])
        tag_map_rot = Rotation.from_quat([tag_map['qx'], tag_map['qy'], tag_map['qz'], tag_map['qw']])
        
        obs_pos = np.array([
            tag_observation.transform.translation.x,
            tag_observation.transform.translation.y,
            tag_observation.transform.translation.z
        ])
        obs_rot = Rotation.from_quat([
            tag_observation.transform.rotation.x,
            tag_observation.transform.rotation.y,
            tag_observation.transform.rotation.z,
            tag_observation.transform.rotation.w
        ])
        
        tag_to_robot_rot = obs_rot.inv()
        tag_to_robot_pos = -tag_to_robot_rot.apply(obs_pos)
        
        robot_map_rot = tag_map_rot * tag_to_robot_rot
        robot_map_pos = tag_map_pos + tag_map_rot.apply(tag_to_robot_pos)
        
        yaw = robot_map_rot.as_euler('xyz')[2]
        self.current_state = np.array([robot_map_pos[0], robot_map_pos[1], yaw])
        self.obs_current_state = np.array([robot_map_pos[0], robot_map_pos[1], yaw])
        
        self.get_logger().info(
            f'Updated pose from tag {tag_id}: '
            f'pos=({robot_map_pos[0]:.3f}, {robot_map_pos[1]:.3f}), yaw={yaw:.3f}'
        )

    def control_loop(self):
        """
        Main control loop with three stages: rotate to goal, drive, rotate to orientation
        """
        if self.current_waypoint_idx >= len(self.waypoints):
            self.get_logger().info('All waypoints reached! Stopping robot.')
            self.stop_robot()
            self.broadcast_tf()
            return

        current_wp = self.waypoints[self.current_waypoint_idx]
        
        if not self.waypoint_reached:
            self.pid.setTarget(current_wp)
            self.drive_backwards = self.should_drive_backwards(current_wp)
            self.waypoint_reached = True
            self.stage = 'rotate_to_goal'
            self.stage_pid.setTarget(current_wp)

        delta_x = current_wp[0] - self.obs_current_state[0]
        delta_y = current_wp[1] - self.obs_current_state[1]
        position_error = np.sqrt(delta_x**2 + delta_y**2)
        twist_msg = Twist()
        
        # Stage 1: Rotate to face the goal (or away if driving backwards)
        if self.stage == 'rotate_to_goal':
            desired_heading = self.get_desired_heading_to_goal(current_wp, self.drive_backwards)
            heading_error = desired_heading - self.current_state[2]
            heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
            
            if abs(heading_error) < 0.05:
                self.stage = 'drive'
                twist_msg.angular.z = 0.0
            else:
                twist_msg.angular.z = float(self.get_rotation_direction(heading_error))
        
        # Stage 2: Drive towards/away from the goal
        elif self.stage == 'drive':
            position_error = np.sqrt(delta_x**2 + delta_y**2)
            if position_error < self.tolerance:
                self.stage = 'rotate_to_orient'
                twist_msg.linear.x = 0.0
            else:
                desired_heading = self.get_desired_heading_to_goal(current_wp, self.drive_backwards)
                heading_error = desired_heading - self.current_state[2]
                heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
                if abs(heading_error) > 0.2:
                    self.stage = 'rotate_to_goal'
                    twist_msg.linear.x = 0.0
                else:
                    update_value = self.pid.update(self.current_state, self.drive_backwards)
                    twist_msg.linear.x = float(update_value[0])
        
        # Stage 3: Rotate to target orientation
        elif self.stage == 'rotate_to_orient':
            heading_error = current_wp[2] - self.current_state[2]
            heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
            if abs(heading_error) < self.angle_tolerance:
                self.current_waypoint_idx += 1
                self.waypoint_reached = False
                twist_msg.angular.z = 0.0
            else:
                twist_msg.angular.z = float(self.get_rotation_direction(heading_error))
        
        # Only update dead reckoning if not using tag localization
        # if not self.using_tag_localization:
        self.update_dead_reckoning(twist_msg.linear.x, twist_msg.angular.z)
        self.broadcast_tf()
        self.cmd_vel_pub.publish(twist_msg)
        
        
    def localization_update(self):
        """Main localization update - tries AprilTag first, then dead reckoning"""
        current_time = time.time()
        tag_detected = False
        closest_tag_id = None
        closest_observation = None
        closest_distance = float('inf')
        for tag_id in self.tag_positions.keys():
            try:
                tag_frame = f'tag_{tag_id}'
                observation = self.tf_buffer.lookup_transform('base_link', tag_frame, rclpy.time.Time())
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
            except Exception:
                continue
        
        if closest_observation is not None:
            self.compute_and_publish_robot_pose_from_tag(closest_tag_id, closest_observation)
            tag_detected = True
            self.last_tag_detection_time = current_time
            self.using_tag_localization = True
        else:
            tag_detected = False
            time_since_last_tag = current_time - self.last_tag_detection_time
            if time_since_last_tag > 1.0:  # 1 second timeout
                self.using_tag_localization = False
    

    def stop_robot(self):
        """
        Stop the robot by publishing zero velocities
        """
        twist_msg = Twist()
        self.cmd_vel_pub.publish(twist_msg)

def main(args=None):
    rclpy.init(args=args)
    node = Hw2SolutionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Stopped by keyboard interrupt')
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
