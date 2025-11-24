import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster
import math


class StaticTFCameraNode(Node):
    def __init__(self):
        super().__init__('static_tf_camera_node')

        self.tf_static_broadcaster = StaticTransformBroadcaster(self)

        self.publish_static_transform()
        
        self.get_logger().info('Static TF Camera Node initialized')
        self.get_logger().info('Broadcasting static TF: base_link -> camera_frame')
        self.get_logger().info('Camera position: x=0.0675m, y=0.0m, z=0.035m')
        self.get_logger().info('Camera orientation: z->front, x->down, y->left')

    def publish_static_transform(self):
        """Publish static transform from base_link to camera_frame"""

        t = TransformStamped()
        
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id = 'camera_frame'
        
        t.transform.translation.x = 0.0675
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.035
        
        t.transform.rotation.w = 0.5
        t.transform.rotation.x = -0.5
        t.transform.rotation.y = 0.5
        t.transform.rotation.z = -0.5
        
        self.tf_static_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    static_tf_camera_node = StaticTFCameraNode()
    try:
        rclpy.spin(static_tf_camera_node)
    except KeyboardInterrupt:
        static_tf_camera_node.get_logger().info('Static TF Camera node stopped')
    finally:
        static_tf_camera_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()