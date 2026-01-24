import sys
import rclpy
from rclpy.node import Node
import math
from geometry_msgs.msg import TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
import tf_transformations # 좌표 변환 계산을 위해 필요할 수 있음

class CameraStaticTfPublisher(Node):
    def __init__(self):
        super().__init__('camera_static_tf_publisher')

        # Static Broadcaster 초기화
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)

        # TF 정보 발행
        self.make_static_transforms()

    def make_static_transforms(self):
        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'link_6'
        t.child_frame_id = 'camera_link'    

        # 예: End-effector 중심에서 앞으로 5cm, 위로 10cm 떨어진 경우
        t.transform.translation.x = -0.0325
        t.transform.translation.y = -0.0595
        t.transform.translation.z = 0.11525

        quat = tf_transformations.quaternion_from_euler(0, 0, math.pi) # R, P, Y
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]


        opt = TransformStamped()
        # opt.header.stamp = t.header.stamp
        opt.header.frame_id = 'camera_link'
        opt.child_frame_id = 'camera_link_optical' # '렌즈' 전용 좌표계

        # 위치는 camera_link와 동일 (이동 없음)
        opt.transform.translation.x = 0.0
        opt.transform.translation.y = 0.0
        opt.transform.translation.z = 0.0

        # [이게 핵심] Z-앞(카메라 데이터)을 X-앞(ROS 표준)으로 매핑하는 고정 회전
        # 사용자님이 수동으로 하려던 '축 맞추기'를 ROS 표준에 맞게 선언하는 겁니다.
        q_opt = tf_transformations.quaternion_from_euler(1.5708, -1.5708, 0) # R:-90, P:0, Y:-90
        opt.transform.rotation.x = q_opt[0]
        opt.transform.rotation.y = q_opt[1]
        opt.transform.rotation.z = q_opt[2]
        opt.transform.rotation.w = q_opt[3]
        self.get_logger().info(f'Published static TF: {opt.header.frame_id} -> {opt.child_frame_id}')

        self.tf_static_broadcaster.sendTransform([t, opt])

def main():
    rclpy.init()
    node = CameraStaticTfPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

if __name__ == '__main__':
    main()