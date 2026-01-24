import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from control_msgs.action import GripperCommand

class GripperControl(Node):
    def __init__(self):
        super().__init__('gripper_control_node')
        self._action_client = ActionClient(
            self, 
            GripperCommand, 
            '/robotiq/robotiq_gripper_controller/gripper_cmd'
        )

    def send_goal(self, position, effort=500.0):
        if not self._action_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error('그리퍼 액션 서버를 찾을 수 없습니다!')
            return

        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = position
        goal_msg.command.max_effort = effort

        self.get_logger().info(f'그리퍼 이동 명령: {position}')
        
        self._send_goal_future = self._action_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('명령이 거부되었습니다.')
            return

        self.get_logger().info('명령이 수락되었습니다.')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'동작 완료! 최종 위치: {result.position}')

def main(args=None):
    rclpy.init(args=args)
    node = GripperControl()
    
    node.send_goal(0.25)
    
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()