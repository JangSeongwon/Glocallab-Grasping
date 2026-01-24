import sys
import os
import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import PointStamped
import tf_transformations
import rclpy
import DR_init
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from scipy.spatial.transform import Rotation as R
import math
import time
from GripperControl import GripperControl
import open3d as o3d

rclpy.init()

ROBOT_ID   = "dsr01"
ROBOT_MODEL= ""

DR_init.__dsr__id   = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL
node = rclpy.create_node('GlocalLabDemo', namespace=ROBOT_ID)
DR_init.__dsr__node = node

from DSR_ROBOT2 import print_ext_result, movej, movel, movec, move_periodic, move_spiral, set_velx, set_accx, DR_BASE, DR_TOOL, DR_AXIS_X, DR_MV_MOD_ABS
gripper = GripperControl()

class RobotTransformer(Node):
    def __init__(self):
        super().__init__('grasp_transformer')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def get_robot_base_pose(self, camera_point):
        for _ in range(5):
            rclpy.spin_once(self, timeout_sec=1.0)

        target_frame = 'base_link'
        source_frame = 'camera_link_optical'

        try:
            t = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(seconds=0, nanoseconds=0),
                rclpy.duration.Duration(seconds=1.0)
            )

            # 카메라 좌표게 기준 POSE
            cx, cy, cz = camera_point
            qx = t.transform.rotation.x
            qy = t.transform.rotation.y
            qz = t.transform.rotation.z
            qw = t.transform.rotation.w

            r11 = 1 - 2*(qy**2 + qz**2)
            r12 = 2*(qx*qy - qz*qw)
            r13 = 2*(qx*qz + qy*qw)

            r21 = 2*(qx*qy + qz*qw)
            r22 = 1 - 2*(qx**2 + qz**2)
            r23 = 2*(qy*qz - qx*qw)

            r31 = 2*(qx*qz - qy*qw)
            r32 = 2*(qy*qz + qx*qw)
            r33 = 1 - 2*(qx**2 + qy**2)

            # 4. 회전 적용 (Matrix Multiplication)
            rotated_x = r11 * cx + r12 * cy + r13 * cz
            rotated_y = r21 * cx + r22 * cy + r23 * cz
            rotated_z = r31 * cx + r32 * cy + r33 * cz
            # print("R", rotated_x, rotated_y, rotated_z)

            tx = t.transform.translation.x * 1000
            ty = t.transform.translation.y * 1000
            tz = t.transform.translation.z * 1000
            # print("tt", tx, ty, tz)

            final_x = tx - rotated_y
            final_y = ty - rotated_z
            final_z = tz + rotated_x

            return [final_x, final_y, final_z]

        except Exception as e:
            self.get_logger().error(f"수동 변환 실패: {e}")
            return None
        
transformer_node = RobotTransformer()
model = YOLO("YOLO_Model/workpiece1_OBB.pt")

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

print("Camera Detected")
profile = pipeline.start(config)
'''Depth 설정'''
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale() 
align_to = rs.stream.color
align = rs.align(align_to)

########################################  1. YOLO Detection  ########################################

def get_robust_depth(depth_frame, x, y, window_size = 5):
    depth_data = np.asanyarray(depth_frame.get_data())
    half_w = window_size // 2
    
    y_start, y_end = max(0, int(y)-half_w), min(depth_data.shape[0], int(y)+half_w+1)
    x_start, x_end = max(0, int(x)-half_w), min(depth_data.shape[1], int(x)+half_w+1)
    
    roi = depth_data[y_start:y_end, x_start:x_end]

    valid_depths = roi[roi > 0]
    
    if len(valid_depths) > 0:
        return np.mean(valid_depths) * depth_scale
    else:
        return 0

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

        if not color_frame:
            continue
        best_target_this_frame = None

        color_image = np.asanyarray(color_frame.get_data())

        results = model(color_image, conf=0.92)
        annotated_frame = results[0].plot()
        if results[0].obb is not None:
            clss = results[0].obb.cls.cpu().numpy()
            confs = results[0].obb.conf.cpu().numpy()
            obb_boxes = results[0].obb.xywhr.cpu().numpy()
            up_indices = [i for i, c in enumerate(clss) if model.names[int(c)] == 'Up']            
            
            if len(up_indices) > 0:
                up_confs = confs[up_indices]
                max_sub_idx = np.argmax(up_confs)
                max_idx = up_indices[max_sub_idx]
                max_conf = confs[max_idx]
                px, py, w, h, rotation = obb_boxes[max_idx]
                
                distance = get_robust_depth(depth_frame, px, py)
                # print(f"현재 인식된 픽셀 좌표: px={px}, py={py}, dist={distance:.4f}m")
                
                camera_coords = rs.rs2_deproject_pixel_to_point(intrinsics, [px, py], distance)
                # print(f"Intrinsics PPX: {intrinsics.ppx}, PPY: {intrinsics.ppy}")
                X_mm, Y_mm, Z_mm = [c * 1000 for c in camera_coords]
                angle_deg = np.degrees(rotation)

                pose_text = f"X:{X_mm:.1f} Y:{Y_mm:.1f} Z:{Z_mm:.1f} Angle:{angle_deg:.1f}"
                # print(pose_text)
                cv2.circle(annotated_frame, (int(px), int(py)), 5, (0, 0, 255), -1)
                cv2.putText(annotated_frame, pose_text, (int(px) - 50, int(py) - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(annotated_frame, pose_text, (int(px) - 50, int(py) - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
                best_target_this_frame = {
                    'pose': (-X_mm, -Y_mm, Z_mm, angle_deg)
                }
                cv2.circle(annotated_frame, (int(px), int(py)), 7, (0, 255, 255), -1) 
                cv2.putText(annotated_frame, f"TOP SCORE: {max_conf:.2f}", (int(px)-50, int(py)-40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow('Detection (Press K to Confirm)', annotated_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('k'):
            if best_target_this_frame is not None:
                target_detected_pose = best_target_this_frame['pose']
                print(f"Pose extracted: X={target_detected_pose[0]:.1f}, Y={target_detected_pose[1]:.1f}, Z={target_detected_pose[2]:.1f}, Angle={target_detected_pose[3]:.1f}")
                break 
            else:
                print("No Detection")
                sys.exit()
finally:
    pipeline.stop()
    cv2.destroyAllWindows()

########################################  2. 로봇 좌표계로 변환  ########################################

if target_detected_pose:
    for _ in range(5):
        rclpy.spin_once(transformer_node, timeout_sec = 1.0)
    available_frames = transformer_node.tf_buffer._getFrameStrings()
    # print(f"Current Available TF List: {available_frames}")

    base_coords = transformer_node.get_robot_base_pose(target_detected_pose[:3])
    if base_coords:
                    target_detected_pose = {
                        'POS': base_coords,
                        'ANGLE': target_detected_pose[3] 
                    }
                    print(f"Robot Base Coordinate: {target_detected_pose['POS']}")
    else:
        print("No Transformation")
        sys.exit()

###############################  3. Multi-view Pose Estimation  ###############################


def get_quaternion_from_euler(p_degree):
    theta = math.radians(p_degree)
    x = 0.0
    y = 0.0
    z = math.sin(theta / 2)
    w = math.cos(theta / 2)
    return [x, y, z, w]

def PoseEstimation(initial_POS, initial_angle):
     
    return initial_POS, initial_angle

def get_look_at_zyz(camera_pos, target_pos, prev_zyz=None):
    z_axis = np.array(target_pos) - np.array(camera_pos)
    z_axis = z_axis / (np.linalg.norm(z_axis)+ 1e-6)
    up = np.array([0, 0, 1])
    if abs(z_axis[2]) > 0.09:
        up = np.array([0, 1, 0])
        
    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-6)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-6)
    
    R_matrix = np.column_stack((x_axis, y_axis, z_axis))
    r = R.from_matrix(R_matrix)
    curr_zyz = r.as_euler('zyz', degrees=True)

    if prev_zyz is not None:
        new_zyz = np.copy(curr_zyz)
        for i in range(3):
            diff = new_zyz[i] - prev_zyz[i]
            if diff > 180: new_zyz[i] -= 360
            elif diff < -180: new_zyz[i] += 360
        return new_zyz
    return curr_zyz

def Multiview_pcd_data(pipeline, align, tf_matrix, save_dir, index, duration=1.0):
    pc = rs.pointcloud()
    all_points = []
    start_time = time.time()
    colorizer = rs.colorizer()
        
    while time.time() - start_time < duration:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame() 

        if not depth_frame or not color_frame:
            continue
            
        depth_color_frame = colorizer.colorize(depth_frame)
        depth_color_image = np.asanyarray(depth_color_frame.get_data())
        cv2.putText(depth_color_image, f"Recording Case Test...", 
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('RealSense Depth - Scanning', depth_color_image)
        cv2.waitKey(1)

        points = pc.calculate(depth_frame)
        v = points.get_vertices()
        verts = np.asanyarray(v).view(np.float32).reshape(-1, 3) * 1000 # mm 변환

        valid_indices = np.where((verts[:, 2] > 0) & (verts[:, 2] < 2000))[0] 
        all_points.append(verts[valid_indices])

    if not all_points:
        print("No valid points collected.")
        return

    merged_verts = np.vstack(all_points)
    homogen_verts = np.c_[merged_verts, np.ones(merged_verts.shape[0])]
    base_verts = (tf_matrix @ homogen_verts.T).T[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(base_verts)
    pcd = pcd.voxel_down_sample(voxel_size=1.0) 

    save_path = os.path.join(save_dir, f"view{index:02d}.pcd")
    o3d.io.write_point_cloud(save_path, pcd)
    print(f"Saved to {save_path}")
    


def get_current_camera_tf(transformer_node):
    try:
        t = transformer_node.tf_buffer.lookup_transform(
            'base_link',
            'camera_link_optical',
            rclpy.time.Time(seconds=0, nanoseconds=0), 
            rclpy.duration.Duration(seconds=2.0)
        )
        
        q = [t.transform.rotation.x, t.transform.rotation.y, 
             t.transform.rotation.z, t.transform.rotation.w]
        r = R.from_quat(q)
        rot_matrix = r.as_matrix()
        
        tx = t.transform.translation.x * 1000
        ty = t.transform.translation.y * 1000
        tz = t.transform.translation.z * 1000
        
        tf_matrix = np.eye(4)
        tf_matrix[:3, :3] = rot_matrix
        tf_matrix[:3, 3] = [tx, ty, tz]
        
        return tf_matrix
    except Exception as e:
        print(f"TF 조회 실패: {type(e).__name__} - {e}")
        return None

try:
    pipeline.start(config)
except RuntimeError:
    pass

if target_detected_pose:
    print("Pose Estimation Process")
    initial_POS = target_detected_pose['POS'] # mm 
    initial_angle = target_detected_pose['ANGLE'] # degree

    # Multi-view Scanning
    pcd_data_dir = "PCD_Data"
    SCAN_HEIGHT = 400.0
    DESIRED_ANGLE_DEG = 60.0
    rad_angle = math.radians(90.0 - DESIRED_ANGLE_DEG)
    SCAN_RADIUS = SCAN_HEIGHT * math.tan(rad_angle)
    NUM_POINTS = 2
    circle_path = []
    last_zyz = None

    for i in range(NUM_POINTS):
        angle = math.radians((360.0 / NUM_POINTS) * i)
        tx = initial_POS[0] + SCAN_RADIUS * math.cos(angle)
        ty = initial_POS[1] + SCAN_RADIUS * math.sin(angle)
        tz = initial_POS[2] + SCAN_HEIGHT
        curr_tcp_pos = [tx, ty, tz]

        zyz = get_look_at_zyz(curr_tcp_pos, initial_POS, last_zyz)
        circle_path.append([tx, ty, tz, zyz[0], zyz[1], zyz[2]])
        last_zyz = zyz

    movel(circle_path[0], v=100, a=200)
    print("Scanning Start")

    for count, wp in enumerate(circle_path, 1):
        movel(wp, v=100, a=200)
        time.sleep(0.5)
        for _ in range(5):
            rclpy.spin_once(transformer_node, timeout_sec = 1.0)

        tf_base_to_cam = get_current_camera_tf(transformer_node)
        if tf_base_to_cam is not None:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            Multiview_pcd_data(pipeline, align, tf_base_to_cam, pcd_data_dir, count, duration=1.0)        
        else:
            print(f"Warning: Failed to get TF for Viewpoint {count}")
        time.sleep(0.3)
        print(f"Scanned {count} Viewpoints")

    initial_POS, initial_angle = PoseEstimation(initial_POS, initial_angle)

####################################  4. Human Demonstration  ####################################


    # Human_GraspingPose_Candidates = [
    # [[-0.00294, -0.0285, 0.1741], [0.707107, -0.707107, 0, 0]], 
    # [[-0.02584, -0.00947, 0.1513], [1, 0, 0, 0]], 
    # [[0.02376, -0.00947, 0.151], [1, 0, 0, 0]], 
    # [[0.00007, -0.00794, 0.1679], [0, -1, 0, 0]], 
    # [[0.00007, 0.02866, 0.1295], [0, -1, 0, 0]], 
    # ]

    # try:
    #     user_input = input("Press Appropriate Candidate: ")
    #     i = int(user_input)
    #     if 0 <= i < len(Human_GraspingPose_Candidates):
    #         offset_pos_local = np.array(Human_GraspingPose_Candidates[i][0])*1000
    #         offset_rot_quat = Human_GraspingPose_Candidates[i][1]
    #         approach_distance = 0.2
    #         approach_offset_local = np.array([0.0, 0.0, approach_distance * 1000])
    #         Gripper_Margin_local = np.array([0.0, 0.0, 0.05 * 1000])
    #         '''거리 기반 공식화 필요'''
    #         if i == 0 or i == 1 or i == 2:
    #             gripper_dis = 0.7
    #         else:
    #             gripper_dis = 0.25
    # except ValueError:
    #     print("Wrong Input")
    #     sys.exit()

    # # Orinetation 반영
    # orientation = get_quaternion_from_euler(initial_angle)
    # rotation_ori_estimation = R.from_quat(orientation)
    # r_offset = R.from_quat(offset_rot_quat)

    # Robot_ROT = r_offset * rotation_ori_estimation

    # # Position 반영 
    # Robot_POS = initial_POS + rotation_ori_estimation.apply(offset_pos_local) + rotation_ori_estimation.apply(Gripper_Margin_local)
    # Robot_POS_Approaching = Robot_POS + rotation_ori_estimation.apply(approach_offset_local)

    # print(f"Calculation Finished")

########################################  5.Grasping  ########################################


# if Robot_POS is not None and Robot_ROT is not None:
#     print("Grasping Start")
#     # Gripper와 Robot End-effector 거리 = 20cm + 1cm(margin)
#     gripper.send_goal(0.0)

#     Robot_ZYZ = Robot_ROT.as_euler('zyz', degrees=True) # Doosan M1509용 ZYZ

#     robot_approaching = [
#                 Robot_POS_Approaching[0], Robot_POS_Approaching[1], Robot_POS_Approaching[2],
#                 Robot_ZYZ[0], Robot_ZYZ[1], Robot_ZYZ[2] ]
#     robot_grasping_pos = [
#                 Robot_POS[0], Robot_POS[1], Robot_POS[2],
#                 Robot_ZYZ[0], Robot_ZYZ[1], Robot_ZYZ[2] ]
#     robot_grasping_pos_go_up = [
#             Robot_POS[0], Robot_POS[1], Robot_POS[2] + 100,
#             Robot_ZYZ[0], Robot_ZYZ[1], Robot_ZYZ[2] ]

#     movel(robot_approaching, v=30, a=60)
#     time.sleep(3)
#     print("Approaching")
#     movel(robot_grasping_pos, v=20, a=40)
#     gripper.send_goal(gripper_dis)
#     print("Grasped")
#     time.sleep(1)
#     movel(robot_grasping_pos_go_up, v=20, a=40)
#     time.sleep(5)
#     gripper.send_goal(0.0)
