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

transformer_node = RobotTransformer()

def get_current_camera_tf(transformer_node):
    try:
        t = transformer_node.tf_buffer.lookup_transform(
            'base_link',
            'camera_link_optical',
            rclpy.time.Time(), 
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


    mesh_base = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200.0, origin=[0, 0, 0])
    mesh_cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=150.0)
    mesh_cam.transform(tf_matrix) 

    o3d.visualization.draw_geometries([pcd, mesh_base, mesh_cam])

    save_path = os.path.join(save_dir, f"view{index:02d}.pcd")
    o3d.io.write_point_cloud(save_path, pcd)

    tf_save_path = os.path.join(save_dir, f"view{index:02d}_tf.npy")
    np.save(tf_save_path, tf_matrix)
    print(f"Saved to {save_path}")


pipeline = rs.pipeline()
config = rs.config() 
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipeline.start(config)
align_to = rs.stream.color
align = rs.align(align_to)

pcd_data_dir = "PCD_Data"
for _ in range(20):
    rclpy.spin_once(transformer_node, timeout_sec = 0.5)
    
tf_base_to_cam = get_current_camera_tf(transformer_node)
print(tf_base_to_cam)
try:
    Multiview_pcd_data(pipeline, align, tf_base_to_cam, pcd_data_dir, 0, duration=1.0)  
finally:
    pipeline.stop()




# import open3d as o3d
# import numpy as np
# import os

# def load_and_visualize_pcd(pcd_path, tf_path):
#     # 1. 파일 존재 여부 확인
#     if not os.path.exists(pcd_path) or not os.path.exists(tf_path):
#         print(f"파일을 찾을 수 없습니다: \n{pcd_path}\n{tf_path}")
#         return

#     # 2. PCD 파일 읽기
#     pcd = o3d.io.read_point_cloud(pcd_path)
#     print(f"로드 완료: {pcd_path}")

#     # 3. TF 행렬(Numpy) 읽기
#     tf_matrix = np.load(tf_path)
#     print(f"TF 행렬 로드 완료:\n{tf_matrix}")

#     # 4. 시각화용 좌표계 생성
#     # 로봇 베이스 좌표계 (Origin)
#     mesh_base = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200.0, origin=[0, 0, 0])
    
#     # 카메라 위치 좌표계
#     mesh_cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=150.0)
#     mesh_cam.transform(tf_matrix)

#     # 5. 시각화 실행
#     print("--- 시각화 창 (Red:X, Green:Y, Blue:Z) ---")
#     print("창을 닫으려면 'q'를 누르세요.")
#     o3d.visualization.draw_geometries([pcd, mesh_base, mesh_cam],
#                                       window_name="PCD Viewer with TF",
#                                       width=1280, height=720)
    
# data_dir = "PCD_Data"  # 데이터가 저장된 폴더명
# view_index = 0         # 보고 싶은 파일의 인덱스 (view00.pcd 등)

# pcd_file = os.path.join(data_dir, f"view{view_index:02d}.pcd")
# tf_file = os.path.join(data_dir, f"view{view_index:02d}_tf.npy")

# load_and_visualize_pcd(pcd_file, tf_file)
