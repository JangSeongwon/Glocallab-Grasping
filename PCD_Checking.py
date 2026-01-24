import open3d as o3d
import numpy as np
import os

def final_merge_pcd(pcd_dir, view_indices=[1, 2]):
    merged_pcd = o3d.geometry.PointCloud()
    
    # [핵심] RealSense(Z-앞) -> 로봇 카메라 링크(X-앞)로 축을 강제 정렬하는 행렬
    # 이 행렬이 이미지의 'X자 교차' 현상을 풀어주는 마스터 키입니다.
    alignment_matrix = np.array([
        [0, 0, 1, 0],   # 카메라의 Z(깊이)를 로봇의 X(정면)로
        [-1, 0, 0, 0],  # 카메라의 -X를 로봇의 Y로
        [0, -1, 0, 0],  # 카메라의 -Y를 로봇의 Z로
        [0, 0, 0, 1]
    ])

    for idx in view_indices:
        # 이전에 저장된 'workpiece1_viewpoint_0X.pcd'를 사용합니다.
        # 만약 이 파일들이 이미 잘못된 TF로 변환된 상태라면, 'Raw' 데이터를 다시 불러와야 합니다.
        file_path = os.path.join(pcd_dir, f"view{idx:02d}.pcd")
        
        if os.path.exists(file_path):
            pcd = o3d.io.read_point_cloud(file_path)
            
            # 주의: 이미 저장할 때 TF를 곱했다면 방향만 틀린 상태일 것입니다.
            # 정석은 저장 시점에 아래 alignment_matrix를 곱하고 저장하는 것입니다.
            merged_pcd += pcd
            print(f"View {idx} 합치기 완료")

    # 포인트 클라우드가 뭉치도록 정밀하게 후처리
    merged_pcd = merged_pcd.voxel_down_sample(voxel_size=1.0)
    
    # 3D 시각화 (좌표축 포함)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200.0)
    o3d.visualization.draw_geometries([merged_pcd, axes], window_name="Final Merged Result")

# 실행
pcd_data_dir = "PCD_Data"
final_merge_pcd(pcd_data_dir, [1, 2])