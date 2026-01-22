import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

model = YOLO("YOLO_Model/workpiece1_OBB.pt")

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

print("Camera Detected")
profile = pipeline.start(config)
'''Depth 설정'''
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale() 
intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
align_to = rs.stream.color
align = rs.align(align_to)

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
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
                
                distance = depth_frame.get_distance(int(px), int(py))
                camera_coords = rs.rs2_deproject_pixel_to_point(intrinsics, [px, py], distance)
                X_mm, Y_mm, Z_mm = [c * 1000 for c in camera_coords]
                angle_deg = np.degrees(rotation)

                pose_text = f"X:{X_mm:.1f} Y:{Y_mm:.1f} Z:{Z_mm:.1f}"
                cv2.circle(annotated_frame, (int(px), int(py)), 5, (0, 0, 255), -1)
                cv2.putText(annotated_frame, pose_text, (int(px) - 50, int(py) - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(annotated_frame, pose_text, (int(px) - 50, int(py) - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
                best_target_this_frame = {
                    'pose': (X_mm, Y_mm, Z_mm, angle_deg),
                    'conf': max_conf
                }
                cv2.circle(annotated_frame, (int(px), int(py)), 7, (0, 255, 255), -1) 
                cv2.putText(annotated_frame, f"TOP SCORE: {max_conf:.2f}", (int(px)-50, int(py)-40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow('Detection (Press K to Confirm)', annotated_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('k'):
            if best_target_this_frame is not None:
                target_pose = best_target_this_frame['pose']
                print(f"Pose extracted: X={target_pose[0]:.1f}, Y={target_pose[1]:.1f}, Z={target_pose[2]:.1f}, Angle={target_pose[3]:.1f}")
                break 
            else:
                print("No Detection")
finally:
    pipeline.stop()
    cv2.destroyAllWindows()

if target_pose:
    print("Grasping Performed")
