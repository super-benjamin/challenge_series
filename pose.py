import cv2
import mediapipe as mp

# 初始化MediaPipe姿态估计算法
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

# 打开视频文件
cap = cv2.VideoCapture('pose.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 将BGR图像转换为RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 将图像传递给MediaPipe姿态估计算法
    results = pose.process(image_rgb)

    # 绘制检测到的姿态
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # 显示处理后的图像
    cv2.imshow('Pose Estimation', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
