import cv2
from ultralytics import YOLO
from pathlib import Path

MODEL_PATH = "runs/detect/checkpoints/detector/rim_detector/weights/best.pt"
VIDEO_PATH = "uploads/LOS ANGELES LAKERS - GOLDEN STATE WARRIORS - 28.01.2024 - HIGHLIGHTS NBA 23-24.mp4"

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

frame_idx = 0
detected_frames = 0

max_conf_all = 0.0
best_frame = -1

while True:
    ok, frame = cap.read()
    if not ok:
        break
    
    results = model.predict(source=frame, conf=0.01, verbose=False)
    if results and len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            conf = boxes.conf.max().item()
            if conf > max_conf_all:
                max_conf_all = conf
                best_frame = frame_idx
            detected_frames += 1
            
    if frame_idx % 500 == 0:
        print(f"Processed {frame_idx} frames...")
    frame_idx += 1

print(f"Total frames processed: {frame_idx}")
print(f"Total frames with detections (conf >= 0.01): {detected_frames}")
print(f"Maximum confidence found: {max_conf_all:.4f} at frame {best_frame}")
cap.release()
