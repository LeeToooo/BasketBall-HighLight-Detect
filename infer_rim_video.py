import cv2
import csv
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = "runs/detect/checkpoints/detector/rim_detector/weights/best.pt"

CONF_THRES = 0.25
SAVE_EVERY_N_FRAMES = 5   # de giam so crop phai luu
EXPAND_RATIO = 0.5        # crop rong hon box mot chut

def expand_box(x1, y1, x2, y2, img_w, img_h, ratio=0.5):
    bw = x2 - x1
    bh = y2 - y1

    pad_w = int(bw * ratio)
    pad_h = int(bh * ratio)

    nx1 = max(0, x1 - pad_w)
    ny1 = max(0, y1 - pad_h)
    nx2 = min(img_w, x2 + pad_w)
    ny2 = min(img_h, y2 + pad_h)

    return nx1, ny1, nx2, ny2

def run_rim_inference(video_path, output_dir="outputs/rim_inference", max_frames=None, progress_callback=None, conf_thres=CONF_THRES):
    output_path = Path(output_dir)
    crops_dir = output_path / "crops"
    csv_path = output_path / "rim_boxes.csv"
    
    output_path.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    # Use yolov8n.pt if the custom weights are not present
    model_weights = MODEL_PATH if Path(MODEL_PATH).exists() else "yolov8n.pt"
    model = YOLO(model_weights)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Khong mo duoc video: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] FPS = {fps}")
    print(f"[INFO] Total frames = {total_frames}")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_idx", "time_sec",
            "x1", "y1", "x2", "y2",
            "conf",
            "crop_path"
        ])

        frame_idx = 0
        saved_crops = 0

        while True:
            if max_frames is not None and frame_idx >= max_frames:
                break

            # Use cap.grab() to skip decoding frames we don't need, saving massive CPU usage
            if frame_idx % SAVE_EVERY_N_FRAMES != 0:
                ok = cap.grab()
                if not ok: break
                frame = None
            else:
                ok, frame = cap.read()
                if not ok: break

            time_sec = frame_idx / fps
            best_box = None
            best_conf = -1.0

            if frame is not None:
                h, w = frame.shape[:2]
                results = model.predict(
                    source=frame,
                    conf=conf_thres,
                    verbose=False,
                    device=0
                )

                if results and len(results) > 0:
                    boxes = results[0].boxes
                    if boxes is not None and len(boxes) > 0:
                        for box in boxes:
                            conf = float(box.conf[0].item())
                            xyxy = box.xyxy[0].tolist()
                            x1, y1, x2, y2 = map(int, xyxy)

                            if conf > best_conf:
                                best_conf = conf
                                best_box = (x1, y1, x2, y2)

            crop_path_str = ""

            if best_box is not None and frame is not None:
                x1, y1, x2, y2 = best_box
                ex1, ey1, ex2, ey2 = expand_box(
                    x1, y1, x2, y2, w, h, ratio=EXPAND_RATIO
                )

                crop = frame[ey1:ey2, ex1:ex2]
                crop_name = f"frame_{frame_idx:06d}.jpg"
                crop_file_path = crops_dir / crop_name

                if crop.size > 0:
                    cv2.imwrite(str(crop_file_path), crop)
                    crop_path_str = str(crop_file_path)
                    saved_crops += 1

                writer.writerow([
                    frame_idx,
                    round(time_sec, 3),
                    x1, y1, x2, y2,
                    round(best_conf, 4),
                    crop_path_str
                ])
            else:
                writer.writerow([
                    frame_idx,
                    round(time_sec, 3),
                    "", "", "", "",
                    "",
                    ""
                ])

            if progress_callback:
                progress_callback(frame_idx, total_frames if max_frames is None else min(max_frames, total_frames))
                
            if frame_idx % 100 == 0:
                print(f"[INFO] Processed frame {frame_idx}/{total_frames}")

            frame_idx += 1

    cap.release()
    print(f"[DONE] Luu csv tai: {csv_path}")
    print(f"[DONE] So crop da luu: {saved_crops}")
    print(f"[DONE] Thu muc crop: {crops_dir}")
    
    return str(csv_path)

if __name__ == "__main__":
    run_rim_inference(video_path="data/full_matches/Full_Match-001.mp4", output_dir="outputs/rim_inference", max_frames=300)