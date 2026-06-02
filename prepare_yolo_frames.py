import cv2
import argparse
from pathlib import Path

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

def extract_frames_from_video(video_path: Path, output_dir: Path, max_images: int = 2000):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[SKIP] Cannot open: {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        print(f"[SKIP] Invalid total frames: {video_path}")
        return 0

    frame_interval = max(1, total_frames // max_images)

    saved = 0
    frame_idx = 0
    safe_stem = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in video_path.stem)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_interval == 0:
            time_sec = frame_idx / fps
            time_str = f"{time_sec:.1f}".replace(".", "_")
            filename = f"{safe_stem}_t{time_str}.jpg"
            out_path = output_dir / filename

            success, buffer = cv2.imencode(".jpg", frame)
            if success:
                out_path.write_bytes(buffer.tobytes())
                saved += 1

            if saved >= max_images:
                break

        frame_idx += 1

    cap.release()
    print(f"[DONE] {video_path.name}: saved {saved} frames (max={max_images}) from {total_frames} total frames")
    return saved

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Folder chua cac video hoac clip")
    parser.add_argument("--dst", default="data/yolo_rim/images/raw_fullmatch", help="Folder output frames")
    parser.add_argument("--max-images", type=int, default=2000, help="So anh toi da moi video")
    args = parser.parse_args()

    src_dir = Path(args.src)
    dst_dir = Path(args.dst)
    dst_dir.mkdir(parents=True, exist_ok=True)

    total_saved = 0
    for file in src_dir.rglob("*"):
        if file.suffix.lower() in VIDEO_EXTS:
            total_saved += extract_frames_from_video(file, dst_dir, max_images=args.max_images)

    print(f"\nTong so frame da trich: {total_saved}")
    print(f"Luu tai: {dst_dir.resolve()}")

if __name__ == "__main__":
    main()