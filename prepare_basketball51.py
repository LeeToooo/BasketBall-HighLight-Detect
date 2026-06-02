import cv2
import argparse
from pathlib import Path

MAKE_LABELS = {"2p1", "3p1", "ft1", "mp1"}
MISS_LABELS = {"2p0", "3p0", "ft0", "mp0"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

def infer_label_from_path(path: Path):
    parts = [p.lower() for p in path.parts]
    for p in parts:
        if p in MAKE_LABELS:
            return "Point"
        if p in MISS_LABELS:
            return "NoPoint"
    return None

def extract_frames(video_path: Path, out_dir: Path, frames_per_clip: int = 4):
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return 0

    # Lấy frame ở 40% cuối clip để tăng khả năng gần khoảnh khắc ghi điểm
    start_idx = max(0, int(total_frames * 0.6))
    end_idx = max(start_idx + 1, total_frames - 1)

    if frames_per_clip == 1:
        indices = [(start_idx + end_idx) // 2]
    else:
        step = max(1, (end_idx - start_idx) // (frames_per_clip - 1))
        indices = [min(end_idx, start_idx + i * step) for i in range(frames_per_clip)]

    saved = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        filename = f"{video_path.stem}_f{idx}.jpg"
        save_path = out_dir / filename
        cv2.imwrite(str(save_path), frame)
        saved += 1

    cap.release()
    return saved

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Thư mục gốc của dataset Basketball-51 đã giải nén")
    parser.add_argument("--dst", default="data/crops", help="Thư mục output")
    parser.add_argument("--frames-per-clip", type=int, default=4, help="Số frame trích từ mỗi clip")
    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)
    point_dir = dst_root / "Point"
    nopoint_dir = dst_root / "NoPoint"
    point_dir.mkdir(parents=True, exist_ok=True)
    nopoint_dir.mkdir(parents=True, exist_ok=True)

    total_videos = 0
    total_saved = 0
    skipped = 0

    for file in src_root.rglob("*"):
        if file.suffix.lower() not in VIDEO_EXTS:
            continue

        label = infer_label_from_path(file)
        if label is None:
            skipped += 1
            continue

        out_dir = point_dir if label == "Point" else nopoint_dir
        saved = extract_frames(file, out_dir, frames_per_clip=args.frames_per_clip)
        total_videos += 1
        total_saved += saved

    print(f"Done.")
    print(f"Videos processed: {total_videos}")
    print(f"Images saved: {total_saved}")
    print(f"Videos skipped (không đoán được label từ path): {skipped}")
    print(f"Output folder: {dst_root.resolve()}")

if __name__ == "__main__":
    main()