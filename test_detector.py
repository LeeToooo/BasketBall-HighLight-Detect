from ultralytics import YOLO
from pathlib import Path
import random

MODEL_PATH = "runs/detect/checkpoints/detector/rim_detector/weights/best.pt"
TEST_DIR = Path("data/yolo_rim/to_label/images")
OUTPUT_DIR = "outputs/detector_test"

VALID_EXTS = {".jpg", ".jpeg", ".png"}


def main():
    model = YOLO(MODEL_PATH)

    image_paths = [p for p in TEST_DIR.iterdir() if p.suffix.lower() in VALID_EXTS]
    image_paths = sorted(image_paths)

    if not image_paths:
        print("[ERROR] Khong tim thay anh test")
        return

    sample_paths = random.sample(image_paths, min(10, len(image_paths)))

    for img_path in sample_paths:
        print(f"[TEST] {img_path}")
        model.predict(
            source=str(img_path),
            save=True,
            project=OUTPUT_DIR,
            name="predict",
            exist_ok=True,
            conf=0.25
        )

    print("[DONE] Kiem tra ket qua trong outputs/detector_test/predict")


if __name__ == "__main__":
    main()