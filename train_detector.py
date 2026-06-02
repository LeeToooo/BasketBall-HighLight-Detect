from ultralytics import YOLO
import torch

DATASET_YAML = "data/yolo_rim/dataset.yaml"
MODEL_NAME = "yolov8n.pt"
PROJECT_DIR = "checkpoints/detector"
RUN_NAME = "rim_detector"


def main():
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = YOLO(MODEL_NAME)

    model.train(
        data=DATASET_YAML,
        epochs=30,
        imgsz=640,
        batch=8 if device != "cpu" else 4,
        patience=5,
        device=device,
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,
        workers=0,
    )


if __name__ == "__main__":
    main()