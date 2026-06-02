import csv
from pathlib import Path

import cv2
import torch
from torch import device, load, no_grad
from torch.cuda import is_available
from torchvision import transforms

from src.models.resnet import generate_resnet

import glob
import os

def get_latest_classifier_path():
    list_of_files = glob.glob('checkpoints/classifier/*.pt')
    if not list_of_files:
        raise Exception("No classifier checkpoints found")
    return max(list_of_files, key=os.path.getctime)

CLASSIFIER_PATH = get_latest_classifier_path()
IMG_SIZE = 128


def build_model(current_device):
    print(f"[INFO] Loading classifier as ResNet18 from {CLASSIFIER_PATH}")

    model = generate_resnet(
        number=18,
        pretrained=False,
        current_device=current_device
    )

    checkpoint = load(CLASSIFIER_PATH, map_location=current_device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def build_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([IMG_SIZE, IMG_SIZE]),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def load_crop_image(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def predict_point(model, transform, image_rgb, current_device):
    x = transform(image_rgb).unsqueeze(0).to(current_device)

    with no_grad():
        logits = model(x)

        if logits.ndim == 2 and logits.shape[1] == 1:
            prob = torch.sigmoid(logits)[0, 0].item()
        else:
            prob = torch.sigmoid(logits.squeeze()).item()

    label = "Point" if prob >= 0.5 else "NoPoint"
    return prob, label


def run_score_crops(csv_input: str, output_dir: str, progress_callback=None):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_output = output_path / "frame_scores.csv"

    current_device = device("cuda:0" if is_available() else "cpu")
    print(f"[INFO] Device: {current_device}")

    model = build_model(current_device)
    transform = build_transform()

    total_rows = 0
    scored_rows = 0
    
    # Count total rows for progress callback
    try:
        with open(csv_input, "r", encoding="utf-8") as f:
            total_input_rows = sum(1 for line in f) - 1 # Subtract header
    except Exception:
        total_input_rows = 1

    with open(csv_input, "r", encoding="utf-8") as f_in, \
         open(csv_output, "w", newline="", encoding="utf-8") as f_out:

        reader = csv.DictReader(f_in)
        writer = csv.writer(f_out)

        writer.writerow([
            "frame_idx",
            "time_sec",
            "crop_path",
            "score_prob",
            "pred_label"
        ])

        for row in reader:
            total_rows += 1

            frame_idx = row["frame_idx"]
            time_sec = row["time_sec"]
            crop_path = row["crop_path"].strip()

            if not crop_path:
                if progress_callback:
                    progress_callback(total_rows, total_input_rows)
                continue

            image_rgb = load_crop_image(crop_path)
            if image_rgb is None:
                if progress_callback:
                    progress_callback(total_rows, total_input_rows)
                continue

            prob, label = predict_point(
                model=model,
                transform=transform,
                image_rgb=image_rgb,
                current_device=current_device,
            )

            writer.writerow([
                frame_idx,
                time_sec,
                crop_path,
                round(prob, 6),
                label
            ])
            scored_rows += 1
            
            if progress_callback:
                progress_callback(total_rows, total_input_rows)

            if scored_rows % 50 == 0:
                print(f"[INFO] Scored crops: {scored_rows}")

    print(f"[DONE] Total rows in {csv_input}: {total_rows}")
    print(f"[DONE] Total scored crops: {scored_rows}")
    print(f"[DONE] Saved: {csv_output}")
    
    return str(csv_output)


if __name__ == "__main__":
    run_score_crops(
        csv_input="outputs/rim_inference/rim_boxes.csv", 
        output_dir="outputs/score_inference"
    )