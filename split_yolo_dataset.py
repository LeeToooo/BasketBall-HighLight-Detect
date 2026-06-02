from pathlib import Path
import random
import shutil

random.seed(42)

IMG_SRC = Path("data/yolo_rim/to_label/images")
LBL_SRC = Path("data/yolo_rim/to_label/labels")

TRAIN_IMG = Path("data/yolo_rim/images/train")
VAL_IMG = Path("data/yolo_rim/images/val")
TRAIN_LBL = Path("data/yolo_rim/labels/train")
VAL_LBL = Path("data/yolo_rim/labels/val")

VALID_EXTS = {".jpg", ".jpeg", ".png"}
VAL_RATIO = 0.2


def clear_dir(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    for f in folder.iterdir():
        if f.is_file():
            f.unlink()


def collect_pairs():
    pairs = []
    for img_path in sorted(IMG_SRC.iterdir()):
        if img_path.suffix.lower() not in VALID_EXTS:
            continue
        lbl_path = LBL_SRC / f"{img_path.stem}.txt"
        if lbl_path.exists():
            pairs.append((img_path, lbl_path))
        else:
            print(f"[SKIP] Missing label: {img_path.name}")
    return pairs


def copy_pairs(pairs, img_dst: Path, lbl_dst: Path):
    for img_path, lbl_path in pairs:
        shutil.copy2(img_path, img_dst / img_path.name)
        shutil.copy2(lbl_path, lbl_dst / lbl_path.name)


def main():
    clear_dir(TRAIN_IMG)
    clear_dir(VAL_IMG)
    clear_dir(TRAIN_LBL)
    clear_dir(VAL_LBL)

    pairs = collect_pairs()
    random.shuffle(pairs)

    n_val = max(1, int(len(pairs) * VAL_RATIO))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    copy_pairs(train_pairs, TRAIN_IMG, TRAIN_LBL)
    copy_pairs(val_pairs, VAL_IMG, VAL_LBL)

    print(f"[DONE] Total: {len(pairs)}")
    print(f"[DONE] Train: {len(train_pairs)}")
    print(f"[DONE] Val: {len(val_pairs)}")


if __name__ == "__main__":
    main()