from pathlib import Path
import shutil

VALID_EXTS = {".jpg", ".jpeg", ".png"}

RAW_FULLMATCH = Path("data/yolo_rim/raw/raw_fullmatch")
RAW_B51 = Path("data/yolo_rim/raw/raw_b51")
OUT_DIR = Path("data/yolo_rim/to_label/images")

N_FULLMATCH = 30
N_B51 = 20


def get_images(folder: Path):
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in VALID_EXTS])


def sample_evenly(files, n):
    if len(files) <= n:
        return files
    step = len(files) / n
    selected = []
    for i in range(n):
        idx = int(i * step)
        selected.append(files[idx])
    return selected


def copy_with_prefix(files, prefix):
    for i, img_path in enumerate(files, start=1):
        new_name = f"{prefix}_{i:04d}{img_path.suffix.lower()}"
        dst = OUT_DIR / new_name
        shutil.copy2(img_path, dst)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fullmatch_files = get_images(RAW_FULLMATCH)
    b51_files = get_images(RAW_B51)

    if not fullmatch_files:
        print("[ERROR] raw_fullmatch không có ảnh")
        return

    if not b51_files:
        print("[ERROR] raw_b51 không có ảnh")
        return

    chosen_fullmatch = sample_evenly(fullmatch_files, N_FULLMATCH)
    chosen_b51 = sample_evenly(b51_files, N_B51)

    copy_with_prefix(chosen_fullmatch, "fm")
    copy_with_prefix(chosen_b51, "b51")

    print(f"[DONE] fullmatch: {len(chosen_fullmatch)} ảnh")
    print(f"[DONE] b51: {len(chosen_b51)} ảnh")
    print(f"[DONE] tổng: {len(chosen_fullmatch) + len(chosen_b51)} ảnh")
    print(f"[SAVE TO] {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()