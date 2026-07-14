"""
Step 3 — Buat Dataset Split & data.yaml
Setelah annotasi LabelImg selesai, split data 70/30 dan generate config.

Input : data/raw/images/ + data/raw/labels/
Output:
    data/dataset/images/train + val
    data/dataset/labels/train + val
    data/dataset/data.yaml

Usage:
    python step3_prepare_dataset.py
    python step3_prepare_dataset.py --train-ratio 0.8
"""

import argparse
import logging
import random
import shutil
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ─── Config ──────────────────────────────────────────────────────────────────

CLASS_NAMES  = ["kasir_area", "person", "hat", "apron"]
TRAIN_RATIO  = 0.70
RANDOM_SEED  = 42
RAW_IMG_DIR  = "data/raw/images"
RAW_LBL_DIR  = "data/raw/labels"
DATASET_DIR  = "data/dataset"


# ─── Validator ───────────────────────────────────────────────────────────────

def validate_label(label_path: Path) -> tuple[bool, str]:
    """Validasi format YOLO label. Return (valid, reason)."""
    try:
        lines = label_path.read_text().strip().splitlines()
        if not lines:
            return False, "file kosong"
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                return False, f"bukan 5 kolom: '{line}'"
            cls_id = int(parts[0])
            coords = [float(v) for v in parts[1:]]
            if cls_id < 0 or cls_id >= len(CLASS_NAMES):
                return False, f"class_id {cls_id} tidak valid"
            if not all(0.0 <= v <= 1.0 for v in coords):
                return False, f"koordinat di luar [0,1]: {coords}"
    except Exception as e:
        return False, str(e)
    return True, "ok"


# ─── Split ───────────────────────────────────────────────────────────────────

def prepare_dataset(
    raw_img_dir  : str | Path = RAW_IMG_DIR,
    raw_lbl_dir  : str | Path = RAW_LBL_DIR,
    dataset_dir  : str | Path = DATASET_DIR,
    train_ratio  : float      = TRAIN_RATIO,
    seed         : int        = RANDOM_SEED,
) -> dict:
    raw_img_dir = Path(raw_img_dir)
    raw_lbl_dir = Path(raw_lbl_dir)
    dataset_dir = Path(dataset_dir)

    # ── Collect valid pairs
    img_exts = {".jpg", ".jpeg", ".png"}
    images   = sorted([p for p in raw_img_dir.iterdir() if p.suffix.lower() in img_exts])

    valid, skipped, invalid_list = [], 0, []
    for img in images:
        lbl = raw_lbl_dir / f"{img.stem}.txt"
        if not lbl.exists():
            log.warning("Label tidak ada, skip: %s", img.name)
            skipped += 1
            continue
        ok, reason = validate_label(lbl)
        if not ok:
            log.warning("Label tidak valid (%s), skip: %s", reason, lbl.name)
            invalid_list.append(img.name)
            skipped += 1
            continue
        valid.append((img, lbl))

    log.info("Valid pairs: %d | Skipped: %d", len(valid), skipped)

    if not valid:
        raise ValueError("Tidak ada data valid! Pastikan annotasi LabelImg sudah selesai.")

    # ── Shuffle + split
    random.seed(seed)
    random.shuffle(valid)

    n_train     = int(len(valid) * train_ratio)
    train_pairs = valid[:n_train]
    val_pairs   = valid[n_train:]

    # ── Copy ke dataset folder
    splits = {"train": train_pairs, "val": val_pairs}
    for split, pairs in splits.items():
        img_out = dataset_dir / "images" / split
        lbl_out = dataset_dir / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img, lbl in pairs:
            shutil.copy2(img, img_out / img.name)
            shutil.copy2(lbl, lbl_out / lbl.name)

        log.info("%-5s: %d pasang → %s", split.upper(), len(pairs), img_out)

    # ── Class distribution
    class_dist = {i: 0 for i in range(len(CLASS_NAMES))}
    for _, lbl in valid:
        for line in lbl.read_text().strip().splitlines():
            parts = line.strip().split()
            if parts:
                cid = int(parts[0])
                class_dist[cid] = class_dist.get(cid, 0) + 1

    # ── Write data.yaml (format Kaggle template)
    yaml_path = _write_yaml(dataset_dir)

    return {
        "total"     : len(valid),
        "train"     : len(train_pairs),
        "val"       : len(val_pairs),
        "skipped"   : skipped,
        "class_dist": class_dist,
        "yaml_path" : yaml_path,
        "invalid"   : invalid_list,
    }


def _write_yaml(dataset_dir: Path) -> Path:
    """Buat data.yaml sesuai format template Kaggle."""
    cfg = {
        "path"  : str(dataset_dir.resolve()),
        "train" : "images/train",
        "val"   : "images/val",
        "nc"    : len(CLASS_NAMES),
        "names" : CLASS_NAMES,
    }
    yaml_path = dataset_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    log.info("data.yaml → %s", yaml_path)
    return yaml_path


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Prepare dataset split + data.yaml")
    p.add_argument("--raw-imgs",    default=RAW_IMG_DIR)
    p.add_argument("--raw-lbls",    default=RAW_LBL_DIR)
    p.add_argument("--dataset-dir", default=DATASET_DIR)
    p.add_argument("--train-ratio", type=float, default=TRAIN_RATIO)
    args = p.parse_args()

    stats = prepare_dataset(
        raw_img_dir = args.raw_imgs,
        raw_lbl_dir = args.raw_lbls,
        dataset_dir = args.dataset_dir,
        train_ratio = args.train_ratio,
    )

    print("\n" + "=" * 55)
    print("  DATASET SIAP")
    print("=" * 55)
    print(f"  Total valid  : {stats['total']}")
    print(f"  Train        : {stats['train']} ({stats['train']/stats['total']*100:.0f}%)")
    print(f"  Val          : {stats['val']} ({stats['val']/stats['total']*100:.0f}%)")
    print(f"  Skipped      : {stats['skipped']}")
    print()
    print("  Distribusi class:")
    for cid, count in stats["class_dist"].items():
        print(f"    [{cid}] {CLASS_NAMES[cid]:12s}: {count} box")
    print()
    print(f"  data.yaml    : {stats['yaml_path']}")
    print()
    print("  NEXT STEP: python step4_train.py")
    print("=" * 55)
