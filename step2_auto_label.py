"""
Step 2 — Auto-Label dengan YOLO Pretrained
Generate label awal untuk class 'person' secara otomatis.
Label hat, apron, kasir_area tetap manual di LabelImg.

Input : data/raw/images/*.jpg
Output: data/raw/labels/*.txt  (YOLO format, siap diedit di LabelImg)

Class mapping final:
    0: kasir_area
    1: person
    2: hat
    3: apron

Usage:
    python step2_auto_label.py
    python step2_auto_label.py --conf 0.4 --preview
"""

import argparse
import logging
from pathlib import Path

import cv2

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ─── Config ──────────────────────────────────────────────────────────────────

IMG_DIR       = "data/raw/images"
LABEL_DIR     = "data/raw/labels"
PREVIEW_DIR   = "data/raw/previews"

# Class ID final (EDIT jika urutan berubah)
CLASS_NAMES   = ["kasir_area", "person", "hat", "apron"]
PERSON_ID     = 1   # ID untuk 'person' di dataset kita

CONF_THRESHOLD = 0.45
YOLO_PERSON_ID = 0   # class 0 = person di COCO (model pretrained)

# Warna preview per class (BGR)
CLASS_COLORS = {
    0: (200, 200, 200),  # kasir_area — abu-abu
    1: (0, 200, 0),      # person — hijau
    2: (0, 165, 255),    # hat — oranye
    3: (255, 0, 255),    # apron — pink
}


# ─── Auto Labeler ────────────────────────────────────────────────────────────

def auto_label(
    img_dir   : str | Path = IMG_DIR,
    label_dir : str | Path = LABEL_DIR,
    conf      : float = CONF_THRESHOLD,
    preview   : bool  = False,
) -> dict:
    """
    Jalankan YOLO pretrained untuk deteksi 'person' di semua frame.
    Simpan hasil sebagai YOLO .txt label (class_id cx cy w h).
    
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Jalankan: pip install ultralytics")

    img_dir   = Path(img_dir)
    label_dir = Path(label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([
        p for p in img_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])

    if not images:
        raise FileNotFoundError(f"Tidak ada gambar di {img_dir}")

    log.info("Loading YOLO pretrained (yolo11n.pt) …")
    model = YOLO("yolo11n.pt")   # download otomatis ~6MB

    if preview:
        preview_dir = Path(PREVIEW_DIR)
        preview_dir.mkdir(parents=True, exist_ok=True)

    stats = {"total": len(images), "labeled": 0, "person_boxes": 0, "empty": 0}

    log.info("Auto-label %d gambar …", len(images))

    for i, img_path in enumerate(images, 1):
        results = model(str(img_path), conf=conf, classes=[YOLO_PERSON_ID], verbose=False)[0]

        lines = []
        for box in results.boxes:
            cx, cy, w, h = box.xywhn[0].tolist()   # normalized xywh
            lines.append(f"{PERSON_ID} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            stats["person_boxes"] += 1

        label_path = label_dir / f"{img_path.stem}.txt"
        label_path.write_text("\n".join(lines))

        if not lines:
            stats["empty"] += 1

        stats["labeled"] += 1

        # Preview dengan bounding box
        if preview and i <= 20:   # preview 20 frame pertama saja
            _save_preview(img_path, lines, preview_dir)

        if i % 50 == 0:
            log.info("  Progress: %d/%d | boxes: %d", i, len(images), stats["person_boxes"])

    log.info("[DONE] %d gambar dilabeli | %d person box | %d frame kosong",
             stats["labeled"], stats["person_boxes"], stats["empty"])
    return stats


def _save_preview(img_path: Path, yolo_lines: list[str], out_dir: Path) -> None:
    """Simpan gambar dengan bounding box untuk review."""
    frame = cv2.imread(str(img_path))
    if frame is None:
        return

    h, w = frame.shape[:2]
    for line in yolo_lines:
        parts = line.strip().split()
        cls_id = int(parts[0])
        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imwrite(str(out_dir / img_path.name), frame)


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Auto-label person menggunakan YOLO pretrained")
    p.add_argument("--img-dir",   default=IMG_DIR)
    p.add_argument("--label-dir", default=LABEL_DIR)
    p.add_argument("--conf",      type=float, default=CONF_THRESHOLD,
                   help="Confidence threshold (default: 0.45)")
    p.add_argument("--preview",   action="store_true",
                   help="Simpan preview gambar dengan bounding box")
    args = p.parse_args()

    stats = auto_label(args.img_dir, args.label_dir, args.conf, args.preview)

    print("\n" + "=" * 55)
    print("  AUTO-LABEL SELESAI")
    print("=" * 55)
    print(f"  Gambar diproses : {stats['total']}")
    print(f"  Label dibuat    : {stats['labeled']}")
    print(f"  Person box      : {stats['person_boxes']}")
    print(f"  Frame kosong    : {stats['empty']} (tidak ada orang)")
    print()
    print("  NEXT STEP — Buka LabelImg dan:")
    print("  1. Load images dari : data/raw/images/")
    print("  2. Load labels dari : data/raw/labels/")
    print("  3. Tambah/koreksi label untuk:")
    print("     - kasir_area (kotak area kasir)")
    print("     - hat        (topi kasir)")
    print("     - apron      (apron/celemek)")
    print("  4. Koreksi box 'person' yang salah")
    print()
    if args.preview:
        print(f"  Preview tersimpan di: data/raw/previews/")
    print("=" * 55)
