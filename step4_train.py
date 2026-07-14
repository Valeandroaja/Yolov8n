"""
Step 4 — Training YOLO11 (adaptasi dari template Kaggle)
Mengganti pose model → detection model, dataset lokal kasir.

Input : data/dataset/data.yaml
Output: runs/detect/kasir_v1/weights/best.pt

Usage:
    python step4_train.py
    python step4_train.py --model yolo11s.pt --epochs 100
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ─── Config (adaptasi dari template Kaggle) ───────────────────────────────────

DEFAULT_CONFIG = {
    # Model: ganti dari yolo11x-pose.pt → yolo11n.pt (detection)
    # Pilihan: yolo11n (nano/cepat) → yolo11s → yolo11m → yolo11l → yolo11x (akurat)
    "model"      : "yolo11n.pt",

    "data"       : "data/dataset/data.yaml",
    "epochs"     : 30,
    "imgsz"      : 640,
    "batch"      : 4,            # turunkan ke 4 jika RAM < 8GB
    "device"     : "cpu",          # GPU; ganti "cpu" jika tidak ada GPU

    # Optimizer — sama seperti template Kaggle
    "optimizer"  : "AdamW",
    "lr0"        : 0.001,
    "lrf"        : 0.01,
    "cos_lr"     : True,

    # Augmentasi — sama seperti template Kaggle
    "mosaic"     : 1.0,
    "mixup"      : 0.2,
    "hsv_h"      : 0.015,
    "hsv_s"      : 0.7,
    "hsv_v"      : 0.4,
    "degrees"    : 10,
    "translate"  : 0.1,
    "scale"      : 0.5,
    "fliplr"     : 0.5,

    # Lainnya
    "cache"      : True,
    "pretrained" : True,
    "amp"        : True,         # mixed precision (hemat VRAM)
    "patience"   : 15,           # early stopping

    "project"    : "runs/detect",
    "name"       : "kasir_v1",
}


# ─── Trainer ─────────────────────────────────────────────────────────────────

def train(config: dict) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Jalankan: pip install ultralytics")

    data_path = Path(config["data"])
    if not data_path.exists():
        raise FileNotFoundError(
            f"data.yaml tidak ditemukan: {data_path}\n"
            "Jalankan step3_prepare_dataset.py terlebih dahulu."
        )

    log.info("Model    : %s", config["model"])
    log.info("Data     : %s", config["data"])
    log.info("Epochs   : %d | Batch: %d | Device: %s",
             config["epochs"], config["batch"], config["device"])

    model = YOLO(config["model"])

    # Ambil hanya param yang relevan untuk model.train()
    train_params = {k: v for k, v in config.items()
                    if k not in ("model",)}

    results = model.train(**train_params)

    best = Path(config["project"]) / config["name"] / "weights" / "best.pt"
    log.info("[DONE] Best weights → %s", best)
    return best


def validate(weights_path: Path, data_yaml: str) -> None:
    from ultralytics import YOLO
    model   = YOLO(str(weights_path))
    metrics = model.val(data=data_yaml)

    print("\n" + "=" * 55)
    print("  VALIDATION METRICS")
    print("=" * 55)
    print(f"  mAP50      : {metrics.box.map50:.4f}")
    print(f"  mAP50-95   : {metrics.box.map:.4f}")
    print(f"  Precision  : {metrics.box.mp:.4f}")
    print(f"  Recall     : {metrics.box.mr:.4f}")
    print("=" * 55)


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train YOLO11 kasir detector")
    p.add_argument("--model",   default=DEFAULT_CONFIG["model"],
                   help="yolo11n.pt / yolo11s.pt / yolo11m.pt")
    p.add_argument("--data",    default=DEFAULT_CONFIG["data"])
    p.add_argument("--epochs",  type=int,   default=DEFAULT_CONFIG["epochs"])
    p.add_argument("--batch",   type=int,   default=DEFAULT_CONFIG["batch"])
    p.add_argument("--device",  default=DEFAULT_CONFIG["device"])
    p.add_argument("--name",    default=DEFAULT_CONFIG["name"])
    p.add_argument("--val-only", action="store_true")
    p.add_argument("--weights",  default=None)
    args = p.parse_args()

    config = {
        **DEFAULT_CONFIG,
        "model"  : args.model,
        "data"   : args.data,
        "epochs" : args.epochs,
        "batch"  : args.batch,
        "device" : args.device,
        "name"   : args.name,
    }

    if args.val_only:
        if not args.weights:
            raise ValueError("--weights wajib diisi jika pakai --val-only")
        validate(Path(args.weights), args.data)
    else:
        best = train(config)
        validate(best, args.data)

        print("\n  NEXT STEP: python step5_inference.py --video path/ke/video.mp4")
