"""
Extract Frames dari Video Kasir
Untuk video berukuran besar, extract frame secara efisien.

Input : video file (mp4/avi/mov) di laptop lokal
Output: data/raw/images/*.jpg  (target ~300 frame)

Usage:
    python step1_extract_frames.py --video "C:/path/to/kasir.mp4"
    python step1_extract_frames.py --video "C:/path/to/kasir.mp4" --max 300 --fps 1
"""

import argparse
import logging
from pathlib import Path

import cv2

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ─── Config ──────────────────────────────────────────────────────────────────

DEFAULT_OUT      = "data/raw/images"
DEFAULT_FPS      = 1      # 1 frame per detik — cukup untuk video kasir
DEFAULT_MAX      = 300    # target 300 frame
IMG_QUALITY      = 95


# ─── Core ────────────────────────────────────────────────────────────────────

def extract_frames(
    video_path : str | Path,
    output_dir : str | Path = DEFAULT_OUT,
    target_fps : float = DEFAULT_FPS,
    max_frames : int   = DEFAULT_MAX,
) -> int:
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video tidak ditemukan: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Tidak bisa membuka video: {video_path}")

    source_fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec   = total_frames / source_fps
    frame_interval = max(1, int(source_fps / target_fps))

    log.info("Video    : %s", video_path.name)
    log.info("Durasi   : %.1f detik (%.1f menit)", duration_sec, duration_sec / 60)
    log.info("FPS asli : %.1f  |  Ambil setiap %d frame", source_fps, frame_interval)
    log.info("Target   : %d frame → %s", max_frames, output_dir)

    stem   = video_path.stem
    saved  = 0
    total  = 0

    while saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        total += 1
        if total % frame_interval != 0:
            continue

        out_path = output_dir / f"{stem}_{total:07d}.jpg"
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, IMG_QUALITY])
        saved += 1

        if saved % 50 == 0:
            log.info("  Progress: %d/%d frame tersimpan …", saved, max_frames)

    cap.release()
    log.info("[DONE] %d frame disimpan → %s", saved, output_dir)
    return saved


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Extract frame dari video kasir")
    p.add_argument("--video", required=True, help="Path ke file video")
    p.add_argument("--output", default=DEFAULT_OUT)
    p.add_argument("--fps",   type=float, default=DEFAULT_FPS, help="Frame per detik yang diambil")
    p.add_argument("--max",   type=int,   default=DEFAULT_MAX, help="Maksimal frame")
    args = p.parse_args()

    count = extract_frames(args.video, args.output, args.fps, args.max)

    print("\n" + "=" * 50)
    print(f"  SELESAI: {count} frame disimpan")
    print(f"  Folder  : {args.output}")
    print(f"\n  NEXT STEP:")
    print(f"  Jalankan: python step2_auto_label.py")
    print("=" * 50)
