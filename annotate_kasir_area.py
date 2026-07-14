"""
annotate_kasir_area.py
Tool anotasi interaktif kasir_area dengan klik titik per titik (polygon).
Hasil polygon otomatis dikonvert ke bounding box YOLO.

"""

from pathlib import Path
import cv2
import numpy as np

# ─── Config ──────────────────────────────────────────────────────────────────
IMG_DIR   = "data/raw/images"
LABEL_DIR = "data/raw/labels"
CLASS_ID  = 0   # kasir_area = class 0

# ─── State ───────────────────────────────────────────────────────────────────
points    = []
poly_done = False
img_orig  = None
img_draw  = None


def mouse_callback(event, x, y, flags, param):
    global points, poly_done, img_draw

    if poly_done:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        _redraw()

    elif event == cv2.EVENT_MOUSEMOVE and points:
        _redraw(cursor=(x, y))

    elif event == cv2.EVENT_RBUTTONDOWN and len(points) >= 3:
        poly_done = True
        _redraw()
        print(f"[INFO] Polygon selesai dengan {len(points)} titik")
        print("[INFO] Tekan ENTER untuk simpan, R untuk gambar ulang")


def _redraw(cursor=None):
    global img_draw
    img_draw = img_orig.copy()

    # Gambar polygon yang sudah diklik
    if len(points) >= 2:
        for i in range(len(points) - 1):
            cv2.line(img_draw, points[i], points[i+1], (0, 255, 255), 2)
        if poly_done:
            cv2.line(img_draw, points[-1], points[0], (0, 255, 255), 2)

    # Gambar titik-titik
    for i, pt in enumerate(points):
        cv2.circle(img_draw, pt, 5, (0, 200, 255), -1)
        cv2.putText(img_draw, str(i+1), (pt[0]+8, pt[1]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

    # Garis preview ke cursor
    if cursor and points and not poly_done:
        cv2.line(img_draw, points[-1], cursor, (100, 100, 100), 1)

    # Gambar bounding box hasil konversi polygon
    if poly_done and len(points) >= 3:
        pts  = np.array(points)
        x1   = int(np.min(pts[:, 0]))
        y1   = int(np.min(pts[:, 1]))
        x2   = int(np.max(pts[:, 0]))
        y2   = int(np.max(pts[:, 1]))
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (200, 200, 200), 2)
        cv2.putText(img_draw, "YOLO bbox", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    _draw_instructions(img_draw)


def _draw_instructions(img):
    h = img.shape[0]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - 70), (700, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    cv2.putText(img, "Klik kiri = tambah titik  |  Klik kanan = selesai polygon",
                (10, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(img, "ENTER = simpan ke semua frame  |  R = ulang  |  Q = keluar",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def polygon_to_yolo_bbox(points, img_w, img_h):
    """Konversi polygon ke YOLO bounding box (normalized)."""
    pts = np.array(points)
    x1  = int(np.min(pts[:, 0]))
    y1  = int(np.min(pts[:, 1]))
    x2  = int(np.max(pts[:, 0]))
    y2  = int(np.max(pts[:, 1]))

    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return cx, cy, bw, bh


def save_to_all_labels(cx, cy, bw, bh, label_dir: Path, img_dir: Path):
    """Tambahkan kasir_area ke semua file label."""
    label_dir = Path(label_dir)
    img_dir   = Path(img_dir)
    label_dir.mkdir(parents=True, exist_ok=True)

    yolo_line = f"{CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"

    images = sorted([
        p for p in img_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])

    updated = 0
    for img_path in images:
        lbl_path = label_dir / f"{img_path.stem}.txt"

        if lbl_path.exists():
            lines = [
                l for l in lbl_path.read_text().strip().splitlines()
                if l and not l.startswith(f"{CLASS_ID} ")
            ]
            lines.insert(0, yolo_line)
        else:
            lines = [yolo_line]

        lbl_path.write_text("\n".join(lines))
        updated += 1

    print(f"[DONE] kasir_area disimpan ke {updated} file label")
    print(f"       YOLO coords: cx={cx:.4f} cy={cy:.4f} w={bw:.4f} h={bh:.4f}")


def main():
    global img_orig, img_draw, points, poly_done

    img_dir   = Path(IMG_DIR)
    label_dir = Path(LABEL_DIR)

    images = sorted([
        p for p in img_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])
    if not images:
        raise FileNotFoundError(f"Tidak ada gambar di {img_dir}")

    # Pakai frame tengah sebagai referensi
    sample   = images[len(images) // 2]
    img_orig = cv2.imread(str(sample))
    img_draw = img_orig.copy()
    h, w     = img_orig.shape[:2]

    print(f"[INFO] Frame referensi: {sample.name}")
    print("\nKontrol:")
    print("  Klik kiri  = tambah titik polygon")
    print("  Klik kanan = selesai (minimal 3 titik)")
    print("  ENTER      = simpan ke semua 300 frame")
    print("  R          = gambar ulang")
    print("  Q          = keluar tanpa simpan\n")

    _draw_instructions(img_draw)

    cv2.namedWindow("Anotasi kasir_area", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Anotasi kasir_area", min(w, 1280), min(h, 720))
    cv2.setMouseCallback("Anotasi kasir_area", mouse_callback)

    while True:
        cv2.imshow("Anotasi kasir_area", img_draw)
        key = cv2.waitKey(20) & 0xFF

        if key == 13 and poly_done:       # ENTER
            cx, cy, bw, bh = polygon_to_yolo_bbox(points, w, h)
            save_to_all_labels(cx, cy, bw, bh, label_dir, img_dir)
            print("\n[SELESAI] Lanjut jalankan step3_prepare_dataset.py dan step4_train.py!")
            break

        elif key == ord("r"):             # R = reset
            points    = []
            poly_done = False
            img_draw  = img_orig.copy()
            _draw_instructions(img_draw)
            print("[RESET] Gambar ulang polygon")

        elif key == ord("q"):             # Q = quit
            print("[KELUAR] Tidak ada yang disimpan")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()