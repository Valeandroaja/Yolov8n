"""
Deteksi objek + tentukan apakah orang di area kasir adalah kasir atau bukan.

Logika:
    Person di kasir_area + hat AND apron terdeteksi     → KASIR ✅
    Person di kasir_area + hat saja (conf tinggi)       → KASIR ✅
    Person di kasir_area + shirt sangat jelas + 1 sinyal pendukung → KASIR ✅
    Person di kasir_area + tidak memenuhi syarat kasir  → pengunjung ⚠️
    Person di luar kasir_area                           → pengunjung

"""

import argparse
import logging
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ~~~~~CONFIG~~~~~

CLASS_NAMES    = ["kasir_area", "person"]
CLS_KASIR_AREA = 0
CLS_PERSON     = 1

CONF_THRESHOLD    = 0.45
IOU_THRESHOLD     = 0.45
OVERLAP_THRESHOLD = 0.3

# Confidence minimum hat untuk kasir tanpa apron
HAT_ONLY_CONF_THRESHOLD = 0.70

# Shirt confidence sangat tinggi yang bisa "menyelamatkan" kasir saat
# hat/apron lemah (mis. sudut membelakangi kamera) tapi tidak nol.
SHIRT_STRONG_CONF_THRESHOLD = 0.85
SHIRT_RESCUE_MIN_SUPPORT    = 0.30   # minimal hat_conf ATAU apron_conf

# ~~~~~Pose (MediaPipe) config~~~~~
POSE_MODEL_DIR  = Path(__file__).resolve().parent / "models"
POSE_MODEL_PATH = POSE_MODEL_DIR / "pose_landmarker_lite.task"
POSE_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
POSE_MIN_DET_CONF      = 0.3
POSE_MIN_PRESENCE_CONF = 0.3
POSE_LANDMARK_VIS_MIN  = 0.3   # visibility minimum supaya landmark dipakai


# ~~~~~Warna BGR untuk visualisasi~~~~~
COLOR = {
    "kasir"      : (0, 200,   0),    # hijau  — kasir valid
    "pengunjung" : (200, 200,  0),   # kuning — pengunjung / bukan kasir
    "kasir_area" : (200, 200, 200),  # abu-abu — overlay area kasir
    "hat"        : (0, 165, 255),    # oranye — indikator topi
    "apron"      : (255,  0, 255),   # pink   — indikator apron
    "shirt"      : (80,  80,  80),   # [P6] abu-abu gelap — indikator shirt
    "pose"       : (0, 255, 255),    # kuning terang — debug pose landmark
}

# Warna topi crew HokBen (coklat tua)
# [P8] Rentang "navy" dihapus — terbukti overlap dengan kemeja/baju
# gelap polos customer (false positive). Warna topi HokBen yang
# sebenarnya dipakai adalah coklat tua saja.
HAT_COLORS_HSV = [
    (5,  22,  35, 180,  18, 115),   # coklat tua (topi HokBen)
]

# Warna apron HokBen (coklat)
APRON_WAIST_BROWN = [
    (5,  22,  50, 200,  25, 130),   # coklat tua-medium
    (10, 28,  40, 180,  35, 160),   # coklat medium
]

# Warna baju kasir (putih / off-white / abu-abu)
SHIRT_LIGHT_HSV = [
    (0, 180, 0, 30, 180, 255),
    (0, 180, 0, 40,  90, 180),
    (0, 180, 0, 50,  60, 130),
]



# ~~~~~POSE ESTIMATOR~~~~~
# fungsinya mencari tahu posisi tubuh orang secara presisi — di mana letak kepala, bahu, dan pinggul orang itu di dalam frame


@dataclass
class PoseKeypoints:
    """Landmark penting (koordinat ABSOLUT dalam frame penuh, satuan piksel)."""
    nose          : tuple[float, float] | None = None
    left_ear      : tuple[float, float] | None = None
    right_ear     : tuple[float, float] | None = None
    left_shoulder : tuple[float, float] | None = None
    right_shoulder: tuple[float, float] | None = None
    left_hip      : tuple[float, float] | None = None
    right_hip     : tuple[float, float] | None = None

    @property
    def valid_head(self) -> bool:
        return self.nose is not None and (self.left_ear is not None or self.right_ear is not None)

    @property
    def valid_torso(self) -> bool:
        return (self.left_shoulder is not None or self.right_shoulder is not None) and \
               (self.left_hip is not None or self.right_hip is not None)


class PoseEstimator:
    """
    Wrapper tipis di atas MediaPipe Pose Landmarker.

    Dirancang agar:
    - Gagal dengan aman (return None) jika model/dependency tidak ada,
      sehingga caller selalu bisa fallback ke metode persentase-bbox.
    - Model di-download otomatis sekali saja ke POSE_MODEL_PATH.
    """

    def __init__(self, enabled: bool = True):
        self.enabled   = enabled
        self._detector = None
        if not enabled:
            return
        try:
            self._init_detector()
        except Exception as e:
            log.warning("Pose estimator tidak aktif (%s). Fallback ke mode persentase-bbox.", e)
            self.enabled = False

    def _init_detector(self) -> None:
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            PoseLandmarker, PoseLandmarkerOptions, RunningMode,
        )

        if not POSE_MODEL_PATH.exists():
            log.info("Model pose belum ada, mengunduh ke %s ...", POSE_MODEL_PATH)
            POSE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(POSE_MODEL_URL, str(POSE_MODEL_PATH))
            log.info("Unduh model pose selesai.")

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
            running_mode=RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=POSE_MIN_DET_CONF,
            min_pose_presence_confidence=POSE_MIN_PRESENCE_CONF,
        )
        self._detector = PoseLandmarker.create_from_options(options)
        self._PoseLandmark = __import__(
            "mediapipe.tasks.python.vision", fromlist=["PoseLandmark"]
        ).PoseLandmark

    def estimate(self, frame: np.ndarray, bbox: tuple) -> PoseKeypoints | None:
        """
        Jalankan pose landmarker pada crop person (bbox), kembalikan
        keypoint penting dalam koordinat ABSOLUT frame penuh.
        Return None jika pose tidak aktif atau tidak ada pose terdeteksi.
        """
        if not self.enabled or self._detector is None:
            return None

        x1, y1, x2, y2 = bbox
        crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if crop.size == 0:
            return None
        ch, cw = crop.shape[:2]

        import mediapipe as mp
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        try:
            result = self._detector.detect(mp_image)
        except Exception as e:
            log.debug("Pose detect gagal: %s", e)
            return None

        if not result.pose_landmarks:
            return None

        lm = result.pose_landmarks[0]
        PL = self._PoseLandmark

        def _get(idx: int) -> tuple[float, float] | None:
            p = lm[idx]
            if p.visibility is not None and p.visibility < POSE_LANDMARK_VIS_MIN:
                return None
            # Koordinat landmark MediaPipe sudah ternormalisasi [0,1]
            # relatif terhadap crop. Konversi ke absolut frame penuh.
            return (x1 + p.x * cw, y1 + p.y * ch)

        return PoseKeypoints(
            nose           = _get(PL.NOSE.value),
            left_ear       = _get(PL.LEFT_EAR.value),
            right_ear      = _get(PL.RIGHT_EAR.value),
            left_shoulder  = _get(PL.LEFT_SHOULDER.value),
            right_shoulder = _get(PL.RIGHT_SHOULDER.value),
            left_hip       = _get(PL.LEFT_HIP.value),
            right_hip      = _get(PL.RIGHT_HIP.value),
        )



# ~~~~~FUNGSI DETEKSI WARNA~~~~~

def _color_ratio(hsv_crop: np.ndarray, color_ranges: list[tuple]) -> float:
    if hsv_crop.size == 0:
        return 0.0
    mask = np.zeros(hsv_crop.shape[:2], dtype=np.uint8)
    for (hl, hh, sl, sh, vl, vh) in color_ranges:
        mask |= cv2.inRange(hsv_crop,
                            np.array([hl, sl, vl], dtype=np.uint8),
                            np.array([hh, sh, vh], dtype=np.uint8))
    px = max(hsv_crop.shape[0] * hsv_crop.shape[1], 1)
    return float(np.count_nonzero(mask)) / px


def _hat_color_ratio_and_uniformity(hsv_crop: np.ndarray) -> tuple[float, float]:
    """Rasio piksel warna topi + skor uniformity (solid color) pada satu crop."""
    if hsv_crop.size == 0:
        return 0.0, 0.0
    mask = np.zeros(hsv_crop.shape[:2], dtype=np.uint8)
    for (hl, hh_v, sl, sh, vl, vh) in HAT_COLORS_HSV:
        mask |= cv2.inRange(hsv_crop,
                            np.array([hl, sl, vl], dtype=np.uint8),
                            np.array([hh_v, sh, vh], dtype=np.uint8))
    px = max(hsv_crop.shape[0] * hsv_crop.shape[1], 1)
    ratio = float(np.count_nonzero(mask)) / px

    ys, xs = np.where(mask > 0)
    if len(ys) >= max(20, int(0.05 * px)):
        hue_vals   = hsv_crop[ys, xs, 0].astype(float)
        hue_var    = float(np.var(hue_vals))
        uniformity = max(0.0, 1.0 - hue_var / 900.0)
    else:
        uniformity = 0.0

    return ratio, uniformity



# ~~~~~DETEKSI HAT~~~~~

def _hat_window_from_pose(kp: PoseKeypoints) -> tuple[int, int, int, int] | None:
    """
    Tentukan kotak pencarian topi dari landmark pose: SELALU di ATAS
    garis telinga/mata, dengan lebar mengikuti jarak antar-telinga.
    Ini yang membuat window topi tidak akan turun ke wajah, karena
    posisinya dikunci relatif ke fitur wajah orang itu sendiri, bukan
    ke persentase tinggi bbox yang bisa salah saat orang menunduk.
    """
    if not kp.valid_head:
        return None

    ear_pts = [p for p in (kp.left_ear, kp.right_ear) if p is not None]
    nose    = kp.nose

    if len(ear_pts) == 2:
        ear_y    = min(ear_pts[0][1], ear_pts[1][1])
        ear_dist = abs(ear_pts[0][0] - ear_pts[1][0])
        cx       = (ear_pts[0][0] + ear_pts[1][0]) / 2.0
    else:
        ear_y    = ear_pts[0][1]
        ear_dist = abs(nose[0] - ear_pts[0][0]) * 2.2
        cx       = nose[0]

    half_w = max(ear_dist * 0.95, 18)
    # tinggi topi diestimasi proporsional ke lebar kepala
    hat_h  = max(ear_dist * 0.85, 16)

    x1 = int(cx - half_w)
    x2 = int(cx + half_w)
    y2 = int(ear_y + ear_dist * 0.12)   # sedikit turun dari garis telinga (karena topi menutupi sebagian telinga atas)
    y1 = int(y2 - hat_h)
    return (x1, y1, x2, y2)


def _detect_hat(
    frame  : np.ndarray,
    bbox   : tuple,
    kp     : PoseKeypoints | None = None,
) -> tuple[bool, float]:
    """
    Jika pose tidak tersedia, fallback ke sliding-window horizontal di
    pita atas bbox (metode lama, kurang presisi tapi tetap berfungsi).

    Returns: (detected, confidence 0-1)
    """
    x1, y1, x2, y2 = bbox
    h = max(y2 - y1, 1)
    w = max(x2 - x1, 1)
    if h < 80 or w < 30:
        return False, 0.0

    pose_window = _hat_window_from_pose(kp) if kp is not None else None

    if pose_window is not None:
        hwx1, hwy1, hwx2, hwy2 = pose_window
        hwx1, hwy1 = max(0, hwx1), max(0, hwy1)
        hwx2, hwy2 = max(hwx1 + 1, hwx2), max(hwy1 + 1, hwy2)
        head_band = frame[hwy1:hwy2, hwx1:hwx2]
        if head_band.size == 0:
            pose_window = None  # fallback di bawah

    if pose_window is not None:
        head_band_hsv = cv2.cvtColor(head_band, cv2.COLOR_BGR2HSV)
        hat_ratio, uniformity = _hat_color_ratio_and_uniformity(head_band_hsv)

        # brim sekitar tepi bawah window topi (lebih kecil bobotnya)
        brim_y1 = hwy2 - int((hwy2 - hwy1) * 0.35)
        brim_y2 = hwy2 + int((hwy2 - hwy1) * 0.25)
        brim_crop = frame[max(0, brim_y1):max(0, brim_y2), hwx1:hwx2]
        brim_score = 0.0
        if brim_crop.size > 0:
            brim_gray  = cv2.cvtColor(brim_crop, cv2.COLOR_BGR2GRAY)
            sobel_brim = cv2.Sobel(brim_gray, cv2.CV_64F, 0, 1, ksize=3)
            brim_score = min(float(np.max(np.abs(sobel_brim))) / 60.0, 1.0)

        confidence = (0.55 * hat_ratio
                    + 0.25 * uniformity
                    + 0.20 * brim_score)
        confidence = min(confidence, 1.0)

        # Window sudah presisi secara geometris (dikunci ke posisi kepala
        # asli), jadi threshold rasio warna boleh sedikit lebih longgar
        # dibanding mode fallback tanpa pose.
        detected = (hat_ratio >= 0.30 and confidence >= 0.35)
        return detected, confidence


    # ~~~~~ FALLBACK: tanpa pose, pakai sliding-window lama ~~~~~
    head_y1 = y1
    head_y2 = y1 + int(h * 0.22)
    head_band = frame[max(0, head_y1):max(0, head_y2), max(0, x1):max(0, x2)]
    if head_band.size == 0:
        return False, 0.0
    head_band_hsv = cv2.cvtColor(head_band, cv2.COLOR_BGR2HSV)

    bh, bw = head_band_hsv.shape[:2]
    win_w  = max(int(bw * 0.40), 10)
    step   = max(int(bw * 0.10), 3)

    best_ratio, best_uniformity, best_x_center = 0.0, 0.0, bw / 2.0
    for wx1 in range(0, max(bw - win_w, 0) + 1, step):
        wx2 = min(wx1 + win_w, bw)
        ratio, uniformity = _hat_color_ratio_and_uniformity(head_band_hsv[:, wx1:wx2])
        if ratio > best_ratio:
            best_ratio, best_uniformity, best_x_center = ratio, uniformity, (wx1 + wx2) / 2.0
    if bw <= win_w:
        best_ratio, best_uniformity = _hat_color_ratio_and_uniformity(head_band_hsv)
        best_x_center = bw / 2.0

    hat_ratio, uniformity = best_ratio, best_uniformity

    brim_y1  = y1 + int(h * 0.16)
    brim_y2  = y1 + int(h * 0.30)
    brim_cx1 = max(0, int(x1 + best_x_center - win_w * 0.6))
    brim_cx2 = min(x2, int(x1 + best_x_center + win_w * 0.6))
    brim_crop = frame[max(0, brim_y1):max(0, brim_y2), brim_cx1:brim_cx2]
    brim_score = 0.0
    if brim_crop.size > 0:
        brim_gray  = cv2.cvtColor(brim_crop, cv2.COLOR_BGR2GRAY)
        sobel_brim = cv2.Sobel(brim_gray, cv2.CV_64F, 0, 1, ksize=3)
        brim_score = min(float(np.max(np.abs(sobel_brim))) / 60.0, 1.0)

    confidence = (0.45 * hat_ratio + 0.20 * uniformity + 0.25 * brim_score)
    confidence = min(confidence, 1.0)

    detected = (hat_ratio >= 0.40 and uniformity >= 0.45) or \
               (hat_ratio >= 0.22 and brim_score >= 0.35)
    detected = detected and confidence >= 0.35

    return detected, confidence


# ~~~~~DETEKSI APRON~~~~~

def _apron_window_from_pose(kp: PoseKeypoints) -> tuple[int, int, int, int] | None:
    """
    Ini menghindari window "nyasar" ke lengan, yang berada di LUAR
    garis bahu dan secara HSV warnanya juga nyaris identik dengan apron
    coklat tua.
    """
    if not kp.valid_torso:
        return None

    shoulders = [p for p in (kp.left_shoulder, kp.right_shoulder) if p is not None]
    hips      = [p for p in (kp.left_hip, kp.right_hip) if p is not None]
    if not shoulders or not hips:
        return None

    shoulder_y = sum(p[1] for p in shoulders) / len(shoulders)
    hip_y      = sum(p[1] for p in hips) / len(hips)
    cx         = sum(p[0] for p in shoulders + hips) / len(shoulders + hips)

    if len(shoulders) == 2:
        shoulder_w = abs(shoulders[0][0] - shoulders[1][0])
    else:
        shoulder_w = abs(hips[0][0] - hips[-1][0]) if len(hips) == 2 else 60.0
    shoulder_w = max(shoulder_w, 30.0)

    # Window pinggang: separuh bawah jarak bahu→pinggul, lebar 70% bahu
    # (cukup sempit supaya tidak melebar sampai ke lengan).
    y1 = int(shoulder_y + (hip_y - shoulder_y) * 0.45)
    y2 = int(hip_y + (hip_y - shoulder_y) * 0.15)
    half_w = shoulder_w * 0.42
    x1 = int(cx - half_w)
    x2 = int(cx + half_w)
    return (x1, y1, x2, y2)


def _detect_apron(
    frame : np.ndarray,
    bbox  : tuple,
    kp    : PoseKeypoints | None = None,
) -> tuple[bool, float]:
    """
    [P9] Deteksi apron HokBen di area pinggang.

    Jika pose tersedia, window dikunci ke garis tengah tubuh di antara
    bahu dan pinggul (lihat _apron_window_from_pose) — sehingga tidak
    akan melebar ke lengan yang berada di luar garis bahu.

    Jika pose tidak tersedia, fallback ke metode lama: cek kolom
    kiri/tengah/kanan pada pita leher & pinggang berdasarkan persentase
    bbox, ambil rasio tertinggi.

    Returns: (detected, confidence 0-1)
    """
    x1, y1, x2, y2 = bbox
    h = max(y2 - y1, 1)
    w = max(x2 - x1, 1)
    if h < 80 or w < 30:
        return False, 0.0

    pose_window = _apron_window_from_pose(kp) if kp is not None else None

    if pose_window is not None:
        wx1, wy1, wx2, wy2 = pose_window
        wx1, wy1 = max(0, wx1), max(0, wy1)
        wx2, wy2 = max(wx1 + 1, wx2), max(wy1 + 1, wy2)
        waist_crop = frame[wy1:wy2, wx1:wx2]
        if waist_crop.size > 0:
            hsv = cv2.cvtColor(waist_crop, cv2.COLOR_BGR2HSV)
            ratio = _color_ratio(hsv, APRON_WAIST_BROWN)

            THRESH, FULL_CONF = 0.15, 0.45
            if ratio < THRESH:
                return False, ratio / THRESH * 0.5
            confidence = min((ratio - THRESH) / (FULL_CONF - THRESH), 1.0)
            return True, 0.5 + confidence * 0.5
        # window pose kosong → lanjut ke fallback

    # ~~~~~ FALLBACK: tanpa pose, pakai metode persentase-bbox lama ~~~~~
    def _ratio_in_columns(hsv: np.ndarray, cx1: int, cx2: int) -> float:
        ch, cw = hsv.shape[:2]
        cx1 = max(0, min(cx1, cw))
        cx2 = max(cx1, min(cx2, cw))
        if cx2 <= cx1:
            return 0.0
        return _color_ratio(hsv[:, cx1:cx2], APRON_WAIST_BROWN)

    def _check_area(y_start: int, y_end: int) -> float:
        crop = frame[max(0, y_start):max(0, y_end), max(0, x1):max(0, x2)]
        if crop.size == 0:
            return 0.0
        hsv    = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        cw     = hsv.shape[1]
        left_ratio   = _ratio_in_columns(hsv, 0,              int(cw * 0.40))
        center_ratio = _ratio_in_columns(hsv, int(cw * 0.20), int(cw * 0.80))
        right_ratio  = _ratio_in_columns(hsv, int(cw * 0.60), cw)
        return max(left_ratio, center_ratio, right_ratio)

    neck_ratio  = _check_area(y1 + int(h * 0.12), y1 + int(h * 0.18))
    waist_ratio = _check_area(y1 + int(h * 0.45), y1 + int(h * 0.85))
    best_ratio  = max(neck_ratio, waist_ratio)

    THRESH, FULL_CONF = 0.18, 0.50
    if best_ratio < THRESH:
        return False, best_ratio / THRESH * 0.5
    confidence = min((best_ratio - THRESH) / (FULL_CONF - THRESH), 1.0)
    return True, 0.5 + confidence * 0.5


# ~~~~~DETEKSI SHIRT~~~~~

def _detect_shirt(frame: np.ndarray, bbox: tuple) -> tuple[bool, float]:
    """
    [P2] Deteksi warna baju kasir — hanya indikator pendukung, tidak
    pernah jadi satu-satunya penentu status kasir.

    Returns: (detected, confidence 0-1)
    """
    x1, y1, x2, y2 = bbox
    h = max(y2 - y1, 1)
    w = max(x2 - x1, 1)
    if h < 80 or w < 30:
        return False, 0.0

    sy1 = y1 + int(h * 0.20)
    sy2 = y1 + int(h * 0.50)
    sx1 = x1 + int(w * 0.15)
    sx2 = x2 - int(w * 0.15)

    torso = frame[max(0, sy1):max(0, sy2), max(0, sx1):max(0, sx2)]
    if torso.size == 0:
        return False, 0.0

    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    match_ratio = _color_ratio(hsv, SHIRT_LIGHT_HSV)

    THRESH    = 0.35
    FULL_CONF = 0.70
    if match_ratio < THRESH:
        return False, match_ratio / THRESH * 0.5
    confidence = min((match_ratio - THRESH) / (FULL_CONF - THRESH), 1.0)
    return True, 0.5 + confidence * 0.5


# ~~~~~DATA MODEL~~~~~

@dataclass
class Detection:
    cls_id : int
    conf   : float
    xyxy   : tuple[int, int, int, int]

    @property
    def name(self) -> str:
        return CLASS_NAMES[self.cls_id] if self.cls_id < len(CLASS_NAMES) else str(self.cls_id)


@dataclass
class FrameStats:
    kasir_count     : int = 0
    non_kasir_count : int = 0
    alerts          : list[str] = field(default_factory=list)


# ~~~~~HELPER~~~~~

def box_inside(inner: tuple, outer: tuple, threshold: float = OVERLAP_THRESHOLD) -> bool:
    ax1, ay1, ax2, ay2 = inner
    bx1, by1, bx2, by2 = outer
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    inner_area = max(1, (ax2 - ax1) * (ay2 - ay1))
    return (inter / inner_area) >= threshold


def _put_label(frame: np.ndarray, text: str, pos: tuple, color: tuple) -> None:
    x, y = pos
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x, y - th - 6), (x + tw + 4, y), color, -1)
    cv2.putText(frame, text, (x + 2, y - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def _draw_alert_banner(frame: np.ndarray, message: str) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 55), (0, 0, 180), -1)
    cv2.putText(frame, message, (12, 38),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)


def _draw_attribute_indicators(
    frame      : np.ndarray,
    person     : Detection,
    hat_det    : bool,
    hat_conf   : float,
    apron_det  : bool,
    apron_conf : float,
    shirt_det  : bool,
    shirt_conf : float,
) -> None:
    """Gambar indikator hat/apron/shirt di bawah box person."""
    x1, y1, x2, y2 = person.xyxy
    offset = 0

    if hat_det:
        tag = f"hat {hat_conf:.2f}"
        _put_label(frame, tag, (x1, y2 + 16 + offset), COLOR["hat"])
        offset += 18

    if apron_det:
        tag = f"apron {apron_conf:.2f}"
        _put_label(frame, tag, (x1, y2 + 16 + offset), COLOR["apron"])
        offset += 18

    # [P2] shirt hanya indikator
    # [P6] warna label shirt abu-abu gelap
    if shirt_det:
        tag = f"shirt {shirt_conf:.2f}"
        _put_label(frame, tag, (x1, y2 + 16 + offset), COLOR["shirt"])


def _draw_pose_debug(frame: np.ndarray, kp: PoseKeypoints | None) -> None:
    """Gambar titik landmark pose untuk debugging visual (opsional)."""
    if kp is None:
        return
    pts = [kp.nose, kp.left_ear, kp.right_ear,
           kp.left_shoulder, kp.right_shoulder, kp.left_hip, kp.right_hip]
    for p in pts:
        if p is not None:
            cv2.circle(frame, (int(p[0]), int(p[1])), 3, COLOR["pose"], -1)


# ~~~~~ANALYZER~~~~~

def analyze_frame(
    frame         : np.ndarray,
    detections    : list[Detection],
    pose_estimator: "PoseEstimator | None" = None,
    debug_pose    : bool = False,
) -> tuple[np.ndarray, FrameStats]:
    annotated = frame.copy()
    stats     = FrameStats()

    areas   = [d for d in detections if d.cls_id == CLS_KASIR_AREA]
    persons = [d for d in detections if d.cls_id == CLS_PERSON]

    # Gambar area kasir (overlay transparan)
    for area in areas:
        x1, y1, x2, y2 = area.xyxy
        overlay = annotated.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR["kasir_area"], -1)
        cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0, annotated)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR["kasir_area"], 2)
        _put_label(annotated, "kasir_area", (x1, y1), COLOR["kasir_area"])

    # Analisis tiap orang
    for person in persons:
        in_area = any(box_inside(person.xyxy, area.xyxy) for area in areas)

        kp = None
        if pose_estimator is not None and pose_estimator.enabled:
            kp = pose_estimator.estimate(frame, person.xyxy)

        hat_det,   hat_conf   = _detect_hat(frame, person.xyxy, kp)
        apron_det, apron_conf = _detect_apron(frame, person.xyxy, kp)
        shirt_det, shirt_conf = _detect_shirt(frame, person.xyxy)

        # [P1][P7] Kasir jika salah satu dari:
        #   (a) hat AND apron terdeteksi
        #   (b) hat confidence sangat tinggi sendirian (HAT_ONLY_CONF_THRESHOLD)
        #   (c) hat AND shirt terdeteksi (kombinasi seragam lengkap)
        #   (d) shirt confidence SANGAT tinggi + minimal satu indikator
        #       lain (hat ATAU apron) terdeteksi walau confidence marginal
        #       — menangani kasus membelakangi kamera dari dekat.
        # [P2] shirt tetap bukan satu-satunya penentu.
        is_kasir = (
            (hat_det and apron_det)
            or (hat_conf >= HAT_ONLY_CONF_THRESHOLD)
            or (hat_det and shirt_det)
            or (shirt_conf >= SHIRT_STRONG_CONF_THRESHOLD
                and (hat_conf >= SHIRT_RESCUE_MIN_SUPPORT
                     or apron_conf >= SHIRT_RESCUE_MIN_SUPPORT))
        )

        if not in_area:
            # Di luar area kasir → selalu pengunjung
            label = "pengunjung"
            color = COLOR["pengunjung"]
        elif is_kasir:
            label = "KASIR"
            color = COLOR["kasir"]
            stats.kasir_count += 1
        else:
            # [P5] Konsisten — semua yang bukan kasir = "pengunjung"
            label = "pengunjung"
            color = COLOR["pengunjung"]
            stats.non_kasir_count += 1
            stats.alerts.append("Pengunjung di area kasir")

        x1, y1, x2, y2 = person.xyxy
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        _put_label(annotated, f"{label} {person.conf:.2f}", (x1, y1), color)

        _draw_attribute_indicators(
            annotated, person,
            hat_det,   hat_conf,
            apron_det, apron_conf,
            shirt_det, shirt_conf,
        )

        if debug_pose:
            _draw_pose_debug(annotated, kp)

    # Alert banner jika ada pengunjung di area kasir
    if stats.non_kasir_count > 0:
        _draw_alert_banner(
            annotated,
            f"WARNING: {stats.non_kasir_count} PENGUNJUNG DI AREA KASIR"
        )

    return annotated, stats


# ~~~~~INFERENCE~~~~~

def run_inference(
    weights    : str | Path,
    video_path : str | Path,
    output_dir : str | Path = "runs/inference",
    conf       : float = CONF_THRESHOLD,
    iou_thresh : float = IOU_THRESHOLD,
    show       : bool  = False,
    use_pose   : bool  = True,
    debug_pose : bool  = False,
) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("pip install ultralytics")

    weights    = Path(weights)
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not weights.exists():
        raise FileNotFoundError(f"Weights tidak ada: {weights}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video tidak ada: {video_path}")

    model = YOLO(str(weights))
    model.to("cuda:0")
    
    pose_estimator = PoseEstimator(enabled=use_pose)
    if use_pose and pose_estimator.enabled:
        log.info("Pose estimator: AKTIF (window topi/apron dikunci ke landmark tubuh)")
    elif use_pose:
        log.info("Pose estimator: GAGAL diinisialisasi, pakai mode fallback persentase-bbox")
    else:
        log.info("Pose estimator: DIMATIKAN (--no-pose), pakai mode fallback persentase-bbox")

    cap      = cv2.VideoCapture(str(video_path))
    fps      = cap.get(cv2.CAP_PROP_FPS) or 30
    width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = output_dir / f"{video_path.stem}_result.mp4"
    writer   = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (width, height),
    )

    log.info("Video    : %s (%dx%d, %.1f fps)", video_path.name, width, height, fps)
    log.info("Weights  : %s", weights.name)
    log.info("Output   : %s", out_path)

    total_non_kasir = 0
    frame_idx       = 0
    t_start         = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = model(
            frame,
            conf=conf,
            iou=iou_thresh,
            device=0,
            verbose=False
        )[0]
        
        detections = _parse_results(results)
        annotated, stats = analyze_frame(
            frame, detections,
            pose_estimator=pose_estimator,
            debug_pose=debug_pose,
        )

        total_non_kasir += stats.non_kasir_count
        writer.write(annotated)

        if show:
            cv2.imshow("Kasir Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                log.info("User quit.")
                break

        if frame_idx % 100 == 0:
            elapsed  = time.perf_counter() - t_start
            fps_proc = frame_idx / elapsed
            log.info("Frame %d/%d | %.1f fps | alerts: %d",
                     frame_idx, n_frames, fps_proc, total_non_kasir)

    elapsed = time.perf_counter() - t_start
    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()

    log.info("[DONE] %d frame | %.1fs | %d frame dengan pengunjung di area kasir",
             frame_idx, elapsed, total_non_kasir)
    log.info("[DONE] Output → %s", out_path)
    return out_path


def _parse_results(results) -> list[Detection]:
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections.append(Detection(
            cls_id = int(box.cls[0]),
            conf   = float(box.conf[0]),
            xyxy   = (x1, y1, x2, y2),
        ))
    return detections


# ~~~~~CLI~~~~~

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Inference kasir detection HokBen")
    p.add_argument("--video",      required=True,  help="Path ke video input")
    p.add_argument("--weights",    default="runs/detect/kasir_v1/weights/best.pt")
    p.add_argument("--output",     default="runs/inference")
    p.add_argument("--conf",       type=float, default=CONF_THRESHOLD)
    p.add_argument("--show",       action="store_true", help="Tampilkan window live")
    p.add_argument("--no-pose",    action="store_true", help="Matikan pose estimator, pakai mode persentase-bbox lama")
    p.add_argument("--debug-pose", action="store_true", help="Gambar titik landmark pose di video output")
    args = p.parse_args()

    run_inference(
        weights    = args.weights,
        video_path = args.video,
        output_dir = args.output,
        conf       = args.conf,
        show       = args.show,
        use_pose   = not args.no_pose,
        debug_pose = args.debug_pose,
    )