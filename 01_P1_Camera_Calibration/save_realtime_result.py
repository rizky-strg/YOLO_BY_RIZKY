import cv2
import numpy as np
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description="Live Undistortion Viewer")
parser.add_argument("--cam", type=int, default=2, help="Index kamera (default: 2)")
args, _ = parser.parse_known_args()

CAMERA_INDEX = args.cam
CALIBRATION_FILE = Path("realtime_calibration.npz")
OUTPUT_DIR = Path("output") / "realtime"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load hasil kalibrasi
if not CALIBRATION_FILE.exists():
    raise SystemExit(f"File {CALIBRATION_FILE} tidak ditemukan! Jalankan realtimeCameraCalibration.py terlebih dahulu.")

data = np.load(CALIBRATION_FILE)
camera_matrix = data["camera_matrix"]
distortion_coefficients = data["distortion_coefficients"]

# Buka kamera dengan fallback
cap = None
for idx in [CAMERA_INDEX, 0, 1, 2, 3]:
    temp = cv2.VideoCapture(idx)
    if temp.isOpened():
        ret, _ = temp.read()
        if ret:
            cap = temp
            CAMERA_INDEX = idx
            print(f"[KAMERA] Berhasil membuka kamera index: {CAMERA_INDEX}")
            break
        temp.release()

if cap is None:
    raise RuntimeError("Kamera tidak dapat dibuka pada index mana pun.")

print(f"Kamera index {CAMERA_INDEX} aktif.")
print("Tekan SPACE untuk mengambil gambar perbandingan.")
print("Tekan Q untuk keluar.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Gagal membaca frame kamera.")
        break

    # Proses undistortion
    undistorted = cv2.undistort(
        frame,
        camera_matrix,
        distortion_coefficients
    )

    # Gabungkan original dan undistorted
    comparison = np.hstack((frame, undistorted))

    cv2.putText(
        comparison,
        "ORIGINAL",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        comparison,
        "UNDISTORTED",
        (660, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Realtime Calibration Result", comparison)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == 32:  # SPACE
        cv2.imwrite(str(OUTPUT_DIR / "realtime_original.jpg"), frame)
        cv2.imwrite(
            str(OUTPUT_DIR / "realtime_undistorted.jpg"),
            undistorted
        )
        cv2.imwrite(
            str(OUTPUT_DIR / "realtime_comparison.jpg"),
            comparison
        )

        print("JPG berhasil disimpan:")
        print(" - output_realtime/realtime_original.jpg")
        print(" - output_realtime/realtime_undistorted.jpg")
        print(" - output_realtime/realtime_comparison.jpg")

cap.release()
cv2.destroyAllWindows()