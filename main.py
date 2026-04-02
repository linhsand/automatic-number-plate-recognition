from ultralytics import YOLO
import cv2
import numpy as np

from sort.sort import *
from util import get_car, read_license_plate, write_csv


results = {}

mot_tracker = Sort()

# load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('license_plate_detector.pt')

# load video
cap = cv2.VideoCapture('./sample.mp4')

vehicles = [2, 3, 5, 7]

# read frames
frame_nmr = -1

while True:
    frame_nmr += 1
    ret, frame = cap.read()

    # 🔥 FIX crash cuối video
    if not ret:
        break

    results[frame_nmr] = {}

    # =========================
    # 1. DETECT VEHICLES
    # =========================
    detections = coco_model(frame)[0]
    detections_ = []

    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles:
            detections_.append([x1, y1, x2, y2, score])

    # =========================
    # 2. TRACK VEHICLES
    # =========================
    track_ids = mot_tracker.update(np.asarray(detections_))

    # =========================
    # 3. DETECT LICENSE PLATES
    # =========================
    license_plates = license_plate_detector(frame)[0]

    # 🔥 FIX: loop phải nằm trong frame
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate

        # =========================
        # 4. MATCH CAR WITH PLATE
        # =========================
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(
            license_plate, track_ids
        )

        if car_id == -1:
            xcar1, ycar1, xcar2, ycar2 = x1, y1, x2, y2

        # =========================
        # 5. CROP LICENSE PLATE
        # =========================
        h, w, _ = frame.shape

        x1i = max(0, int(x1))
        y1i = max(0, int(y1))
        x2i = min(w, int(x2))
        y2i = min(h, int(y2))

        if x2i <= x1i or y2i <= y1i:
            continue

        license_plate_crop = frame[y1i:y2i, x1i:x2i, :]

        # =========================
        # 6. OCR
        # =========================
        license_plate_crop_gray = cv2.cvtColor(
            license_plate_crop, cv2.COLOR_BGR2GRAY
        )

        _, license_plate_crop_thresh = cv2.threshold(
            license_plate_crop_gray,
            64,
            255,
            cv2.THRESH_BINARY_INV,
        )

        license_plate_text, license_plate_text_score = read_license_plate(
            license_plate_crop_thresh
        )

        # =========================
        # 7. SAVE RESULT
        # =========================
        if license_plate_text is not None:
            results[frame_nmr][car_id] = {
                'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                'license_plate': {
                    'bbox': [x1, y1, x2, y2],
                    'text': license_plate_text,
                    'bbox_score': score,
                    'text_score': license_plate_text_score
                }
            }

# =========================
# 8. WRITE CSV
# =========================
write_csv(results, './test.csv')

print("Done. Results saved to test.csv")