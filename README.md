# Automatic Number Plate Recognition (ANPR)

This project implements an end-to-end Automatic Number Plate Recognition (ANPR) system using computer vision and deep learning techniques. The system detects vehicles, tracks them across frames, recognizes license plates, and exports structured results to a CSV file.

---

## Features

* Vehicle detection using YOLOv8 (COCO pretrained)
* Multi-object tracking using SORT
* License plate detection using a custom YOLOv8 model
* Optical Character Recognition (OCR) using EasyOCR
* Frame-by-frame video processing
* Export results to CSV for analysis

---

## System Overview

The system processes a video stream frame-by-frame and applies a multi-stage pipeline combining detection, tracking, and OCR.

---

## Pipeline

1. **Vehicle Detection**

   * Model: YOLOv8 (COCO pretrained)
   * Detects vehicles (car, bus, truck, motorcycle)
   * Filters relevant classes using predefined IDs

2. **Vehicle Tracking**

   * Algorithm: SORT (Simple Online Realtime Tracking)
   * Assigns unique `car_id` across frames
   * Ensures temporal consistency

3. **License Plate Detection**

   * Model: Custom YOLOv8
   * Detects bounding boxes of license plates

4. **Plate-to-Vehicle Matching**

   * Function: `get_car()`
   * Associates each license plate with the closest tracked vehicle

5. **Image Preprocessing**

   * Cropping plate region
   * Grayscale conversion
   * Thresholding to enhance OCR performance

6. **OCR Recognition**

   * Library: EasyOCR
   * Extracts license plate text and confidence score

7. **Result Storage**

   * Stores results in a structured dictionary
   * Exports to CSV format (`test.csv`)

---

## Project Structure

```
automatic-number-plate-recognition/
│
├── main.py                  # Main pipeline
├── util.py                  # Helper functions (OCR, matching, CSV)
├── visualize.py             # Visualization script
├── sort/
│   ├── sort.py              # Tracking algorithm (SORT)
│   └── data/                # Tracking utilities
│
├── yolov8n.pt               # Vehicle detection model
├── license_plate_detector.pt# Plate detection model
├── sample.mp4               # Input video
├── test.csv                 # Output results
├── requirements.txt
└── README.md
```

---

##  Output (test.csv)

Each row in the CSV file represents a detected license plate in a specific frame:

* `frame_nmr`: Frame index
* `car_id`: Unique vehicle ID
* `car_bbox`: Bounding box of the vehicle
* `license_plate_bbox`: Bounding box of the license plate
* `license_plate_bbox_score`: Detection confidence
* `license_number`: Recognized plate text
* `license_number_score`: OCR confidence

---

## Result Analysis

### Strengths

* **Vehicle Tracking Works**

  * `car_id` remains consistent across frames
  * Same vehicle is correctly tracked over time
  * Demonstrates proper integration of SORT tracking

* **Detection Performance**

  * License plate detection confidence typically ranges from **0.5 to 0.7**
  * Indicates stable detection performance

* **End-to-End Pipeline**

  * Successfully integrates detection, tracking, and OCR
  * Produces structured output for further analysis

---

### Limitations

* **OCR Accuracy is Inconsistent**

  * Same plate may appear as:

    * `NA13NRU`
    * `MA13NRU`
    * `NA13MRU`
  * Caused by:

    * motion blur
    * low resolution
    * imperfect thresholding

* **Low OCR Confidence**

  * Many OCR scores are below **0.5**
  * Indicates uncertainty in text recognition

* **Occasional Matching Errors**

  * Some frames show `car_id = -1`
  * License plate is not matched to a vehicle
  * Likely due to bounding box mismatch

---

### Example Insight

* Plate **`AP05JEO`** appears consistently across multiple frames
* This demonstrates:

  * Stable vehicle tracking
  * Temporal consistency in OCR

However:

* Variations such as `AP05JED` indicate OCR noise

---

## Suggested Improvements

1. **Improve OCR Accuracy**

   * Use PaddleOCR or fine-tuned EasyOCR
   * Apply advanced preprocessing (denoise, sharpening)

2. **Temporal Aggregation**

   * Store OCR results per `car_id`
   * Use majority voting for final plate prediction

3. **Improve Matching Logic**

   * Replace simple overlap with IoU-based matching
   * Enhance `get_car()` function

4. **Real-time Visualization**

   * Display bounding boxes and plate text on video output

5. **Performance Optimization**

   * Run models on GPU for faster inference

---

##  Installation

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

##  Run

```bash
python main.py
```

---

##  Requirements

* ultralytics
* opencv-python
* numpy
* easyocr
* filterpy

---

##  Conclusion

This project demonstrates a complete ANPR system pipeline combining detection, tracking, and OCR.

While detection and tracking are reliable, OCR remains the primary bottleneck. With improvements in text recognition and temporal aggregation, the system can achieve production-level performance.

---

<<<<<<< HEAD
=======
## Dataset / Video

Download sample video separately.

---
>>>>>>> 9f2403905a0d30cfc71bf5f0c7cd833ed00463d5
## Author

* Vũ Trần Cát Linh
