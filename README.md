# 🧠 Fingerprint Image Enhancement API

This API allows users to upload cropped finger images (detected using YOLOv8), which are then enhanced through a series of image processing steps designed for biometric applications such as touchless fingerprint recognition. The processed output is returned in base64 format.

---

## 🚀 Features

- Automatic finger detection using YOLOv8
- Advanced image enhancement pipeline
- Ridge orientation and frequency analysis
- Gabor filtering for ridge structure enhancement
- RESTful API with base64 output for integration

---

## 🧰 Pipeline Overview

1. **YOLOv8 Detection**  
   Input images are cropped using the YOLOv8 model by detecting four fingers.

2. **Image Upload & Processing**  
   Users upload cropped finger images via API. Images are processed and enhanced.

3. **Base64 Output**  
   Enhanced images are returned as base64-encoded strings.

---

## 🧠 Key Functions

### `frequest`
- Calculates the **spatial frequency** of ridges using orientation information.
- Rotates the image, projects ridges, and determines ridge wavelengths.

### `ridge_orient`
- Computes **ridge orientation** at each pixel by analyzing gradients.
- Applies smoothing for consistent directional information.

### `ridge_freq`
- Determines **ridge frequency** in small blocks using local orientation data.

### `ridge_segment`
- Segments relevant image areas via:
  - Histogram equalization
  - Standard deviation-based thresholding
- Normalizes the image for further processing.

### `ridge_filter`
- Applies **Gabor filter** to enhance ridge-like structures.
- Uses calculated orientation and frequency maps.

### `image_enhance`
- Main enhancement function:
  - Normalization
  - Segmentation
  - Orientation & frequency analysis
  - Ridge filtering

### `processImage`
- Main function for API integration.
- Processes input and visualizes intermediate outputs.

---

## 📤 API Input/Output

| **Type**  | **Format**         |
|-----------|--------------------|
| Input     | Cropped finger image (RGB or grayscale) |
| Output    | Enhanced image (base64 string)          |

---

## 🛠️ Tech Stack

- YOLOv8 – Finger detection
- Python, OpenCV, NumPy, SciPy – Image processing
- FastAPI/Flask – API handling
- Gabor Filter – Ridge enhancement

---

## 🔍 Applications

- 🔐 Biometric authentication
- 🕵️‍♂️ Forensics & identity verification
- 🏥 Healthcare patient ID systems
- 💸 AML (Anti-Money Laundering) with biometric proof

---
## 🧪 Sample Results
Original ➡️ Enhanced
Ridge structure becomes significantly clearer, aiding in accurate feature extraction for matching or verification purposes.

---
# Sample usage flow
input_image -> YOLOv8 crop -> processImage() -> enhanced_image_base64
