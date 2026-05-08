# Face Isomorphism Tracker

> Lightweight facial structure tracking using OpenCV with optional landmark detection support.

## Overview

Face Isomorphism Tracker is a real-time facial geometry analysis project built primarily with OpenCV.
It detects faces, analyzes structural relationships, and tracks facial proportions using either:

* **dlib facial landmarks** *(preferred for accuracy)*
  or
* A pure **OpenCV geometry-based fallback system** *(no dlib required)*

The project is designed to stay lightweight, fast, and easy to run without requiring deep-learning frameworks.

---

## Features

* Real-time face detection
* OpenCV-only compatible
* Optional dlib landmark enhancement
* Geometry-based facial structure analysis
* Lightweight dependencies
* Webcam-ready
* Minimal setup

---

## Tech Stack

* Python
* OpenCV
* NumPy
* dlib *(optional)*

---

## Installation

Clone the repository:

```bash
git clone git https://github.com/leonaselishta/face-isomorphic.git
cd face-isomorphism-tracker (or your file)
```

Install dependencies:

```bash
pip install opencv-python opencv-contrib-python numpy
```

Optional (recommended for better landmark precision):

```bash
pip install dlib
```

---

## Usage

Run the tracker:

```bash
python face_isomorphic.py
```

---

## How It Works

### Face Detection

Uses OpenCV Haar cascades to detect faces in real time.

### Landmark Tracking

If dlib is installed, facial landmarks are extracted for more accurate facial mapping.

### Fallback Geometry Mode

Without dlib, the system estimates facial structure using proportional geometry derived from OpenCV detections.

---

## Requirements

* Python 3.9+
* Webcam / camera device

---

## Project Structure

```text
.
├── face_isomorphic.py
├── README.md
```

---

## Example Use Cases

* Facial structure analysis
* Experimental computer vision projects
* Geometry-based face tracking
* Lightweight real-time CV systems
* Educational OpenCV projects

---

## Notes

* Detection quality depends on lighting conditions.
* dlib mode improves landmark accuracy significantly.
* The fallback system keeps the project portable and dependency-light.

---

## Author

