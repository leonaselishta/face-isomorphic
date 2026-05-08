Face Isomorphism Tracker
========================
Detects 2 faces in real-time via webcam and measures how "isomorphic"
(structurally / proportionally similar) they are to each other.

Algorithm:
  1. MediaPipe Face Mesh extracts 468 facial landmarks per face.
  2. Each face's landmarks are normalised (centered + scale-normalised).
  3. Full Procrustes alignment removes rotation/reflection differences.
  4. Residual disparity → similarity score in [0, 1].
  5. Score smoothed with EMA for stable real-time display.

Install:
    pip install opencv-python mediapipe numpy scipy

Run:
    python face_isomorphic.py
    python face_isomorphic.py 1   # use camera index 1

Controls:
    Q / ESC  → quit
    S        → save screenshot