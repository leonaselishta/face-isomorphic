"""
recognize.py  —  Multi-face recognition with pose normalization + pose gating.

Improvements:
  1. Pose-normalised features  — face always "faces forward" before matching
  2. Pose-gated confidence     — extreme angles show "?" instead of wrong name
  3. Distance ratio features   — scale/rotation invariant geometry
  4. Per-face smoothing        — 20-frame majority vote per tracked face

Controls:  Q = quit
"""

import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
from collections import deque

from face_utils import (extract_features, build_graph, laplacian_spectrum,
                        is_pose_extreme, N_SPECTRAL)

mp_face_mesh   = mp.solutions.face_mesh
mp_drawing     = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles

MODEL_FILE     = "face_model.pkl"
MAX_FACES      = 6
SMOOTH_WINDOW  = 20
SPECTRAL_EVERY = 10
MLP_THRESHOLD  = 0.75

# Pose limits — beyond these angles show "?" not a wrong name
YAW_LIMIT   = 40
PITCH_LIMIT = 30
ROLL_LIMIT  = 25


def to_pca(bundle, feat_raw):
    feat = feat_raw.copy()
    feat[-N_SPECTRAL:] *= bundle["spectral_weight"]
    feat_s = bundle["scaler"].transform(feat.reshape(1, -1))
    return bundle["pca"].transform(feat_s)[0]


def predict(bundle, feat_p):
    if bundle["mode"] == "one_person":
        dist  = float(np.linalg.norm(feat_p - bundle["centroid"]))
        thresh = bundle["threshold"]
        if dist < thresh:
            conf = float(np.clip(1.0 - 0.4 * dist / thresh, 0.6, 1.0))
            return bundle["name"], conf
        return "Unknown", 0.0
    else:
        proba = bundle["model"].predict_proba(feat_p.reshape(1, -1))[0]
        idx   = int(np.argmax(proba))
        conf  = float(proba[idx])
        return (bundle["encoder"].classes_[idx] if conf >= MLP_THRESHOLD
                else "Unknown"), conf


class FaceSmoother:
    def __init__(self):
        self.names       = deque(maxlen=SMOOTH_WINDOW)
        self.confs       = deque(maxlen=SMOOTH_WINDOW)
        self.cached_spec = None
        self.spec_frame  = -999
        self.center      = None

    def update(self, name, conf):
        self.names.append(name)
        self.confs.append(conf)

    @property
    def label(self):
        if not self.names:
            return "..."
        counts = {}
        for n in self.names:
            counts[n] = counts.get(n, 0) + 1
        return max(counts, key=counts.get)

    @property
    def confidence(self):
        w = self.label
        vals = [c for n, c in zip(self.names, self.confs) if n == w]
        return float(np.mean(vals)) if vals else 0.0


def face_center(lms, w, h):
    xs = [lm.x * w for lm in lms.landmark]
    ys = [lm.y * h for lm in lms.landmark]
    return float(np.mean(xs)), float(np.mean(ys))


def match_smoother(smoothers, cx, cy, max_dist=120):
    best, best_d = None, float("inf")
    for s in smoothers:
        if s.center is None:
            continue
        d = np.hypot(s.center[0] - cx, s.center[1] - cy)
        if d < best_d:
            best_d, best = d, s
    return best if best_d < max_dist else None


def main():
    if not os.path.isfile(MODEL_FILE):
        print(f"ERROR: {MODEL_FILE} not found. Run train.py first.")
        return

    bundle = joblib.load(MODEL_FILE)
    print(f"Mode: {'Centroid' if bundle['mode'] == 'one_person' else 'MLP'}")
    print(f"Enrolled: {bundle['people']}\n")

    cap      = cv2.VideoCapture(0)
    smoothers = []
    frame_n  = 0

    with mp_face_mesh.FaceMesh(
        max_num_faces=MAX_FACES,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = face_mesh.process(rgb)
            rgb.flags.writeable = True
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            active_centers = []

            if results.multi_face_landmarks:
                for face_lms in results.multi_face_landmarks:
                    cx, cy = face_center(face_lms, w, h)
                    active_centers.append((cx, cy))

                    smoother = match_smoother(smoothers, cx, cy)
                    if smoother is None:
                        smoother = FaceSmoother()
                        smoothers.append(smoother)
                    smoother.center = (cx, cy)

                    # update Laplacian cache
                    if frame_n - smoother.spec_frame >= SPECTRAL_EVERY:
                        G = build_graph(face_lms)
                        smoother.cached_spec = laplacian_spectrum(G, k=N_SPECTRAL)
                        smoother.spec_frame  = frame_n

                    # draw mesh
                    mp_drawing.draw_landmarks(
                        image=frame, landmark_list=face_lms,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_draw_styles
                            .get_default_face_mesh_tesselation_style())
                    mp_drawing.draw_landmarks(
                        image=frame, landmark_list=face_lms,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_draw_styles
                            .get_default_face_mesh_contours_style())

                    # extract + predict
                    try:
                        feat, yaw, pitch, roll = extract_features(
                            face_lms, smoother.cached_spec)
                        feat_p = to_pca(bundle, feat)

                        if is_pose_extreme(yaw, pitch, roll,
                                           YAW_LIMIT, PITCH_LIMIT, ROLL_LIMIT):
                            # pose too extreme — don't guess
                            name, conf = "?", 0.0
                        else:
                            name, conf = predict(bundle, feat_p)

                        smoother.update(name, conf)
                    except Exception:
                        pass

                    # bounding box
                    xs = [lm.x * w for lm in face_lms.landmark]
                    ys = [lm.y * h for lm in face_lms.landmark]
                    x1, y1 = int(min(xs)), int(min(ys))
                    x2, y2 = int(max(xs)), int(max(ys))

                    label = smoother.label
                    conf  = smoother.confidence
                    color = (0, 220, 0) if label not in ("Unknown", "...", "?") \
                            else ((200, 200, 0) if label == "?"
                                  else (0, 0, 220))

                    cv2.rectangle(frame, (x1, y1-5), (x2, y2+5), color, 2)
                    conf_str = f"{conf*100:.0f}%" if label != "?" else "pose?"
                    cv2.putText(frame, f"{label}  {conf_str}",
                                (x1, y1-12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)

                    # pose angles
                    try:
                        cv2.putText(frame,
                                    f"Y:{yaw:+.0f} P:{pitch:+.0f} R:{roll:+.0f}",
                                    (x1, y2+16),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                                    (140, 200, 140), 1)
                    except Exception:
                        pass

                # clean up stale smoothers
                smoothers = [s for s in smoothers if s.center is not None and
                             any(np.hypot(s.center[0]-cx, s.center[1]-cy) < 120
                                 for cx, cy in active_centers)]

                cv2.putText(frame,
                            f"Faces: {len(results.multi_face_landmarks)}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)

            cv2.putText(frame, "Pose-normalised | Q=quit",
                        (10, h-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (100, 100, 100), 1)
            cv2.imshow("Face Recognition — Graph Theory", frame)
            frame_n += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
