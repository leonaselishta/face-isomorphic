"""Live face recognition using the mesh classifier."""

import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
import logging
from collections import deque
from pathlib import Path

try:
    from .face_utils import (
        extract_features, landmark_bbox, is_pose_extreme, MeshLaplacianWorker,
        N_SPECTRAL, N_RATIOS, N_COORDS, SCHEMA_VER,
    )
except ImportError:
    import sys
    from pathlib import Path
    ROOT_DIR = Path(__file__).resolve().parent.parent
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    from face_isomorphic.face_utils import (
        extract_features, landmark_bbox, is_pose_extreme, MeshLaplacianWorker,
        N_SPECTRAL, N_RATIOS, N_COORDS, SCHEMA_VER,
    )
# embedding backend removed; mesh-only recognition

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

mp_face_mesh   = mp.solutions.face_mesh
mp_drawing     = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_FILE     = ROOT_DIR / "face_model.pkl"
MAX_FACES      = 6
SMOOTH_WINDOW  = 30
SPECTRAL_EVERY = 5           # refresh Laplacian more often for fresher features
MLP_THRESHOLD  = 0.55
TRACK_IOU_THRESHOLD = 0.25

YAW_LIMIT   = 40
PITCH_LIMIT = 30
ROLL_LIMIT  = 25


# ── inference pipeline ────────────────────────────────────────────────────────
def to_discriminant(bundle, feat_raw):
    """
    Apply the full preprocessing pipeline to a raw feature vector:
      weight -> scale -> PCA -> optional LDA
    Returns the projected vector ready for the classifier.
    """
    feat = feat_raw.copy()

    # apply the same feature weights used during training
    feat[N_COORDS : N_COORDS + N_RATIOS] *= bundle["ratio_weight"]
    feat[-N_SPECTRAL:]                   *= bundle["spectral_weight"]

    feat_s = bundle["scaler"].transform(feat.reshape(1, -1))
    feat_p = bundle["pca"].transform(feat_s)[0]

    if bundle["mode"] == "multi_person" and bundle.get("use_lda") and "lda" in bundle:
        feat_p = bundle["lda"].transform(feat_p.reshape(1, -1))[0]

    return feat_p


def predict(bundle, feat_d):
    """
    Classify a projected feature vector.
    Returns (name, confidence).
    """
    if bundle["mode"] == "one_person":
        dist   = float(np.linalg.norm(feat_d - bundle["centroid"]))
        thresh = bundle["threshold"]
        if dist < thresh:
            conf = float(np.clip(1.0 - 0.4 * dist / thresh, 0.6, 1.0))
            return bundle["name"], conf
        return "Unknown", 0.0

    proba = bundle["model"].predict_proba(feat_d.reshape(1, -1))[0]
    idx   = int(np.argmax(proba))
    conf  = float(proba[idx])
    conf_threshold = float(bundle.get("conf_threshold", MLP_THRESHOLD))

    if conf >= conf_threshold:
        return bundle["encoder"].classes_[idx], conf
    return "Unknown", conf





# ── per-face state ────────────────────────────────────────────────────────────
class FaceSmoother:
    """Tracks identity and confidence for one face across frames."""

    def __init__(self, use_laplacian=True):
        self.names       = deque(maxlen=SMOOTH_WINDOW)
        self.confs       = deque(maxlen=SMOOTH_WINDOW)
        self.cached_spec = None
        self.spec_frame  = -999
        self.center      = None
        self.bbox        = None
        self._lap_worker = MeshLaplacianWorker() if use_laplacian else None
        if self._lap_worker is not None:
            self._lap_worker.start()

    def update(self, name, conf):
        self.names.append(name)
        self.confs.append(conf)

    def submit_laplacian(self, face_lms):
        if self._lap_worker is None:
            return
        self._lap_worker.submit(face_lms)

    def poll_laplacian(self):
        if self._lap_worker is None:
            return self.cached_spec
        spec = self._lap_worker.poll()
        if spec is not None:
            self.cached_spec = spec
        return self.cached_spec

    def stop(self):
        if self._lap_worker is not None:
            self._lap_worker.stop()

    @property
    def label(self):
        if not self.names:
            return "..."
        # Confidence-weighted vote: Unknown and ? frames get low weight (0.1)
        # so a few bad frames can't flip the displayed label.
        scores = {}
        for name, conf in zip(self.names, self.confs):
            weight = conf if name not in ("Unknown", "?", "...") else 0.1
            scores[name] = scores.get(name, 0.0) + weight
        # Only return a real name if it has a clear lead over Unknown
        best = max(scores, key=scores.get)
        if best in ("Unknown", "?", "..."):
            return best
        unknown_score = scores.get("Unknown", 0.0) + scores.get("?", 0.0)
        if scores[best] > unknown_score * 1.5:   # real name must be 1.5× stronger
            return best
        return "Unknown"

    @property
    def confidence(self):
        w    = self.label
        vals = [c for n, c in zip(self.names, self.confs)
                if n == w and n not in ("Unknown", "?", "...")]
        if not vals:
            vals = [c for n, c in zip(self.names, self.confs) if n == w]
        return float(np.mean(vals)) if vals else 0.0


# ── face tracking helpers ─────────────────────────────────────────────────────
def face_center(lms, w, h):
    xs = [lm.x * w for lm in lms.landmark]
    ys = [lm.y * h for lm in lms.landmark]
    return float(np.mean(xs)), float(np.mean(ys))


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def match_smoother(smoothers, cx, cy, frame_w, frame_h, rel_thresh=0.15):
    """
    Match a detected face centre to an existing smoother.
    Uses a relative distance threshold (fraction of frame diagonal)
    so it works correctly at any resolution.
    """
    diag     = np.hypot(frame_w, frame_h)
    max_dist = rel_thresh * diag
    best, best_d = None, float("inf")
    for s in smoothers:
        if s.center is None:
            continue
        d = np.hypot(s.center[0] - cx, s.center[1] - cy)
        if d < best_d:
            best_d, best = d, s
    return best if best_d < max_dist else None


def match_smoother_bbox(smoothers, bbox, cx, cy, frame_w, frame_h):
    best, best_score = None, 0.0
    for s in smoothers:
        if s.bbox is None:
            continue
        score = bbox_iou(s.bbox, bbox)
        if score > best_score:
            best_score, best = score, s
    if best is not None and best_score >= TRACK_IOU_THRESHOLD:
        return best
    return match_smoother(smoothers, cx, cy, frame_w, frame_h)



# ── main ──────────────────────────────────────────────────────────────────────
def main():
    if not os.path.isfile(MODEL_FILE):
        log.error(f"{MODEL_FILE} not found. Run train.py first.")
        return

    bundle = joblib.load(MODEL_FILE)

    # embedding backend removed — proceed with mesh recognition

    # schema version check
    model_ver = bundle.get("schema_ver", 1)
    if model_ver != SCHEMA_VER:
        log.warning(
            f"Model was trained with schema v{model_ver}, "
            f"current code is v{SCHEMA_VER}. Re-enroll and retrain."
        )

    if bundle["mode"] == "one_person":
        mode_str = "Centroid"
    elif bundle.get("use_lda", "lda" in bundle):
        mode_str = "PCA + LDA + MLP"
    else:
        mode_str = "PCA + MLP"
    log.info(f"Mode: {mode_str}")
    log.info(f"Enrolled: {bundle['people']}\n")

    cap       = cv2.VideoCapture(0)
    smoothers = []
    frame_n   = 0

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
            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = face_mesh.process(rgb)
            rgb.flags.writeable = True
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            active_centers = []

            if results.multi_face_landmarks:
                for face_lms in results.multi_face_landmarks:
                    cx, cy = face_center(face_lms, w, h)
                    bbox = landmark_bbox(face_lms, w, h)
                    active_centers.append((cx, cy))

                    smoother = match_smoother_bbox(smoothers, bbox, cx, cy, w, h)
                    if smoother is None:
                        smoother = FaceSmoother()
                        smoothers.append(smoother)
                    smoother.center = (cx, cy)
                    smoother.bbox = bbox

                    # submit Laplacian computation to background thread
                    if frame_n - smoother.spec_frame >= SPECTRAL_EVERY:
                        smoother.submit_laplacian(face_lms)
                        smoother.spec_frame = frame_n

                    # poll for latest spectrum (non-blocking)
                    smoother.poll_laplacian()

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

                    # extract features + predict
                    yaw = pitch = roll = 0.0
                    pose_extreme = False
                    try:
                        feat, yaw, pitch, roll = extract_features(
                            face_lms, smoother.cached_spec)

                        pose_extreme = is_pose_extreme(
                            yaw, pitch, roll, YAW_LIMIT, PITCH_LIMIT, ROLL_LIMIT)

                        if not pose_extreme:
                            feat_d     = to_discriminant(bundle, feat)
                            name, conf = predict(bundle, feat_d)
                            smoother.update(name, conf)
                    except Exception as exc:
                        log.debug("Feature extraction error: %s", exc)

                    x1, y1, x2, y2 = landmark_bbox(face_lms, w, h)
                    smoother.bbox = (x1, y1, x2, y2)

                    if pose_extreme:
                        # show yellow "?" without touching the smoother history
                        cv2.rectangle(frame, (x1, y1-5), (x2, y2+5), (200, 200, 0), 2)
                        cv2.putText(frame, "?  pose",
                                    (x1, y1-12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (200, 200, 0), 2)
                    else:
                        label    = smoother.label
                        conf_val = smoother.confidence
                        if label not in ("Unknown", "...", "?"):
                            color = (0, 220, 0)
                        elif label == "?":
                            color = (200, 200, 0)
                        else:
                            color = (0, 0, 220)

                        cv2.rectangle(frame, (x1, y1-5), (x2, y2+5), color, 2)
                        conf_str = f"{conf_val*100:.0f}%"
                        cv2.putText(frame, f"{label}  {conf_str}",
                                    (x1, y1-12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)

                    cv2.putText(frame,
                                f"Y:{yaw:+.0f} P:{pitch:+.0f} R:{roll:+.0f}",
                                (x1, y2+16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                                (140, 200, 140), 1)

                # remove smoothers whose face has left the frame
                # use relative distance so it works at any resolution
                diag     = np.hypot(w, h)
                max_dist = 0.15 * diag
                smoothers = [
                    s for s in smoothers
                    if s.center is not None and any(
                        np.hypot(s.center[0] - cx, s.center[1] - cy) < max_dist
                        for cx, cy in active_centers
                    )
                ]

                cv2.putText(frame,
                            f"Faces: {len(results.multi_face_landmarks)}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)

            if bundle["mode"] == "multi_person":
                mode_label = "PCA+LDA+MLP" if bundle.get("use_lda", "lda" in bundle) else "PCA+MLP"
            else:
                mode_label = "Centroid"
            cv2.putText(frame, f"Pose-normalised | {mode_label} | Q=quit",
                        (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (100, 100, 100), 1)
            cv2.imshow("Face Recognition — Graph Theory", frame)
            frame_n += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # clean up background threads
    for s in smoothers:
        s.stop()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
