"""Guided enrollment for mesh features."""

import cv2
import mediapipe as mp
import numpy as np
import argparse
import logging
import os
import csv
from pathlib import Path

try:
    from .face_utils import (
        extract_features, face_quality, landmark_bbox, pose_matches_target,
        MeshLaplacianWorker, FEAT_DIM, SCHEMA_VER,
    )
except ImportError:
    import sys
    from pathlib import Path
    ROOT_DIR = Path(__file__).resolve().parent.parent
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    from face_isomorphic.face_utils import (
        extract_features, face_quality, landmark_bbox, pose_matches_target,
        MeshLaplacianWorker, FEAT_DIM, SCHEMA_VER,
    )
# embedding backend removed — mesh-only enrollment

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

mp_face_mesh = mp.solutions.face_mesh
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT_DIR / "face_data.csv"

# How often to request a new Laplacian computation (frames).
# The worker runs async so this just controls how fresh the cached value is.
SPECTRAL_REQUEST_EVERY = 5

POSES = [
    ("Front",       "Look straight at the camera",   0,   0),
    ("Left ~30°",   "Turn your head to the LEFT",   -30,  0),
    ("Right ~30°",  "Turn your head to the RIGHT",   30,  0),
    ("Up ~15°",     "Tilt your head UP slightly",     0, -15),
    ("Down ~15°",   "Tilt your head DOWN slightly",   0,  15),
]

JITTER_SIGMA = 0.004   # stronger jitter — better simulates live lighting/motion variation
MIN_REAL_SAMPLES_PER_PERSON = 120


# ── augmentation ──────────────────────────────────────────────────────────────
def augment_feature(feat, rng, n_copies=2):
    """
    Return n_copies augmented versions of feat by adding small Gaussian
    noise to the coordinate slice only (ratios and spectrum unchanged).
    """
    copies = []
    for _ in range(n_copies):
        aug = feat.copy()
        aug[:N_COORDS] += rng.normal(0, JITTER_SIGMA, N_COORDS).astype(np.float32)
        copies.append(aug)
    return copies


# ── UI helpers ────────────────────────────────────────────────────────────────
def draw_progress(frame, collected, target, x, y, bar_w=280):
    pct = min(collected / target, 1.0)
    cv2.rectangle(frame, (x, y), (x + bar_w, y + 16), (50, 50, 50), -1)
    cv2.rectangle(frame, (x, y), (x + int(bar_w * pct), y + 16), (0, 200, 80), -1)
    cv2.rectangle(frame, (x, y), (x + bar_w, y + 16), (100, 100, 100), 1)
    cv2.putText(frame, f"{collected}/{target}", (x + bar_w + 6, y + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)


def draw_overlay(frame, name, pose_idx, pose_name, instruction,
                 collected, per_pose, capturing, yaw, pitch, roll,
                 face_found, spec_ready, quality_reason=""):
    h, w = frame.shape[:2]

    # dark banner at top
    cv2.rectangle(frame, (0, 0), (w, 120), (20, 20, 20), -1)

    cv2.putText(frame, f"Enrolling: {name}", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(frame,
                f"Pose {pose_idx + 1}/{len(POSES)}: {pose_name}",
                (10, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
    cv2.putText(frame, instruction, (10, 77),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    draw_progress(frame, collected, per_pose, 10, 95)

    # status line
    if not face_found:
        status_text  = "No face detected — move into frame"
        status_color = (0, 0, 255)
    elif capturing and quality_reason == "good":
        status_text  = "Capturing...  (SPACE=pause  N=next  Q=quit)"
        status_color = (0, 220, 0)
    elif capturing:
        status_text = f"Waiting: {quality_reason}"
        status_color = (0, 165, 255)
    else:
        status_text  = "Ready — press SPACE to start  |  N=skip  Q=quit"
        status_color = (0, 165, 255)

    cv2.putText(frame, status_text, (10, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, status_color, 1)

    # pose angles + spectrum indicator
    spec_str = "spec:ready" if spec_ready else "spec:wait"
    cv2.putText(frame,
                f"Yaw:{yaw:+.0f}  Pitch:{pitch:+.0f}  Roll:{roll:+.0f}  {spec_str}",
                (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",     required=False)
    parser.add_argument("--per-pose", type=int, default=60)
    parser.add_argument("--augment",  type=int, default=2,
                        help="Jittered copies per real sample (0 = off)")
    args = parser.parse_args()

    if args.name:
        name = args.name.strip()
    else:
        name = input("Enter your name: ").strip()
        while not name:
            name = input("Name cannot be empty: ").strip()

    per_pose = args.per_pose
    n_aug    = max(0, args.augment)
    mesh_rows = []
    rng      = np.random.default_rng(seed=None)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        log.error("Cannot open camera. Check that no other app is using it.")
        return

    # start one shared background Laplacian worker for the whole session
    lap_worker = MeshLaplacianWorker()
    lap_worker.start()

    try:
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh:

            for pose_idx, (pose_name, instruction, target_yaw, target_pitch) in enumerate(POSES):
                collected    = 0
                capturing    = False
                frame_n      = 0
                cached_spec  = None
                last_req_frm = -SPECTRAL_REQUEST_EVERY  # trigger immediately
                quality_reason = ""

                log.info(f"\nPose {pose_idx + 1}/{len(POSES)}: {pose_name}")
                log.info(f"  → {instruction}")
                log.info(f"  Press SPACE to start, N to skip, Q to quit.\n")

                while cap.isOpened() and collected < per_pose:
                    ret, frame = cap.read()
                    if not ret:
                        log.warning("Camera read failed — retrying...")
                        continue

                    h, w = frame.shape[:2]
                    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb.flags.writeable = False
                    results = face_mesh.process(rgb)
                    rgb.flags.writeable = True
                    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                    face_found = results.multi_face_landmarks is not None
                    yaw = pitch = roll = 0.0
                    quality_ok = False
                    quality_reason = "no face"

                    if face_found:
                        lms = results.multi_face_landmarks[0]

                        # submit Laplacian request to background thread
                        if frame_n - last_req_frm >= SPECTRAL_REQUEST_EVERY:
                            lap_worker.submit(lms)
                            last_req_frm = frame_n

                        # poll for latest result (non-blocking)
                        result = lap_worker.poll()
                        if result is not None:
                            cached_spec = result

                        # extract features and optionally record
                        try:
                            feat, yaw, pitch, roll = extract_features(
                                lms, cached_spec)
                            quality_ok, quality_reason, _metrics = face_quality(
                                frame, lms, yaw, pitch, roll)

                            if not pose_matches_target(
                                    yaw, pitch, target_yaw, target_pitch):
                                quality_ok = False
                                quality_reason = "match requested pose"

                            if capturing and quality_ok:
                                mesh_rows.append([name] + feat.tolist())
                                for aug_feat in augment_feature(feat, rng, n_aug):
                                    mesh_rows.append([name] + aug_feat.tolist())
                                collected += 1

                            x1, y1, x2, y2 = landmark_bbox(lms, w, h)
                            color = (0, 220, 0) if quality_ok else (0, 165, 255)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                        except Exception as exc:
                            log.warning("Feature extraction failed frame %d: %s",
                                        frame_n, exc)

                    # draw UI
                    draw_overlay(
                        frame, name, pose_idx, pose_name, instruction,
                        collected, per_pose, capturing,
                        yaw, pitch, roll,
                        face_found, cached_spec is not None, quality_reason,
                    )

                    cv2.imshow("Enroll", frame)
                    frame_n += 1

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(" "):
                        capturing = not capturing
                        log.info("Capturing: %s", capturing)
                    elif key == ord("n"):
                        log.info(f"  Skipped pose (collected {collected})")
                        break
                    elif key == ord("q"):
                        _save_mesh(mesh_rows, name, n_aug)
                        return

                # auto-advance message
                if collected >= per_pose:
                    log.info(f"  Pose complete ({collected} samples)")

    finally:
        lap_worker.stop()
        cap.release()
        cv2.destroyAllWindows()
        _save_mesh(mesh_rows, name, n_aug)


def _save_mesh(rows, name, n_aug):
    if not rows:
        log.warning("No mesh data collected.")
        return

    n_features  = len(rows[0]) - 1
    file_exists = os.path.isfile(DATA_FILE)

    if n_features != FEAT_DIM:
        log.error(
            f"Feature dimension mismatch: got {n_features}, expected {FEAT_DIM}. "
            "Check face_utils.py."
        )
        return

    with open(DATA_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            header = [f"label_v{SCHEMA_VER}"] + [f"f{i}" for i in range(n_features)]
            writer.writerow(header)
        writer.writerows(rows)

    real_samples = len(rows) // (1 + n_aug) if n_aug > 0 else len(rows)
    log.info(
        f"\nSaved {len(rows)} rows for '{name}' "
        f"({real_samples} real + {len(rows) - real_samples} augmented) → {DATA_FILE}"
    )
    if real_samples < MIN_REAL_SAMPLES_PER_PERSON:
        log.warning(
            "Only %d real mesh samples saved. Aim for at least %d per person.",
            real_samples, MIN_REAL_SAMPLES_PER_PERSON)


    


if __name__ == "__main__":
    main()
