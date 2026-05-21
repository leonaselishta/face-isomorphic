"""
enroll.py  —  Guided multi-pose enrollment for maximum accuracy.

Captures samples in 5 defined poses so the model has explicit coverage
of the angle space:
  1. Front       (look straight at camera)
  2. Left  ~30°  (turn head left)
  3. Right ~30°  (turn head right)
  4. Up    ~15°  (tilt head up)
  5. Down  ~15°  (tilt head down)

60 samples per pose = 300 total per person.
Each sample is also augmented with 2 jittered copies → 900 rows saved.

Usage:
    python enroll.py
    python enroll.py --name "Alice" --per-pose 60 --augment 2

Controls:
    SPACE  – start / pause capturing current pose
    N      – skip to next pose
    Q      – quit and save all collected data
"""

import cv2
import mediapipe as mp
import numpy as np
import argparse
import logging
import os
import csv

from face_utils import (
    extract_features, estimate_pose, build_graph,
    laplacian_spectrum, N_SPECTRAL, FEAT_DIM, SCHEMA_VER,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

mp_face_mesh   = mp.solutions.face_mesh
DATA_FILE      = "face_data.csv"
SPECTRAL_EVERY = 5

POSES = [
    ("Front",       "Look straight at the camera",          0,   0),
    ("Left ~30°",   "Turn your head to the LEFT",          -30,  0),
    ("Right ~30°",  "Turn your head to the RIGHT",          30,  0),
    ("Up ~15°",     "Tilt your head UP slightly",            0, -15),
    ("Down ~15°",   "Tilt your head DOWN slightly",          0,  15),
]

# Jitter augmentation: small Gaussian noise added to landmark coords.
# Simulates minor head movement / camera noise without extra camera time.
JITTER_SIGMA = 0.0015   # in normalised landmark units (~0.15% of face width)


def augment_feature(feat, rng, n_copies=2):
    """
    Return n_copies augmented versions of feat by adding Gaussian noise
    to the coordinate slice only (ratios and spectrum are left intact).
    """
    from face_utils import N_COORDS
    copies = []
    for _ in range(n_copies):
        aug = feat.copy()
        aug[:N_COORDS] += rng.normal(0, JITTER_SIGMA, N_COORDS).astype(np.float32)
        copies.append(aug)
    return copies


def draw_progress(frame, collected, target, x, y, w=280):
    pct = min(collected / target, 1.0)
    cv2.rectangle(frame, (x, y), (x + w, y + 16), (50, 50, 50), -1)
    cv2.rectangle(frame, (x, y), (x + int(w * pct), y + 16), (0, 200, 80), -1)
    cv2.rectangle(frame, (x, y), (x + w, y + 16), (100, 100, 100), 1)
    cv2.putText(frame, f"{collected}/{target}", (x + w + 6, y + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",     required=False)
    parser.add_argument("--per-pose", type=int, default=60)
    parser.add_argument("--augment",  type=int, default=2,
                        help="Number of jittered copies per real sample (0 = off)")
    args = parser.parse_args()

    if args.name:
        name = args.name.strip()
    else:
        name = input("Enter your name: ").strip()
        while not name:
            name = input("Name cannot be empty: ").strip()

    per_pose = args.per_pose
    n_aug    = max(0, args.augment)
    all_rows = []
    rng      = np.random.default_rng(seed=None)
    cap      = cv2.VideoCapture(0)

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
            spec_frame_n = -999

            log.info(f"\nPose {pose_idx + 1}/{len(POSES)}: {pose_name}")
            log.info(f"  → {instruction}")
            log.info(f"  Press SPACE to start, N to skip, Q to quit.\n")

            while cap.isOpened() and collected < per_pose:
                ret, frame = cap.read()
                if not ret:
                    break

                h, w = frame.shape[:2]
                rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = face_mesh.process(rgb)
                rgb.flags.writeable = True
                frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                face_found = results.multi_face_landmarks is not None
                yaw = pitch = roll = 0.0

                if face_found:
                    lms = results.multi_face_landmarks[0]

                    # recompute Laplacian spectrum periodically
                    if frame_n - spec_frame_n >= SPECTRAL_EVERY:
                        try:
                            G            = build_graph(lms)
                            cached_spec  = laplacian_spectrum(G, k=N_SPECTRAL)
                            spec_frame_n = frame_n
                        except Exception as exc:
                            log.debug("Laplacian error: %s", exc)

                    try:
                        feat, yaw, pitch, roll = extract_features(lms, cached_spec)
                        if capturing:
                            # real sample
                            all_rows.append([name] + feat.tolist())
                            # augmented copies
                            for aug_feat in augment_feature(feat, rng, n_aug):
                                all_rows.append([name] + aug_feat.tolist())
                            collected += 1
                    except Exception as exc:
                        log.warning("Feature extraction failed on frame %d: %s",
                                    frame_n, exc)

                # ── overlay ──────────────────────────────────────────────────
                cv2.rectangle(frame, (0, 0), (w, 110), (20, 20, 20), -1)

                cv2.putText(frame, f"Enrolling: {name}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                cv2.putText(frame,
                            f"Pose {pose_idx + 1}/{len(POSES)}: {pose_name}",
                            (10, 52), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (0, 200, 255), 2)
                cv2.putText(frame, instruction, (10, 76),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

                draw_progress(frame, collected, per_pose, 10, 88)

                pose_color = (0, 220, 0) if capturing else (0, 165, 255)
                status     = "Capturing..." if capturing else "SPACE=start  N=skip  Q=quit"
                cv2.putText(frame, status, (10, h - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, pose_color, 1)

                cv2.putText(frame,
                            f"Yaw:{yaw:+.0f}°  Pitch:{pitch:+.0f}°  Roll:{roll:+.0f}°",
                            (10, h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1)

                if not face_found:
                    cv2.putText(frame, "No face detected", (10, 130),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow("Enroll", frame)
                frame_n += 1

                key = cv2.waitKey(1) & 0xFF
                if key == ord(" "):
                    capturing = not capturing
                elif key == ord("n"):
                    log.info(f"  Skipped (collected {collected})")
                    break
                elif key == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    _save(all_rows, name, n_aug)
                    return

    cap.release()
    cv2.destroyAllWindows()
    _save(all_rows, name, n_aug)


def _save(rows, name, n_aug):
    if not rows:
        log.warning("No data collected.")
        return

    n_features   = len(rows[0]) - 1
    file_exists  = os.path.isfile(DATA_FILE)

    # validate feature dimension
    if n_features != FEAT_DIM:
        log.error(
            f"Feature dimension mismatch: got {n_features}, expected {FEAT_DIM}. "
            "Check face_utils.py."
        )
        return

    with open(DATA_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            # embed schema version in the header label column name
            header = [f"label_v{SCHEMA_VER}"] + [f"f{i}" for i in range(n_features)]
            writer.writerow(header)
        writer.writerows(rows)

    real_samples = len(rows) // (1 + n_aug) if n_aug > 0 else len(rows)
    log.info(
        f"\nSaved {len(rows)} rows for '{name}' "
        f"({real_samples} real + {len(rows) - real_samples} augmented) → {DATA_FILE}"
    )


if __name__ == "__main__":
    main()
