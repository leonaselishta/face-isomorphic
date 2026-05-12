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

Usage:
    python enroll.py
    python enroll.py --name "Alice" --per-pose 60

Controls:
    SPACE  – start / pause capturing current pose
    N      – skip to next pose
    Q      – quit and save all collected data
"""

import cv2
import mediapipe as mp
import numpy as np
import argparse
import os
import csv

from face_utils import extract_features, estimate_pose, build_graph, \
                      laplacian_spectrum, N_SPECTRAL

mp_face_mesh = mp.solutions.face_mesh
DATA_FILE    = "face_data.csv"
SPECTRAL_EVERY = 5

POSES = [
    ("Front",       "Look straight at the camera",          0,   0),
    ("Left ~30°",   "Turn your head to the LEFT",          -30,  0),
    ("Right ~30°",  "Turn your head to the RIGHT",          30,  0),
    ("Up ~15°",     "Tilt your head UP slightly",            0, -15),
    ("Down ~15°",   "Tilt your head DOWN slightly",          0,  15),
]


def draw_progress(frame, collected, target, x, y, w=280):
    pct = min(collected / target, 1.0)
    cv2.rectangle(frame, (x, y), (x+w, y+16), (50, 50, 50), -1)
    cv2.rectangle(frame, (x, y), (x+int(w*pct), y+16), (0, 200, 80), -1)
    cv2.rectangle(frame, (x, y), (x+w, y+16), (100, 100, 100), 1)
    cv2.putText(frame, f"{collected}/{target}", (x+w+6, y+13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",      required=False)
    parser.add_argument("--per-pose",  type=int, default=60)
    args = parser.parse_args()

    if args.name:
        name = args.name.strip()
    else:
        name = input("Enter your name: ").strip()
        while not name:
            name = input("Name cannot be empty: ").strip()

    per_pose  = args.per_pose
    all_rows  = []
    cap       = cv2.VideoCapture(0)

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

            print(f"\nPose {pose_idx+1}/{len(POSES)}: {pose_name}")
            print(f"  → {instruction}")
            print(f"  Press SPACE to start, N to skip, Q to quit.\n")

            while cap.isOpened() and collected < per_pose:
                ret, frame = cap.read()
                if not ret:
                    break

                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = face_mesh.process(rgb)
                rgb.flags.writeable = True
                frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                face_found = results.multi_face_landmarks is not None
                yaw = pitch = roll = 0.0

                if face_found:
                    lms = results.multi_face_landmarks[0]

                    if frame_n - spec_frame_n >= SPECTRAL_EVERY:
                        G           = build_graph(lms)
                        cached_spec = laplacian_spectrum(G, k=N_SPECTRAL)
                        spec_frame_n = frame_n

                    try:
                        feat, yaw, pitch, roll = extract_features(
                            lms, cached_spec)
                        if capturing:
                            all_rows.append([name] + feat.tolist())
                            collected += 1
                    except Exception:
                        pass

                # ── overlay ──────────────────────────────────────────────────
                # dark banner at top
                cv2.rectangle(frame, (0, 0), (w, 110), (20, 20, 20), -1)

                cv2.putText(frame, f"Enrolling: {name}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                cv2.putText(frame,
                            f"Pose {pose_idx+1}/{len(POSES)}: {pose_name}",
                            (10, 52), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (0, 200, 255), 2)
                cv2.putText(frame, instruction, (10, 76),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

                draw_progress(frame, collected, per_pose, 10, 88)

                # pose indicator
                pose_color = (0, 220, 0) if capturing else (0, 165, 255)
                status = "Capturing..." if capturing else "SPACE=start  N=skip  Q=quit"
                cv2.putText(frame, status, (10, h - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, pose_color, 1)

                # live pose angles
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
                    print(f"  Skipped (collected {collected})")
                    break
                elif key == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    _save(all_rows, name)
                    return

    cap.release()
    cv2.destroyAllWindows()
    _save(all_rows, name)


def _save(rows, name):
    if not rows:
        print("No data collected.")
        return

    file_exists = os.path.isfile(DATA_FILE)
    with open(DATA_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            header = ["label"] + [f"f{i}" for i in range(len(rows[0]) - 1)]
            writer.writerow(header)
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} samples for '{name}' → {DATA_FILE}")


if __name__ == "__main__":
    main()
