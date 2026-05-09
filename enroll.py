
import cv2
import mediapipe as mp
import numpy as np
import argparse
import os
import csv

from graph_features import build_weighted_graph, laplacian_spectrum, N_SPECTRAL

mp_face_mesh = mp.solutions.face_mesh
DATA_FILE    = "face_data.csv"
SPECTRAL_EVERY = 5   # recompute Laplacian every N frames during enroll


def extract_features(face_landmarks, cached_spec):
    """1434 normalised coords + 50 Laplacian eigenvalues = 1484 features."""
    coords = []
    for lm in face_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    feat = np.array(coords, dtype=np.float32).reshape(-1, 3)
    feat -= feat.mean(axis=0)
    s = np.linalg.norm(feat)
    if s > 0:
        feat /= s
    feat = feat.flatten()
    if cached_spec is not None:
        feat = np.concatenate([feat, cached_spec])
    return feat


def draw_progress_bar(frame, collected, target, x, y, bar_w=300, bar_h=18):
    pct = collected / target
    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (60, 60, 60), -1)
    cv2.rectangle(frame, (x, y), (x + int(bar_w * pct), y + bar_h),
                  (0, 200, 80), -1)
    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (120, 120, 120), 1)
    cv2.putText(frame, f"{collected}/{target}", (x + bar_w + 8, y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",    required=False)
    parser.add_argument("--samples", type=int, default=300)
    args = parser.parse_args()

    if args.name:
        name = args.name.strip()
    else:
        name = input("Enter your name: ").strip()
        while not name:
            name = input("Name cannot be empty. Enter your name: ").strip()

    target    = args.samples
    collected = 0
    capturing = False
    rows      = []
    frame_n   = 0
    cached_spec  = None
    spec_frame_n = -999

    cap = cv2.VideoCapture(0)

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        print(f"\nEnrolling: {name}  (target: {target} samples)")
        print("Tip: slowly move your head in different directions while capturing.")
        print("Press SPACE to start, Q to quit.\n")

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

            face_found = results.multi_face_landmarks is not None

            if face_found:
                lms = results.multi_face_landmarks[0]

                # update Laplacian cache
                if frame_n - spec_frame_n >= SPECTRAL_EVERY:
                    G           = build_weighted_graph(lms)
                    cached_spec = laplacian_spectrum(G, k=N_SPECTRAL)
                    spec_frame_n = frame_n

                if capturing and collected < target:
                    try:
                        feat = extract_features(lms, cached_spec)
                        rows.append([name] + feat.tolist())
                        collected += 1
                    except Exception:
                        pass

            # ── overlay ──────────────────────────────────────────────────────
            status = "Capturing..." if capturing else "Press SPACE to start"
            color  = (0, 200, 0) if capturing else (0, 165, 255)

            cv2.putText(frame, f"Enrolling: {name}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame, status, (10, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            draw_progress_bar(frame, collected, target, 10, 85)

            hints = [
                "Move head slowly L/R/U/D for best accuracy",
                "Keep face well lit",
                "SPACE = start/pause   Q = save & quit",
            ]
            for i, hint in enumerate(hints):
                cv2.putText(frame, hint, (10, h - 55 + i * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1)

            if not face_found:
                cv2.putText(frame, "No face detected", (10, 115),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Enroll", frame)
            frame_n += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                capturing = not capturing
            elif key == ord("q") or collected >= target:
                break

    cap.release()
    cv2.destroyAllWindows()

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
