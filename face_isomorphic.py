import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import procrustes
import time
import sys

# ── MediaPipe setup ────────────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

FACE_MESH_CFG = dict(
    static_image_mode=False,
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ── Visual constants ───────────────────────────────────────────────────────────
WIN_NAME = "Face Isomorphism Tracker"
GREEN = (50, 220, 50)
AMBER = (30, 180, 255)
RED = (60, 60, 230)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARK = (25, 25, 25)
FONT = cv2.FONT_HERSHEY_SIMPLEX

THRESH_ISO = 0.90  # >= ISOMORPHIC  🟢
THRESH_SIM = 0.78  # >= SIMILAR     🟡


# ── Maths helpers ──────────────────────────────────────────────────────────────


def lm_to_array(face_landmarks, w, h):
    return np.array(
        [[lm.x * w, lm.y * h] for lm in face_landmarks.landmark], dtype=np.float64
    )


def normalise(pts):
    pts = pts - pts.mean(axis=0)
    s = np.sqrt((pts**2).sum(axis=1).mean())
    return pts / s if s > 0 else pts


def face_similarity(pts_a, pts_b):
    try:
        _, _, d = procrustes(pts_a, pts_b)
        return float(max(0.0, 1.0 - d))
    except Exception:
        return 0.0


def verdict(score):
    if score >= THRESH_ISO:
        return "ISOMORPHIC", GREEN
    elif score >= THRESH_SIM:
        return "SIMILAR", AMBER
    else:
        return "NOT ISOMORPHIC", RED


# ── Drawing helpers ────────────────────────────────────────────────────────────


def put_text(frame, text, pos, scale=0.75, color=WHITE, thick=2):
    cv2.putText(frame, text, (pos[0] + 1, pos[1] + 1), FONT, scale, BLACK, thick + 1)
    cv2.putText(frame, text, pos, FONT, scale, color, thick)


def score_bar(frame, score, x, y, w=320, h=30):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (55, 55, 55), -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), WHITE, 1)
    fill = int(w * score)
    _, col = verdict(score)
    if fill > 0:
        cv2.rectangle(frame, (x, y), (x + fill, y + h), col, -1)
    lbl = f"{score*100:.1f}%"
    tw = cv2.getTextSize(lbl, FONT, 0.65, 2)[0][0]
    put_text(frame, lbl, (x + w // 2 - tw // 2, y + h - 6), 0.65, BLACK, 2)
    put_text(frame, lbl, (x + w // 2 - tw // 2, y + h - 6), 0.65, WHITE, 1)


def face_bbox(frame, face_lm, W, H, colour, label):
    pts = lm_to_array(face_lm, W, H).astype(int)
    x1, y1 = pts.min(axis=0) - 12
    x2, y2 = pts.max(axis=0) + 12
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W - 1, x2), min(H - 1, y2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
    put_text(frame, label, (x1 + 4, y1 - 8), 0.7, colour, 2)


# ── Main loop ──────────────────────────────────────────────────────────────────


def main():
    cam_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {cam_index}.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    smooth = 0.0
    alpha = 0.20  # EMA smoothing factor
    t0, fc = time.time(), 0
    fps = 0.0
    shot_n = 0
    FACE_COLS = [GREEN, AMBER]

    with mp_face_mesh.FaceMesh(**FACE_MESH_CFG) as mesh:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)
            H, W = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = mesh.process(rgb)

            # FPS
            fc += 1
            now = time.time()
            if now - t0 >= 1.0:
                fps = fc / (now - t0)
                fc = 0
                t0 = now

            faces = res.multi_face_landmarks or []
            n = len(faces)

            # Mesh + boxes
            for i, flm in enumerate(faces):
                mp_drawing.draw_landmarks(
                    frame,
                    flm,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style(),
                )
                face_bbox(frame, flm, W, H, FACE_COLS[i % 2], f"Face {i+1}")

            # Header
            cv2.rectangle(frame, (0, 0), (W, 52), DARK, -1)
            put_text(frame, "Face Isomorphism Tracker", (14, 36), 1.0, WHITE, 2)
            put_text(frame, f"FPS {fps:.0f}", (W - 110, 36), 0.65, (150, 150, 150), 1)

            # Bottom panel
            py = H - 90
            cv2.rectangle(frame, (0, py), (W, H), DARK, -1)

            if n == 2:
                a = normalise(lm_to_array(faces[0], W, H))
                b = normalise(lm_to_array(faces[1], W, H))
                raw = face_similarity(a, b)
                smooth = alpha * raw + (1 - alpha) * smooth

                label, col = verdict(smooth)

                bw, bx = 320, W // 2 - 160
                put_text(
                    frame, "Isomorphism Score", (bx, py + 14), 0.60, (200, 200, 200), 1
                )
                score_bar(frame, smooth, bx, py + 22, bw, 28)

                tw = cv2.getTextSize(label, FONT, 1.1, 3)[0][0]
                put_text(frame, label, (W // 2 - tw // 2, H - 10), 1.1, col, 3)

            elif n == 1:
                smooth = max(0.0, smooth - 0.015)
                put_text(
                    frame,
                    "Waiting for 2nd face ...",
                    (W // 2 - 180, H - 18),
                    0.9,
                    (140, 140, 140),
                    2,
                )
            else:
                smooth = max(0.0, smooth - 0.015)
                put_text(
                    frame,
                    "Stand in front of the camera",
                    (W // 2 - 210, H - 18),
                    0.9,
                    (100, 100, 100),
                    2,
                )

            ind_col = GREEN if n == 2 else (AMBER if n == 1 else RED)
            put_text(frame, f"Faces: {n}/2", (14, 76), 0.7, ind_col, 2)
            put_text(
                frame, "Q/ESC=quit  S=screenshot", (14, H - 96), 0.5, (100, 100, 100), 1
            )

            cv2.imshow(WIN_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("s"):
                fname = f"screenshot_{shot_n:03d}.png"
                cv2.imwrite(fname, frame)
                print(f"[INFO] Saved {fname}")
                shot_n += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
