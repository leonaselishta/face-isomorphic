import cv2
import numpy as np
import time
import sys

# Constants
WIN_NAME = "Face Isomorphism Tracker"
GREEN = (50, 220, 50)
AMBER = (30, 180, 255)
RED = (60, 60, 230)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARK = (25, 25, 25)
FONT = cv2.FONT_HERSHEY_SIMPLEX

THRESH_ISO = 0.88
THRESH_SIM = 0.72

# Load detectors
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


# Feature extraction


def extract_features(face_roi_gray, face_rect):
    """
    Extract a compact feature vector from a face ROI:
      - Face aspect ratio (w/h)
      - Eye positions (normalised to face box)
      - Face box proportions
    Returns a 1-D numpy array.
    """
    x, y, w, h = face_rect
    features = []

    # 1) Aspect ratio
    features.append(w / h)

    # 2) Eye detection within face ROI
    eyes = eye_cascade.detectMultiScale(
        face_roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
    )

    if len(eyes) >= 2:
        # Sort eyes left -> right
        eyes = sorted(eyes, key=lambda e: e[0])[:2]
        for ex, ey, ew, eh in eyes:
            # Normalise eye centre to face dimensions
            features.append((ex + ew / 2) / w)
            features.append((ey + eh / 2) / h)
            features.append(ew / w)  # eye width ratio
            features.append(eh / h)  # eye height ratio
    else:
        # Pad with neutral values if eyes not found
        features.extend([0.25, 0.35, 0.15, 0.10, 0.75, 0.35, 0.15, 0.10])

    # 3) Normalised face size (relative to image, captured separately)
    features.append(w / h)  # redundant but adds weight to shape

    return np.array(features, dtype=np.float64)


def cosine_similarity(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def face_similarity(feat_a, feat_b):
    """Combine cosine similarity + feature-difference penalty."""
    cos = cosine_similarity(feat_a, feat_b)
    diff = np.abs(feat_a - feat_b).mean()
    score = cos * (1.0 - min(diff, 1.0) * 0.3)
    return float(np.clip(score, 0, 1))


def verdict(score):
    if score >= THRESH_ISO:
        return "ISOMORPHIC", GREEN
    elif score >= THRESH_SIM:
        return "SIMILAR", AMBER
    else:
        return "NOT ISOMORPHIC", RED


# Drawing helpers
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


# Main
def main():
    cam_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {cam_index}.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    smooth = 0.0
    alpha = 0.18
    t0, fc = time.time(), 0
    fps = 0.0
    shot_n = 0
    FACE_COLS = [GREEN, AMBER]

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)
        H, W = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # FPS
        fc += 1
        now = time.time()
        if now - t0 >= 1.0:
            fps = fc / (now - t0)
            fc = 0
            t0 = now

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )

        n = len(faces)

        # Draw face boxes
        for i, (x, y, w, h) in enumerate(faces[:2]):
            col = FACE_COLS[i % 2]
            cv2.rectangle(frame, (x, y), (x + w, y + h), col, 2)
            put_text(frame, f"Face {i+1}", (x + 4, y - 8), 0.7, col, 2)

            # Draw eyes inside face
            roi_gray = gray[y : y + h, x : x + w]
            eyes = eye_cascade.detectMultiScale(
                roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
            )
            for ex, ey, ew, eh in eyes[:2]:
                cv2.rectangle(
                    frame,
                    (x + ex, y + ey),
                    (x + ex + ew, y + ey + eh),
                    (200, 200, 50),
                    1,
                )

        # Header
        cv2.rectangle(frame, (0, 0), (W, 52), DARK, -1)
        put_text(frame, "Face Isomorphism Tracker -  UKZ ", (14, 36), 1.0, WHITE, 2)
        put_text(frame, f"FPS {fps:.0f}", (W - 110, 36), 0.65, (150, 150, 150), 1)

        # Bottom panel / Footer
        py = H - 90
        cv2.rectangle(frame, (0, py), (W, H), DARK, -1)

        if n >= 2:
            x0, y0, w0, h0 = faces[0]
            x1, y1, w1, h1 = faces[1]
            fa = extract_features(gray[y0 : y0 + h0, x0 : x0 + w0], faces[0])
            fb = extract_features(gray[y1 : y1 + h1, x1 : x1 + w1], faces[1])
            raw = face_similarity(fa, fb)
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

        ind_col = GREEN if n >= 2 else (AMBER if n == 1 else RED)
        put_text(frame, f"Faces: {min(n,2)}/2", (14, 76), 0.7, ind_col, 2)
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
