"""
emotion_detector.py
--------------------
Real-time emotion detection from your webcam.
Uses DeepFace.

Install dependencies first:
    pip install deepface opencv-python tf-keras

Run:
    python emotion_detector.py
"""

import cv2
import time
from deepface import DeepFace

# ── Config ────────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0  # 0 = default webcam; change if you have multiple
ANALYZE_EVERY = 0.4  # seconds between emotion analyses (reduce for faster CPU)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Emotion → color (BGR)
EMOTION_COLORS = {
    "happy": (0, 220, 100),
    "sad": (200, 80, 40),
    "angry": (30, 30, 220),
    "fear": (180, 50, 180),
    "surprise": (0, 200, 220),
    "disgust": (40, 160, 40),
    "neutral": (180, 180, 180),
}
DEFAULT_COLOR = (200, 200, 200)

# Emoji overlay per emotion
EMOTION_EMOJI = {
    "happy": "😊",
    "sad": "😢",
    "angry": "😠",
    "fear": "😨",
    "surprise": "😲",
    "disgust": "🤢",
    "neutral": "😐",
}

# ── Helpers ───────────────────────────────────────────────────────────────────


def draw_emotion_bar(frame, emotions: dict, x: int, y: int, bar_w: int = 160):
    """Draw a small bar chart of all emotion scores."""
    sorted_emotions = sorted(emotions.items(), key=lambda e: e[1], reverse=True)
    bar_h = 14
    gap = 4
    for i, (emo, score) in enumerate(sorted_emotions):
        color = EMOTION_COLORS.get(emo, DEFAULT_COLOR)
        fill_w = int(bar_w * score / 100)
        top = y + i * (bar_h + gap)
        # Background track
        cv2.rectangle(frame, (x, top), (x + bar_w, top + bar_h), (50, 50, 50), -1)
        # Filled portion
        cv2.rectangle(frame, (x, top), (x + fill_w, top + bar_h), color, -1)
        # Label
        label = f"{emo[:7]:<7} {score:4.1f}%"
        cv2.putText(
            frame,
            label,
            (x + bar_w + 6, top + bar_h - 2),
            FONT,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )


def draw_rounded_rect(frame, pt1, pt2, color, thickness=2, r=12):
    """Draw a rectangle with rounded corners."""
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.line(frame, (x1 + r, y1), (x2 - r, y1), color, thickness)
    cv2.line(frame, (x1 + r, y2), (x2 - r, y2), color, thickness)
    cv2.line(frame, (x1, y1 + r), (x1, y2 - r), color, thickness)
    cv2.line(frame, (x2, y1 + r), (x2, y2 - r), color, thickness)
    cv2.ellipse(frame, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    cv2.ellipse(frame, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    cv2.ellipse(frame, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    cv2.ellipse(frame, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


def overlay_text(
    frame, text, pos, scale=0.6, color=(255, 255, 255), thickness=1, shadow=True
):
    x, y = pos
    if shadow:
        cv2.putText(
            frame,
            text,
            (x + 1, y + 1),
            FONT,
            scale,
            (0, 0, 0),
            thickness + 1,
            cv2.LINE_AA,
        )
    cv2.putText(frame, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("❌  Could not open camera. Check CAMERA_INDEX in the script.")
        return

    print("🎥  Camera opened. Press  Q  to quit.")
    print("    Analysing emotions every", ANALYZE_EVERY, "s …\n")

    last_analysis = 0.0
    last_result = None  # cached DeepFace result
    fps_time = time.time()
    fps = 0.0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Frame grab failed.")
            break

        now = time.time()
        frame_count += 1

        # ── FPS counter ──────────────────────────────────────────────────────
        elapsed = now - fps_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            fps_time = now
            frame_count = 0

        # ── Run DeepFace analysis at throttled interval ───────────────────────
        if now - last_analysis >= ANALYZE_EVERY:
            last_analysis = now
            try:
                results = DeepFace.analyze(
                    frame,
                    actions=["emotion"],
                    enforce_detection=False,  # don't raise if no face found
                    silent=True,
                )
                last_result = results[0] if isinstance(results, list) else results
            except Exception as e:
                last_result = None

        # ── Draw overlay ──────────────────────────────────────────────────────
        h, w = frame.shape[:2]

        if last_result:
            dominant = last_result.get("dominant_emotion", "unknown")
            emotions = last_result.get("emotion", {})
            region = last_result.get("region", {})
            color = EMOTION_COLORS.get(dominant, DEFAULT_COLOR)

            # Face bounding box
            fx = region.get("x", 0)
            fy = region.get("y", 0)
            fw = region.get("w", 0)
            fh = region.get("h", 0)
            if fw > 0 and fh > 0:
                draw_rounded_rect(frame, (fx, fy), (fx + fw, fy + fh), color, 2)

            # Dominant emotion badge above the face box
            badge_text = f"{EMOTION_EMOJI.get(dominant, '')} {dominant.upper()}"
            (tw, th), _ = cv2.getTextSize(badge_text, FONT, 0.75, 2)
            bx = max(fx, 0)
            by = max(fy - 36, 0)
            cv2.rectangle(
                frame, (bx - 4, by - 4), (bx + tw + 8, by + th + 8), color, -1
            )
            cv2.putText(
                frame,
                badge_text,
                (bx + 2, by + th + 2),
                FONT,
                0.75,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

            # Emotion bar chart — right side panel
            panel_x = w - 230
            panel_y = 20
            # Semi-transparent dark background for the panel
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (panel_x - 10, panel_y - 10),
                (w - 4, panel_y + 8 * 18 + 20),
                (20, 20, 20),
                -1,
            )
            cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
            overlay_text(
                frame,
                "EMOTIONS",
                (panel_x, panel_y + 12),
                scale=0.5,
                color=(200, 200, 200),
            )
            draw_emotion_bar(frame, emotions, panel_x, panel_y + 22)

        else:
            # No face detected
            overlay_text(
                frame,
                "No face detected",
                (20, h - 20),
                scale=0.55,
                color=(100, 100, 100),
            )

        # FPS badge (top-left)
        overlay_text(
            frame, f"FPS {fps:.1f}", (10, 22), scale=0.5, color=(160, 160, 160)
        )

        # Quit hint
        overlay_text(
            frame,
            "Q  quit",
            (10, h - 10),
            scale=0.42,
            color=(120, 120, 120),
            shadow=False,
        )

        cv2.imshow("Emotion Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("👋  Closed.")


if __name__ == "__main__":
    main()
