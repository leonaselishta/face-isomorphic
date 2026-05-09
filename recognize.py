
import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
from collections import deque

from graph_features import build_weighted_graph, laplacian_spectrum, N_SPECTRAL

mp_face_mesh   = mp.solutions.face_mesh
mp_drawing     = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles

MODEL_FILE     = "face_model.pkl"
SMOOTH_WINDOW  = 12
SPECTRAL_EVERY = 10
MLP_THRESHOLD  = 0.70   # minimum softmax confidence to show a name


def get_features(face_landmarks, cached_spec):
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


def to_pca(bundle, feat_raw):
    feat = feat_raw.copy()
    feat[-N_SPECTRAL:] *= bundle["spectral_weight"]
    feat_s = bundle["scaler"].transform(feat.reshape(1, -1))
    return bundle["pca"].transform(feat_s)[0]


def predict(bundle, feat_p):
    mode = bundle["mode"]

    if mode == "one_person":
        dist  = float(np.linalg.norm(feat_p - bundle["centroid"]))
        thresh = bundle["threshold"]
        if dist < thresh:
            conf = float(np.clip(1.0 - 0.4 * dist / thresh, 0.6, 1.0))
            return bundle["name"], conf
        return "Unknown", 0.0

    else:  # multi_person MLP
        proba   = bundle["model"].predict_proba(feat_p.reshape(1, -1))[0]
        idx     = int(np.argmax(proba))
        conf    = float(proba[idx])
        name    = bundle["encoder"].classes_[idx] if conf >= MLP_THRESHOLD \
                  else "Unknown"
        return name, conf


class Smoother:
    def __init__(self):
        self.names = deque(maxlen=SMOOTH_WINDOW)
        self.confs = deque(maxlen=SMOOTH_WINDOW)

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
        winner = self.label
        vals = [c for n, c in zip(self.names, self.confs) if n == winner]
        return float(np.mean(vals)) if vals else 0.0


def main():
    if not os.path.isfile(MODEL_FILE):
        print(f"ERROR: {MODEL_FILE} not found. Run train.py first.")
        return

    bundle = joblib.load(MODEL_FILE)
    mode   = bundle["mode"]
    print(f"Mode: {'One-person centroid' if mode == 'one_person' else 'MLP classifier'}")
    print(f"Enrolled: {bundle['people']}\n")

    cap          = cv2.VideoCapture(0)
    smoother     = Smoother()
    frame_n      = 0
    cached_spec  = None
    spec_frame_n = -999

    with mp_face_mesh.FaceMesh(
        max_num_faces=2,
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

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]

                mp_drawing.draw_landmarks(
                    image=frame, landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_draw_styles
                        .get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(
                    image=frame, landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_draw_styles
                        .get_default_face_mesh_contours_style())

                # update Laplacian cache
                if frame_n - spec_frame_n >= SPECTRAL_EVERY:
                    G            = build_weighted_graph(face_landmarks)
                    cached_spec  = laplacian_spectrum(G, k=N_SPECTRAL)
                    spec_frame_n = frame_n

                try:
                    feat_raw = get_features(face_landmarks, cached_spec)
                    feat_p   = to_pca(bundle, feat_raw)
                    name, conf = predict(bundle, feat_p)
                    smoother.update(name, conf)
                except Exception:
                    pass

                xs = [lm.x * w for lm in face_landmarks.landmark]
                ys = [lm.y * h for lm in face_landmarks.landmark]
                x1, y1 = int(min(xs)), int(min(ys))
                x2, y2 = int(max(xs)), int(max(ys))

                label = smoother.label
                conf  = smoother.confidence
                color = (0, 220, 0) if label not in ("Unknown", "...") \
                        else (0, 0, 220)

                cv2.rectangle(frame, (x1, y1 - 5), (x2, y2 + 5), color, 2)
                cv2.putText(frame, f"{label}  {conf*100:.0f}%",
                            (x1, y1 - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                if cached_spec is not None:
                    spec_str = "λ: " + "  ".join(f"{v:.3f}" for v in cached_spec[:5])
                    cv2.putText(frame, spec_str, (x1, y2 + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 200, 150), 1)

            mode_str = "Centroid" if mode == "one_person" else "MLP"
            cv2.putText(frame, f"{mode_str} | PCA+Graph features | Q=quit",
                        (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (100, 100, 100), 1)
            cv2.imshow("Face Recognition — Graph Theory", frame)
            frame_n += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
