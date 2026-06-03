import logging
import os

import cv2
import numpy as np

log = logging.getLogger(__name__)

EMBEDDING_DATA_FILE = "face_embeddings.csv"
EMBEDDING_DIM = 512


class FaceEmbedder:
    """Thin wrapper around InsightFace embeddings.

    InsightFace is optional so the mesh pipeline still works on machines where
    the heavier embedding dependencies are not installed yet.
    """

    def __init__(self, det_size=(640, 640), providers=None):
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise RuntimeError(
                "InsightFace is not installed. Install optional dependencies "
                "with: pip install insightface onnxruntime"
            ) from exc

        if providers is None:
            providers = ["CPUExecutionProvider"]

        self.app = FaceAnalysis(name="buffalo_l", providers=providers)
        self.app.prepare(ctx_id=0, det_size=det_size)

    def extract(self, frame_bgr):
        faces = self.app.get(frame_bgr)
        out = []
        h, w = frame_bgr.shape[:2]
        for face in faces:
            emb = np.asarray(face.normed_embedding, dtype=np.float32)
            if emb.size == 0:
                continue
            x1, y1, x2, y2 = np.asarray(face.bbox, dtype=np.float32)
            x1 = float(np.clip(x1, 0, w - 1))
            y1 = float(np.clip(y1, 0, h - 1))
            x2 = float(np.clip(x2, 0, w - 1))
            y2 = float(np.clip(y2, 0, h - 1))
            score = float(getattr(face, "det_score", 0.0))
            out.append({"embedding": emb, "bbox": (x1, y1, x2, y2), "score": score})
        return out


def l2_normalize(X):
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        return X / max(float(np.linalg.norm(X)), 1e-8)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)


def cosine_similarity(a, b):
    return float(np.dot(l2_normalize(a), l2_normalize(b)))


def bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def largest_face(faces):
    if not faces:
        return None
    return max(faces, key=lambda f: bbox_area(f["bbox"]))


def embedding_data_exists(path=EMBEDDING_DATA_FILE):
    return os.path.isfile(path)
