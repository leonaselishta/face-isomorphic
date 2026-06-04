"""Shared MediaPipe mesh feature extraction, pose checks, and quality gates."""

import logging
import queue
import threading

import cv2
import numpy as np
import networkx as nx
import mediapipe as mp

logger = mp.solutions.face_mesh  # silence mediapipe; use our own logger
log    = logging.getLogger(__name__)

mp_face_mesh     = mp.solutions.face_mesh
TESS_CONNECTIONS = frozenset(mp_face_mesh.FACEMESH_TESSELATION)

# ── feature dimension constants (single source of truth) ─────────────────────
N_SPECTRAL = 50    # Laplacian eigenvalues
N_RATIOS   = 50    # distance ratio features
N_COORDS   = 1434  # 478 landmarks × 3
FEAT_DIM   = N_COORDS + N_RATIOS + N_SPECTRAL   # 1534
SCHEMA_VER = 2     # increment when feature layout changes

# ── key landmark indices ──────────────────────────────────────────────────────
NOSE_TIP        = 4
CHIN            = 152
LEFT_EYE_OUTER  = 33
RIGHT_EYE_OUTER = 263
LEFT_EYE_INNER  = 133
RIGHT_EYE_INNER = 362
LEFT_MOUTH      = 61
RIGHT_MOUTH     = 291
LEFT_EYEBROW    = 70
RIGHT_EYEBROW   = 300
FOREHEAD        = 10
LEFT_CHEEK      = 234
RIGHT_CHEEK     = 454
NOSE_BASE       = 2
UPPER_LIP       = 13
LOWER_LIP       = 14

# 50 distance pairs — chosen to capture face geometry robustly
DISTANCE_PAIRS = [
    # eye measurements
    (LEFT_EYE_OUTER,  LEFT_EYE_INNER),
    (RIGHT_EYE_OUTER, RIGHT_EYE_INNER),
    (LEFT_EYE_OUTER,  RIGHT_EYE_OUTER),   # inter-ocular (used as denominator)
    (LEFT_EYE_INNER,  RIGHT_EYE_INNER),
    # nose
    (NOSE_TIP,        NOSE_BASE),
    (NOSE_TIP,        LEFT_EYE_OUTER),
    (NOSE_TIP,        RIGHT_EYE_OUTER),
    (NOSE_TIP,        LEFT_MOUTH),
    (NOSE_TIP,        RIGHT_MOUTH),
    (NOSE_TIP,        CHIN),
    # mouth
    (LEFT_MOUTH,      RIGHT_MOUTH),
    (UPPER_LIP,       LOWER_LIP),
    (LEFT_MOUTH,      CHIN),
    (RIGHT_MOUTH,     CHIN),
    # jaw / face width
    (LEFT_CHEEK,      RIGHT_CHEEK),
    (LEFT_CHEEK,      CHIN),
    (RIGHT_CHEEK,     CHIN),
    (LEFT_CHEEK,      NOSE_TIP),
    (RIGHT_CHEEK,     NOSE_TIP),
    # forehead
    (FOREHEAD,        CHIN),
    (FOREHEAD,        NOSE_TIP),
    (FOREHEAD,        LEFT_EYE_OUTER),
    (FOREHEAD,        RIGHT_EYE_OUTER),
    # eyebrows
    (LEFT_EYEBROW,    LEFT_EYE_OUTER),
    (RIGHT_EYEBROW,   RIGHT_EYE_OUTER),
    (LEFT_EYEBROW,    RIGHT_EYEBROW),
    (LEFT_EYEBROW,    NOSE_TIP),
    (RIGHT_EYEBROW,   NOSE_TIP),
    # cross measurements
    (LEFT_EYE_OUTER,  CHIN),
    (RIGHT_EYE_OUTER, CHIN),
    (LEFT_EYE_OUTER,  LEFT_MOUTH),
    (RIGHT_EYE_OUTER, RIGHT_MOUTH),
    (LEFT_EYE_OUTER,  RIGHT_MOUTH),
    (RIGHT_EYE_OUTER, LEFT_MOUTH),
    (NOSE_BASE,       LEFT_MOUTH),
    (NOSE_BASE,       RIGHT_MOUTH),
    (NOSE_BASE,       LEFT_CHEEK),
    (NOSE_BASE,       RIGHT_CHEEK),
    (FOREHEAD,        LEFT_CHEEK),
    (FOREHEAD,        RIGHT_CHEEK),
    (LEFT_CHEEK,      LEFT_MOUTH),
    (RIGHT_CHEEK,     RIGHT_MOUTH),
    (LEFT_CHEEK,      UPPER_LIP),
    (RIGHT_CHEEK,     UPPER_LIP),
    (CHIN,            UPPER_LIP),
    (CHIN,            LOWER_LIP),
    (LEFT_EYEBROW,    CHIN),
    (RIGHT_EYEBROW,   CHIN),
    (FOREHEAD,        LEFT_MOUTH),
    (FOREHEAD,        RIGHT_MOUTH),
]

assert len(DISTANCE_PAIRS) == N_RATIOS, \
    f"DISTANCE_PAIRS has {len(DISTANCE_PAIRS)} entries but N_RATIOS={N_RATIOS}"


# ── pose estimation ───────────────────────────────────────────────────────────
def estimate_pose(lms):
    """
    Estimate yaw, pitch, roll in degrees from face landmarks.
    Returns (yaw, pitch, roll).
    Positive yaw  = face turned right
    Positive pitch = face tilted up
    Positive roll  = face tilted clockwise
    """
    nose  = np.array([lms[NOSE_TIP].x,       lms[NOSE_TIP].y,       lms[NOSE_TIP].z])
    chin  = np.array([lms[CHIN].x,            lms[CHIN].y,            lms[CHIN].z])
    l_eye = np.array([lms[LEFT_EYE_OUTER].x,  lms[LEFT_EYE_OUTER].y,  lms[LEFT_EYE_OUTER].z])
    r_eye = np.array([lms[RIGHT_EYE_OUTER].x, lms[RIGHT_EYE_OUTER].y, lms[RIGHT_EYE_OUTER].z])

    vert  = chin - nose
    horiz = r_eye - l_eye

    yaw   = float(np.degrees(np.arctan2(horiz[2], horiz[0])))
    pitch = float(np.degrees(np.arctan2(-vert[2], vert[1])))
    roll  = float(np.degrees(np.arctan2(horiz[1], horiz[0])))

    return yaw, pitch, roll


def is_pose_extreme(yaw, pitch, roll, yaw_limit=40, pitch_limit=30, roll_limit=25):
    """Return True if the head pose is too extreme for reliable recognition."""
    return (abs(yaw) > yaw_limit or
            abs(pitch) > pitch_limit or
            abs(roll) > roll_limit)


def landmark_bbox(face_landmarks, frame_w, frame_h, pad=0):
    """Return a clamped pixel bounding box for a MediaPipe face landmark set."""
    xs = np.array([lm.x * frame_w for lm in face_landmarks.landmark])
    ys = np.array([lm.y * frame_h for lm in face_landmarks.landmark])
    x1 = max(0, int(xs.min()) - pad)
    y1 = max(0, int(ys.min()) - pad)
    x2 = min(frame_w - 1, int(xs.max()) + pad)
    y2 = min(frame_h - 1, int(ys.max()) + pad)
    return x1, y1, x2, y2


def face_quality(frame_bgr, face_landmarks, yaw, pitch, roll,
                 min_face_frac=0.10, min_blur=45.0,
                 min_brightness=35.0, max_brightness=225.0,
                 yaw_limit=42, pitch_limit=32, roll_limit=28):
    """
    Score whether a frame is worth saving for enrollment.

    Returns (ok, reason, metrics). Rejecting bad enrollment frames usually helps
    more than adding a larger classifier later.
    """
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = landmark_bbox(face_landmarks, w, h, pad=4)
    face_w = max(1, x2 - x1)
    face_h = max(1, y2 - y1)
    face_frac = (face_w * face_h) / float(w * h)

    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return False, "bad crop", {"face_frac": face_frac}

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())

    metrics = {
        "face_frac": face_frac,
        "blur": blur,
        "brightness": brightness,
        "yaw": yaw,
        "pitch": pitch,
        "roll": roll,
    }

    if face_frac < min_face_frac:
        return False, "move closer", metrics
    if blur < min_blur:
        return False, "hold still", metrics
    if brightness < min_brightness:
        return False, "too dark", metrics
    if brightness > max_brightness:
        return False, "too bright", metrics
    if is_pose_extreme(yaw, pitch, roll, yaw_limit, pitch_limit, roll_limit):
        return False, "pose too extreme", metrics
    return True, "good", metrics


def pose_matches_target(yaw, pitch, target_yaw, target_pitch,
                        yaw_tol=16, pitch_tol=13):
    """Return True when the current head pose is close to the requested pose."""
    return (abs(yaw - target_yaw) <= yaw_tol and
            abs(pitch - target_pitch) <= pitch_tol)


# ── pose normalization ────────────────────────────────────────────────────────
def normalize_pose(coords_3d):
    """
    Rotate the (478, 3) landmark array so the face points forward.
    Steps:
      1. Translate centroid to origin
      2. Align the inter-ocular axis to the X axis (remove roll)
      3. Align the nose-chin axis to the Y axis (remove pitch)
      4. Scale to unit inter-ocular distance
    Returns normalised (478, 3) array.
    """
    pts = coords_3d.copy()

    # 1. centre
    pts -= pts.mean(axis=0)

    # 2. remove roll — rotate so eye line is horizontal
    l_eye   = pts[LEFT_EYE_OUTER]
    r_eye   = pts[RIGHT_EYE_OUTER]
    eye_vec = r_eye - l_eye
    roll_angle = np.arctan2(eye_vec[1], eye_vec[0])
    cos_r, sin_r = np.cos(-roll_angle), np.sin(-roll_angle)
    Rz = np.array([[cos_r, -sin_r, 0],
                   [sin_r,  cos_r, 0],
                   [0,      0,     1]])
    pts = pts @ Rz.T

    # 3. remove pitch — rotate so nose-chin axis is vertical
    nose      = pts[NOSE_TIP]
    chin      = pts[CHIN]
    vert_vec  = chin - nose
    pitch_angle = np.arctan2(vert_vec[2], vert_vec[1])
    cos_p, sin_p = np.cos(-pitch_angle), np.sin(-pitch_angle)
    Rx = np.array([[1, 0,      0     ],
                   [0, cos_p, -sin_p ],
                   [0, sin_p,  cos_p ]])
    pts = pts @ Rx.T

    # 4. scale to unit inter-ocular distance
    iod = np.linalg.norm(pts[RIGHT_EYE_OUTER] - pts[LEFT_EYE_OUTER])
    if iod > 1e-6:
        pts /= iod

    return pts


# ── distance ratio features ───────────────────────────────────────────────────
def distance_ratios(pts):
    """
    Compute N_RATIOS inter-landmark distances normalised by inter-ocular
    distance.  Ratios are invariant to head scale and more robust to
    rotation than raw coordinates.
    Returns a float32 vector of length N_RATIOS (50).
    """
    iod = np.linalg.norm(pts[RIGHT_EYE_OUTER] - pts[LEFT_EYE_OUTER])
    if iod < 1e-6:
        iod = 1.0

    ratios = []
    for i, j in DISTANCE_PAIRS:
        d = np.linalg.norm(pts[i] - pts[j])
        ratios.append(d / iod)

    return np.array(ratios, dtype=np.float32)


# ── Laplacian spectrum ────────────────────────────────────────────────────────
def build_graph(face_landmarks):
    """Build a weighted NetworkX graph from MediaPipe face mesh landmarks."""
    G   = nx.Graph()
    lms = face_landmarks.landmark
    for idx, lm in enumerate(lms):
        G.add_node(idx, x=lm.x, y=lm.y, z=lm.z)
    for i, j in TESS_CONNECTIONS:
        li, lj = lms[i], lms[j]
        dist = np.sqrt((li.x - lj.x) ** 2 +
                       (li.y - lj.y) ** 2 +
                       (li.z - lj.z) ** 2)
        G.add_edge(i, j, weight=dist)
    return G


def laplacian_spectrum(G, k=N_SPECTRAL):
    """
    Return the first k eigenvalues of the normalised Laplacian.
    Runs in O(n³) — cache the result and recompute only every N frames.
    """
    nodelist = sorted(G.nodes())
    L    = nx.normalized_laplacian_matrix(G, nodelist=nodelist).toarray()
    eigs = np.linalg.eigvalsh(L)[:k]
    if len(eigs) < k:
        eigs = np.pad(eigs, (0, k - len(eigs)))
    return eigs.astype(np.float32)


class MeshLaplacianWorker(threading.Thread):
    """Compute Laplacian eigenvalues off the camera/UI thread."""

    def __init__(self):
        super().__init__(daemon=True)
        self._in_q = queue.Queue(maxsize=1)
        self._out_q = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self.latest = None

    def run(self):
        while not self._stop_event.is_set():
            try:
                face_lms = self._in_q.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                spec = laplacian_spectrum(build_graph(face_lms), k=N_SPECTRAL)
                try:
                    self._out_q.get_nowait()
                except queue.Empty:
                    pass
                self._out_q.put(spec)
            except Exception as exc:
                log.debug("MeshLaplacianWorker error: %s", exc)

    def submit(self, face_lms):
        try:
            self._in_q.get_nowait()
        except queue.Empty:
            pass
        try:
            self._in_q.put_nowait(face_lms)
        except queue.Full:
            pass

    def poll(self):
        try:
            self.latest = self._out_q.get_nowait()
        except queue.Empty:
            pass
        return self.latest

    def stop(self):
        self._stop_event.set()


# ── full feature extraction ───────────────────────────────────────────────────
def extract_features(face_landmarks, cached_spec=None):
    """
    Extract the full 1534-D feature vector from a MediaPipe face landmark set.

    Returns
    -------
    feat : np.ndarray, shape (FEAT_DIM,), dtype float32
        Layout:
          [0        : N_COORDS]              pose-normalised landmark coords
          [N_COORDS : N_COORDS + N_RATIOS]   distance ratio features
          [N_COORDS + N_RATIOS : FEAT_DIM]   Laplacian eigenvalues
    yaw, pitch, roll : float
        Head pose angles in degrees (estimated before normalisation).
    """
    lms = face_landmarks.landmark

    # raw 3-D coords
    coords = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)

    # pose estimation (on raw coords, before normalisation)
    yaw, pitch, roll = estimate_pose(lms)

    # pose normalisation
    coords_norm = normalize_pose(coords)
    coord_feat  = coords_norm.flatten()          # N_COORDS = 1434

    # distance ratio features (on normalised coords)
    ratio_feat = distance_ratios(coords_norm)    # N_RATIOS = 50

    # Laplacian spectrum (reuse cached value when available)
    if cached_spec is not None:
        spec = cached_spec
    else:
        G    = build_graph(face_landmarks)
        spec = laplacian_spectrum(G, k=N_SPECTRAL)

    feat = np.concatenate([coord_feat, ratio_feat, spec])  # FEAT_DIM = 1534
    assert feat.shape[0] == FEAT_DIM, \
        f"Feature dim mismatch: got {feat.shape[0]}, expected {FEAT_DIM}"
    return feat.astype(np.float32), yaw, pitch, roll
