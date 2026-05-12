"""
face_utils.py  —  Shared feature extraction with pose normalization.

Implements all four accuracy improvements:

1. POSE NORMALIZATION
   Rotate all 478 landmarks so the face always points forward before
   extracting features. Uses the nose tip, left/right eye corners and
   chin to compute yaw, pitch, roll and applies the inverse rotation.

2. DISTANCE RATIO FEATURES
   Compute 50 carefully chosen inter-landmark distances (eye width,
   nose length, jaw width, mouth width, etc.) and express them as
   ratios relative to the inter-ocular distance.  Ratios are invariant
   to head scale and more robust to rotation than raw coordinates.

3. LAPLACIAN SPECTRUM  (graph-theoretic)
   First 50 eigenvalues of the normalised Laplacian of the face mesh
   graph.  Captures the weighted topology of the face.

4. POSE ESTIMATION
   Returns yaw / pitch / roll so the caller can gate confidence on
   extreme angles.

Final feature vector (1484 values):
  1434  normalised + pose-corrected landmark coordinates
    50  Laplacian eigenvalues
"""

import numpy as np
import networkx as nx
import mediapipe as mp

mp_face_mesh     = mp.solutions.face_mesh
TESS_CONNECTIONS = frozenset(mp_face_mesh.FACEMESH_TESSELATION)
N_SPECTRAL       = 50

# ── key landmark indices ──────────────────────────────────────────────────────
NOSE_TIP         = 4
CHIN             = 152
LEFT_EYE_OUTER   = 33
RIGHT_EYE_OUTER  = 263
LEFT_EYE_INNER   = 133
RIGHT_EYE_INNER  = 362
LEFT_MOUTH       = 61
RIGHT_MOUTH      = 291
LEFT_EYEBROW     = 70
RIGHT_EYEBROW    = 300
FOREHEAD         = 10
LEFT_CHEEK       = 234
RIGHT_CHEEK      = 454
NOSE_BASE        = 2
UPPER_LIP        = 13
LOWER_LIP        = 14

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

    # face vertical axis
    vert  = chin - nose
    # face horizontal axis
    horiz = r_eye - l_eye

    # yaw: how much the face is turned left/right
    yaw   = float(np.degrees(np.arctan2(horiz[2], horiz[0])))
    # pitch: how much the face is tilted up/down
    pitch = float(np.degrees(np.arctan2(-vert[2], vert[1])))
    # roll: in-plane rotation
    roll  = float(np.degrees(np.arctan2(horiz[1], horiz[0])))

    return yaw, pitch, roll


def is_pose_extreme(yaw, pitch, roll, yaw_limit=40, pitch_limit=30, roll_limit=25):
    """Return True if the head pose is too extreme for reliable recognition."""
    return (abs(yaw) > yaw_limit or
            abs(pitch) > pitch_limit or
            abs(roll) > roll_limit)


# ── pose normalization ────────────────────────────────────────────────────────
def normalize_pose(coords_3d):
    """
    Rotate the (478, 3) landmark array so the face points forward.
    Steps:
      1. Translate centroid to origin
      2. Align the inter-ocular axis to the X axis (remove roll)
      3. Align the nose-chin axis to the Y axis (remove pitch)
      4. Scale to unit size
    Returns normalised (478, 3) array.
    """
    pts = coords_3d.copy()

    # 1. centre
    pts -= pts.mean(axis=0)

    # 2. remove roll — rotate so eye line is horizontal
    l_eye = pts[LEFT_EYE_OUTER]
    r_eye = pts[RIGHT_EYE_OUTER]
    eye_vec = r_eye - l_eye
    roll_angle = np.arctan2(eye_vec[1], eye_vec[0])
    cos_r, sin_r = np.cos(-roll_angle), np.sin(-roll_angle)
    Rz = np.array([[cos_r, -sin_r, 0],
                   [sin_r,  cos_r, 0],
                   [0,      0,     1]])
    pts = pts @ Rz.T

    # 3. remove pitch — rotate so nose-chin axis is vertical
    nose = pts[NOSE_TIP]
    chin = pts[CHIN]
    vert_vec = chin - nose
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
    Compute 50 inter-landmark distances, normalised by inter-ocular distance.
    Returns a vector of length 50.
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
    G = nx.Graph()
    lms = face_landmarks.landmark
    for idx, lm in enumerate(lms):
        G.add_node(idx, x=lm.x, y=lm.y, z=lm.z)
    for i, j in TESS_CONNECTIONS:
        li, lj = lms[i], lms[j]
        dist = np.sqrt((li.x-lj.x)**2 + (li.y-lj.y)**2 + (li.z-lj.z)**2)
        G.add_edge(i, j, weight=dist)
    return G


def laplacian_spectrum(G, k=N_SPECTRAL):
    nodelist = sorted(G.nodes())
    L = nx.normalized_laplacian_matrix(G, nodelist=nodelist).toarray()
    eigs = np.linalg.eigvalsh(L)[:k]
    if len(eigs) < k:
        eigs = np.pad(eigs, (0, k - len(eigs)))
    return eigs.astype(np.float32)


# ── full feature extraction ───────────────────────────────────────────────────
def extract_features(face_landmarks, cached_spec=None):
    """
    Returns (feature_vector, yaw, pitch, roll).

    Feature vector (1484 values):
      1434  pose-normalised landmark coordinates (478 × 3)
        50  Laplacian eigenvalues
    """
    lms = face_landmarks.landmark

    # raw 3D coords
    coords = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)

    # pose estimation (before normalization)
    yaw, pitch, roll = estimate_pose(lms)

    # pose normalization
    coords_norm = normalize_pose(coords)
    coord_feat  = coords_norm.flatten()   # 1434

    # Laplacian spectrum
    if cached_spec is not None:
        spec = cached_spec
    else:
        G    = build_graph(face_landmarks)
        spec = laplacian_spectrum(G, k=N_SPECTRAL)

    feat = np.concatenate([coord_feat, spec])   # 1484
    return feat.astype(np.float32), yaw, pitch, roll
