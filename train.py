"""
train.py  —  Train face recognition on pose-normalised + ratio + spectral features.

Pipeline
--------
  1. Load features (1534-D: pose-normalised coords + distance ratios + Laplacian)
  2. Upweight Laplacian features ×3, ratio features ×2
  3. StandardScaler
  4. PCA  → reduce to 95 % variance (capped at N_PCA_MAX)
  5. LDA  → reduce to (n_classes - 1) dimensions  [multi-person only]
  6. 1 person  → nearest-centroid with adaptive threshold
     2+ people → MLP neural network on PCA+LDA space

Usage
-----
    python train.py
    python train.py --data face_data.csv --model face_model.pkl
"""

import argparse
import logging
import numpy as np
import csv
import joblib
import os

from sklearn.preprocessing    import StandardScaler, LabelEncoder
from sklearn.decomposition    import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neural_network   import MLPClassifier
from sklearn.model_selection  import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics          import classification_report
from sklearn.pipeline         import Pipeline

from face_utils import N_SPECTRAL, N_RATIOS, N_COORDS, FEAT_DIM, SCHEMA_VER

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DATA_FILE        = "face_data.csv"
MODEL_FILE       = "face_model.pkl"
N_PCA_MAX        = 200          # hard cap on PCA components
PCA_VARIANCE     = 0.97         # keep enough components for 97 % variance
SPECTRAL_WEIGHT  = 3.0          # Laplacian eigenvalues upweighted ×3
RATIO_WEIGHT     = 2.0          # distance ratios upweighted ×2
THRESHOLD_PCTILE = 95
THRESHOLD_FACTOR = 1.2


# ── feature weighting ─────────────────────────────────────────────────────────
def apply_weights(X):
    """
    Upweight the distance-ratio and Laplacian-spectrum slices.
    Slice layout must match face_utils.FEAT_DIM = 1534:
      [0        : N_COORDS]              coords  (no extra weight)
      [N_COORDS : N_COORDS + N_RATIOS]   ratios  × RATIO_WEIGHT
      [-N_SPECTRAL:]                     spectrum × SPECTRAL_WEIGHT
    """
    X = X.copy()
    ratio_start = N_COORDS
    ratio_end   = N_COORDS + N_RATIOS
    X[:, ratio_start:ratio_end] *= RATIO_WEIGHT
    X[:, -N_SPECTRAL:]          *= SPECTRAL_WEIGHT
    return X


# ── one-person mode ───────────────────────────────────────────────────────────
def train_one_person(X_pca, name):
    centroid = X_pca.mean(axis=0)
    dists    = np.linalg.norm(X_pca - centroid, axis=1)
    thresh   = np.percentile(dists, THRESHOLD_PCTILE) * THRESHOLD_FACTOR
    log.info(f"  {name}: {len(X_pca)} samples | "
             f"median dist: {np.median(dists):.3f} | threshold: {thresh:.3f}")
    return {"mode": "one_person", "centroid": centroid,
            "threshold": thresh, "name": name}


# ── multi-person mode ─────────────────────────────────────────────────────────
def train_multi(X_pca, labels, le, n_classes):
    """
    Apply LDA on top of PCA, then train an MLP.
    LDA maximises between-class scatter relative to within-class scatter —
    ideal for distinguishing multiple faces simultaneously.
    """
    y_enc = le.fit_transform(labels)

    # LDA: reduces to at most (n_classes - 1) dimensions
    n_lda = min(n_classes - 1, X_pca.shape[1])
    lda   = LDA(n_components=n_lda, solver="svd", store_covariance=True)
    X_lda = lda.fit_transform(X_pca, y_enc)
    log.info(f"LDA: {X_pca.shape[1]}D → {X_lda.shape[1]}D  "
             f"(maximising between-class separation for {n_classes} people)")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_lda, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

    # MLP — wider first layer to handle the richer feature space
    model = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-3,           # L2 regularisation
        max_iter=2000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        verbose=False,
    )
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    log.info("\n── Evaluation ──────────────────────────────────────────────")
    log.info("\n" + classification_report(y_te, y_pred, target_names=le.classes_))

    # cross-validation on full LDA space for a more robust accuracy estimate
    if len(X_lda) >= 10:
        cv_scores = cross_val_score(
            MLPClassifier(
                hidden_layer_sizes=(512, 256, 128, 64),
                activation="relu", solver="adam", alpha=1e-3,
                max_iter=2000, random_state=42,
                early_stopping=True, validation_fraction=0.15,
                n_iter_no_change=30,
            ),
            X_lda, y_enc,
            cv=StratifiedKFold(n_splits=min(5, len(np.unique(y_enc))),
                               shuffle=True, random_state=42),
            scoring="accuracy",
        )
        log.info(f"Cross-val accuracy: {cv_scores.mean()*100:.1f}% "
                 f"± {cv_scores.std()*100:.1f}%")

    return {"mode": "multi_person", "model": model, "encoder": le, "lda": lda}


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  default=DATA_FILE)
    parser.add_argument("--model", default=MODEL_FILE)
    args = parser.parse_args()

    if not os.path.isfile(args.data):
        log.error(f"{args.data} not found. Run enroll.py first.")
        return

    # ── load data ─────────────────────────────────────────────────────────────
    with open(args.data, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows   = list(reader)

    if not rows:
        log.error("Data file is empty.")
        return

    labels        = np.array([r[0] for r in rows])
    X             = np.array([r[1:] for r in rows], dtype=np.float32)
    unique_people = list(dict.fromkeys(labels.tolist()))
    n_classes     = len(unique_people)

    # schema version check
    stored_ver = int(header[0].split("_v")[-1]) if "_v" in header[0] else 1
    if stored_ver != SCHEMA_VER:
        log.warning(
            f"Data schema version mismatch: file has v{stored_ver}, "
            f"code expects v{SCHEMA_VER}. Re-enroll for best results."
        )

    if X.shape[1] != FEAT_DIM:
        log.error(
            f"Feature dimension mismatch: CSV has {X.shape[1]} features, "
            f"code expects {FEAT_DIM}. Re-enroll with the current face_utils.py."
        )
        return

    log.info(f"Loaded {len(X)} samples, {n_classes} person(s): {unique_people}")
    log.info(f"Features: {X.shape[1]}D  "
             f"(coords={N_COORDS}, ratios={N_RATIOS}, spectral={N_SPECTRAL})\n")

    # ── preprocessing ─────────────────────────────────────────────────────────
    X_w      = apply_weights(X)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_w)

    # PCA: keep 97 % variance, capped at N_PCA_MAX
    n_comp_max = min(N_PCA_MAX, X_scaled.shape[0] - 1, X_scaled.shape[1])
    pca_full   = PCA(n_components=n_comp_max, random_state=42)
    pca_full.fit(X_scaled)
    cumvar     = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp     = int(np.searchsorted(cumvar, PCA_VARIANCE) + 1)
    n_comp     = max(n_comp, n_classes * 4)   # keep at least 4× n_classes dims
    n_comp     = min(n_comp, n_comp_max)

    pca    = PCA(n_components=n_comp, random_state=42)
    X_pca  = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_.sum() * 100
    log.info(f"PCA: {X_scaled.shape[1]}D → {n_comp}D  ({explained:.1f}% variance)\n")

    # ── train ─────────────────────────────────────────────────────────────────
    if n_classes == 1:
        bundle = train_one_person(X_pca, unique_people[0])
    else:
        le     = LabelEncoder()
        bundle = train_multi(X_pca, labels, le, n_classes)

    bundle.update({
        "scaler":          scaler,
        "pca":             pca,
        "spectral_weight": SPECTRAL_WEIGHT,
        "ratio_weight":    RATIO_WEIGHT,
        "people":          unique_people,
        "schema_ver":      SCHEMA_VER,
        "feat_dim":        FEAT_DIM,
    })

    joblib.dump(bundle, args.model)
    log.info(f"\nModel saved → {args.model}")


if __name__ == "__main__":
    main()
