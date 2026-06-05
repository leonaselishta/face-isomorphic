"""Train the mesh classifier from enrolled mesh features."""

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
from sklearn.svm              import SVC
from sklearn.model_selection  import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics          import classification_report
from sklearn.pipeline         import Pipeline
from sklearn.calibration      import CalibratedClassifierCV

from face_utils import N_SPECTRAL, N_RATIOS, N_COORDS, FEAT_DIM, SCHEMA_VER
# embedding backend removed; mesh-only training

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DATA_FILE        = "face_data.csv"
MODEL_FILE       = "face_model.pkl"
N_PCA_MAX        = 200
PCA_VARIANCE     = 0.97
PCA_MIN_DIMS     = 80           # raised from 50 — more dims = more signal for SVM
SPECTRAL_WEIGHT  = 3.0
RATIO_WEIGHT     = 2.0
THRESHOLD_PCTILE = 95
THRESHOLD_FACTOR = 1.2
CLEAN_PCA_COMPONENTS = 50


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


def clean_mesh_rows(rows, percentile):
    labels = [r[0] for r in rows]
    X = np.array([r[1:] for r in rows], dtype=np.float32)
    X_w = apply_weights(X)
    X_scaled = StandardScaler().fit_transform(X_w)
    n_comp = min(CLEAN_PCA_COMPONENTS, X_scaled.shape[0] - 1, X_scaled.shape[1])
    X_pca = PCA(n_components=n_comp, random_state=42).fit_transform(X_scaled)

    keep = np.ones(len(rows), dtype=bool)
    for person in dict.fromkeys(labels):
        idx = np.array([i for i, label in enumerate(labels) if label == person])
        samples = X_pca[idx]
        centroid = samples.mean(axis=0)
        dists = np.linalg.norm(samples - centroid, axis=1)
        cutoff = np.percentile(dists, percentile)
        keep[idx[dists > cutoff]] = False
        log.info(
            "  %s: keeping %d/%d samples after cleaning",
            person, int(keep[idx].sum()), len(idx))

    cleaned = [row for row, keep_row in zip(rows, keep) if keep_row]
    log.info(
        "Cleaned mesh data: keeping %d/%d rows", len(cleaned), len(rows))

    return cleaned


# ── one-person mode ───────────────────────────────────────────────────────────
def train_one_person(X_pca, name):
    centroid = X_pca.mean(axis=0)
    dists    = np.linalg.norm(X_pca - centroid, axis=1)
    thresh   = np.percentile(dists, THRESHOLD_PCTILE) * THRESHOLD_FACTOR
    log.info(f"  {name}: {len(X_pca)} samples | "
             f"median dist: {np.median(dists):.3f} | threshold: {thresh:.3f}")
    return {"mode": "one_person", "centroid": centroid,
            "threshold": thresh, "name": name}


def make_mlp():
    return MLPClassifier(
        hidden_layer_sizes=(512, 256, 128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        max_iter=2000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        verbose=False,
    )


def make_svm():
    """
    RBF SVM with Platt scaling (sigmoid calibration).
    Platt scaling gives more realistic probabilities than isotonic
    on small datasets — isotonic overfits and collapses to 0/1.
    """
    base = SVC(kernel="rbf", C=10.0, gamma="scale",
               class_weight="balanced", random_state=42)
    return CalibratedClassifierCV(base, cv=5, method="sigmoid")


def calibrate_unknown_rejection(model, X, n_classes):
    """
    Set a fixed threshold based on what SVM/MLP confidences look like
    in practice. We use a hard floor of 0.55 — well above random (0.5)
    but low enough that live-video frames with slight noise still pass.
    """
    proba = model.predict_proba(X)
    best  = proba.max(axis=1)
    median_conf = float(np.median(best))
    log.info("Training confidence — median: %.3f  min: %.3f  max: %.3f",
             median_conf, float(best.min()), float(best.max()))
    # Use a fixed floor of 0.55 — do NOT derive from training data because
    # calibrated SVMs on clean augmented data return near-1.0 on training
    # samples, which would set an unreachable threshold for live video.
    conf_threshold = 0.55
    log.info("Unknown rejection threshold: %.2f  (fixed floor)", conf_threshold)
    return conf_threshold


# ── multi-person mode ─────────────────────────────────────────────────────────
def train_multi(X_pca, labels, le, n_classes, classifier):
    """
    Train the multi-person mesh classifier.

    classifier="svm"     — RBF SVM (default, best accuracy on small datasets)
    classifier="lda-mlp" — PCA -> LDA -> MLP  (faster inference)
    classifier="mlp"     — PCA -> MLP (no LDA compression)
    """
    y_enc = le.fit_transform(labels)

    lda = None
    if classifier in ("lda-mlp", "svm"):
        n_lda   = min(n_classes - 1, X_pca.shape[1])
        lda     = LDA(n_components=n_lda, solver="svd", store_covariance=True)
        X_model = lda.fit_transform(X_pca, y_enc)
        log.info("LDA: %dD -> %dD  (class-separating compression for %d people)",
                 X_pca.shape[1], X_model.shape[1], n_classes)
    else:
        X_model = X_pca
        log.info("Classifier input: PCA features at %dD (no LDA)", X_model.shape[1])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_model, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

    if classifier == "svm":
        model = make_svm()
        log.info("Training RBF SVM + Platt sigmoid calibration …")
    else:
        model = make_mlp()
        log.info("Training MLP …")

    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    log.info("\n── Evaluation ──────────────────────────────────────────────")
    log.info("\n" + classification_report(y_te, y_pred, target_names=le.classes_))

    # cross-validation accuracy
    if len(X_model) >= 10:
        cv_model = make_svm() if classifier == "svm" else make_mlp()
        cv_scores = cross_val_score(
            cv_model, X_model, y_enc,
            cv=StratifiedKFold(n_splits=min(5, len(np.unique(y_enc))),
                               shuffle=True, random_state=42),
            scoring="accuracy",
        )
        log.info("Cross-val accuracy: %.1f%% ± %.1f%%",
                 cv_scores.mean()*100, cv_scores.std()*100)

    conf_threshold = calibrate_unknown_rejection(model, X_model, n_classes)

    bundle = {
        "mode":           "multi_person",
        "model":          model,
        "encoder":        le,
        "classifier":     classifier,
        "use_lda":        lda is not None,
        "conf_threshold": conf_threshold,
        "margin_threshold": 0.0,   # not used for SVM — single threshold only
    }
    if lda is not None:
        bundle["lda"] = lda
    return bundle


# ── embedding mode ────────────────────────────────────────────────────────────



# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  default=DATA_FILE)
    parser.add_argument("--model", default=MODEL_FILE)
    # mesh-only backend
    parser.add_argument("--clean-percentile", type=float,
                        help="Drop per-person mesh outliers above this percentile")
    parser.add_argument("--classifier", choices=("svm", "mlp", "lda-mlp"),
                        default="svm",
                        help="Classifier: svm (default, most accurate), lda-mlp, mlp")
    args = parser.parse_args()

    # embedding backend removed; always train mesh model

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

    if args.clean_percentile is not None:
        rows = clean_mesh_rows(rows, args.clean_percentile)
        labels = np.array([r[0] for r in rows])
        X = np.array([r[1:] for r in rows], dtype=np.float32)
        unique_people = list(dict.fromkeys(labels.tolist()))
        n_classes = len(unique_people)

    log.info(f"Loaded {len(X)} samples, {n_classes} person(s): {unique_people}")
    log.info(f"Features: {X.shape[1]}D  "
             f"(coords={N_COORDS}, ratios={N_RATIOS}, spectral={N_SPECTRAL})\n")

    # ── preprocessing ─────────────────────────────────────────────────────────
    X_w      = apply_weights(X)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_w)

    # PCA: keep 97 % variance, capped at N_PCA_MAX, minimum 50 components
    n_comp_max = min(N_PCA_MAX, X_scaled.shape[0] - 1, X_scaled.shape[1])
    pca_full   = PCA(n_components=n_comp_max, random_state=42)
    pca_full.fit(X_scaled)
    cumvar     = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp     = int(np.searchsorted(cumvar, PCA_VARIANCE) + 1)
    n_comp     = max(n_comp, PCA_MIN_DIMS)    # never fewer than PCA_MIN_DIMS
    n_comp     = max(n_comp, n_classes * 4)
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
        bundle = train_multi(X_pca, labels, le, n_classes, args.classifier)

    bundle.update({
        "backend":         "mesh",
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
