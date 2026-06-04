"""Train either the mesh classifier or the recommended embedding matcher."""

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

from face_utils import N_SPECTRAL, N_RATIOS, N_COORDS, FEAT_DIM, SCHEMA_VER
from embedding_utils import (
    EMBEDDING_DATA_FILE, EMBEDDING_DIM, l2_normalize,
)

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
EMBEDDING_THRESHOLD_FLOOR = 0.38
EMBEDDING_MARGIN = 0.08
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


def calibrate_unknown_rejection(model, X, floor=0.65):
    proba = model.predict_proba(X)
    best = proba.max(axis=1)
    ordered = np.sort(proba, axis=1)
    margins = ordered[:, -1] - ordered[:, -2] if proba.shape[1] > 1 else best
    conf_threshold = max(floor, float(np.percentile(best, 10) * 0.95))
    margin_threshold = max(0.05, float(np.percentile(margins, 10) * 0.80))
    log.info(
        "Unknown rejection: confidence >= %.2f and margin >= %.2f",
        conf_threshold, margin_threshold)
    return conf_threshold, margin_threshold


# ── multi-person mode ─────────────────────────────────────────────────────────
def train_multi(X_pca, labels, le, n_classes, classifier):
    """
    Train the multi-person mesh classifier.

    classifier="mlp" keeps the richer PCA representation and sends it directly
    to the MLP. classifier="lda-mlp" keeps the older PCA -> LDA -> MLP path.
    """
    y_enc = le.fit_transform(labels)

    lda = None
    if classifier == "lda-mlp":
        # LDA is limited to (n_classes - 1) dimensions. This can be useful with
        # many classes, but it is very aggressive for two-person datasets.
        n_lda = min(n_classes - 1, X_pca.shape[1])
        lda = LDA(n_components=n_lda, solver="svd", store_covariance=True)
        X_model = lda.fit_transform(X_pca, y_enc)
        log.info(f"LDA: {X_pca.shape[1]}D -> {X_model.shape[1]}D  "
                 f"(class-separating compression for {n_classes} people)")
    else:
        X_model = X_pca
        log.info(
            "Classifier input: PCA features kept at %dD (no LDA compression)",
            X_model.shape[1])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_model, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

    model = make_mlp()
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    log.info("\n── Evaluation ──────────────────────────────────────────────")
    log.info("\n" + classification_report(y_te, y_pred, target_names=le.classes_))

    # cross-validation for a more robust accuracy estimate
    if len(X_model) >= 10:
        cv_scores = cross_val_score(
            make_mlp(),
            X_model, y_enc,
            cv=StratifiedKFold(n_splits=min(5, len(np.unique(y_enc))),
                               shuffle=True, random_state=42),
            scoring="accuracy",
        )
        log.info(f"Cross-val accuracy: {cv_scores.mean()*100:.1f}% "
                 f"± {cv_scores.std()*100:.1f}%")

    conf_threshold, margin_threshold = calibrate_unknown_rejection(model, X_model)

    bundle = {
        "mode": "multi_person",
        "model": model,
        "encoder": le,
        "classifier": classifier,
        "use_lda": lda is not None,
        "conf_threshold": conf_threshold,
        "margin_threshold": margin_threshold,
    }
    if lda is not None:
        bundle["lda"] = lda
    return bundle


# ── embedding mode ────────────────────────────────────────────────────────────
def train_embedding(data_file, model_file):
    if not os.path.isfile(data_file):
        log.error(f"{data_file} not found. Run enroll.py --backend embedding first.")
        return

    with open(data_file, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    if not rows:
        log.error("Embedding data file is empty.")
        return

    labels = np.array([r[0] for r in rows])
    X = l2_normalize(np.array([r[1:] for r in rows], dtype=np.float32))
    if X.shape[1] != EMBEDDING_DIM:
        log.error(
            f"Embedding dimension mismatch: CSV has {X.shape[1]}, "
            f"expected {EMBEDDING_DIM}."
        )
        return

    people = list(dict.fromkeys(labels.tolist()))
    centroids = {}
    thresholds = {}
    log.info(f"Loaded {len(X)} embeddings, {len(people)} person(s): {people}")

    for person in people:
        samples = X[labels == person]
        centroid = l2_normalize(samples.mean(axis=0))
        sims = samples @ centroid
        threshold = max(
            EMBEDDING_THRESHOLD_FLOOR,
            float(np.percentile(sims, 5) - 0.03),
        )
        centroids[person] = centroid.astype(np.float32)
        thresholds[person] = threshold
        log.info(
            "  %s: %d samples | median cosine %.3f | threshold %.3f",
            person, len(samples), float(np.median(sims)), threshold)

    centroid_names = list(centroids)
    centroid_matrix = np.stack([centroids[p] for p in centroid_names])
    train_scores = X @ centroid_matrix.T
    pred_idx = np.argmax(train_scores, axis=1)
    pred = np.array([centroid_names[i] for i in pred_idx])
    train_acc = float(np.mean(pred == labels))
    log.info("Centroid training accuracy: %.1f%%", train_acc * 100)

    if len(people) > 1:
        same = train_scores[np.arange(len(X)), pred_idx]
        sorted_scores = np.sort(train_scores, axis=1)
        margins = sorted_scores[:, -1] - sorted_scores[:, -2]
        log.info(
            "Median best cosine %.3f | median identity margin %.3f",
            float(np.median(same)), float(np.median(margins)))

    bundle = {
        "backend": "embedding",
        "mode": "embedding_centroid",
        "people": people,
        "centroid_names": centroid_names,
        "centroids": centroid_matrix.astype(np.float32),
        "thresholds": thresholds,
        "margin_threshold": EMBEDDING_MARGIN,
        "embedding_dim": EMBEDDING_DIM,
        "schema_ver": "embedding_v1",
    }
    joblib.dump(bundle, model_file)
    log.info(f"\nEmbedding model saved → {model_file}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  default=DATA_FILE)
    parser.add_argument("--model", default=MODEL_FILE)
    parser.add_argument("--backend", choices=("mesh", "embedding"),
                        default="mesh")
    parser.add_argument("--embedding-data", default=EMBEDDING_DATA_FILE)
    parser.add_argument("--clean-percentile", type=float,
                        help="Drop per-person mesh outliers above this percentile")
    parser.add_argument("--classifier", choices=("mlp", "lda-mlp"),
                        default="mlp",
                        help="Mesh multi-person classifier (default: mlp keeps PCA dimensions)")
    args = parser.parse_args()

    if args.backend == "embedding":
        train_embedding(args.embedding_data, args.model)
        return

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
