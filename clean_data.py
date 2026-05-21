"""
clean_data.py  —  Remove outlier samples from face_data.csv.

For each person, finds samples that are far from their own centroid
(in PCA space) and removes them.  These are typically bad frames:
extreme angles, motion blur, partial occlusion.

This script is self-contained — it does NOT require a pre-trained model.
It fits its own lightweight PCA purely for outlier detection.

Usage:
    python clean_data.py
    python clean_data.py --percentile 90 --data face_data.csv

Reads  : face_data.csv
Writes : face_data.csv  (cleaned, in-place)
         face_data_backup.csv  (original backup)
"""

import argparse
import logging
import numpy as np
import csv
import os
import shutil

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from face_utils import N_SPECTRAL, N_RATIOS, N_COORDS, FEAT_DIM, SCHEMA_VER

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DATA_FILE   = "face_data.csv"
BACKUP_FILE = "face_data_backup.csv"

# Remove samples beyond this percentile of distance from their centroid.
# 90 = remove the worst 10 % of each person's samples.
DEFAULT_PERCENTILE = 90
# PCA components used only for outlier detection (not saved)
CLEAN_PCA_COMPONENTS = 50


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       default=DATA_FILE)
    parser.add_argument("--percentile", type=float, default=DEFAULT_PERCENTILE,
                        help="Outlier cutoff percentile (default 90)")
    args = parser.parse_args()

    if not os.path.isfile(args.data):
        log.error(f"{args.data} not found.")
        return

    # ── load data ─────────────────────────────────────────────────────────────
    with open(args.data, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows   = list(reader)

    if not rows:
        log.error("Data file is empty.")
        return

    labels = [r[0] for r in rows]
    X      = np.array([r[1:] for r in rows], dtype=np.float32)
    unique = list(dict.fromkeys(labels))

    if X.shape[1] != FEAT_DIM:
        log.warning(
            f"Feature dimension is {X.shape[1]}, expected {FEAT_DIM}. "
            "Proceeding anyway — results may be unreliable."
        )

    log.info(f"Loaded {len(rows)} samples: {unique}")

    # ── fit a lightweight PCA for outlier detection only ─────────────────────
    # Apply the same spectral/ratio upweighting as train.py so the distance
    # metric is consistent with what the model will see.
    X_w = X.copy()
    X_w[:, N_COORDS : N_COORDS + N_RATIOS] *= 2.0   # ratio weight
    X_w[:, -N_SPECTRAL:]                   *= 3.0   # spectral weight

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_w)

    n_comp = min(CLEAN_PCA_COMPONENTS, X_scaled.shape[0] - 1, X_scaled.shape[1])
    pca    = PCA(n_components=n_comp, random_state=42)
    X_pca  = pca.fit_transform(X_scaled)

    log.info(f"Outlier-detection PCA: {X_scaled.shape[1]}D → {n_comp}D  "
             f"({pca.explained_variance_ratio_.sum()*100:.1f}% variance)\n")

    # ── find outliers per person ──────────────────────────────────────────────
    keep_mask     = np.ones(len(rows), dtype=bool)
    total_removed = 0

    for person in unique:
        idx      = np.array([i for i, l in enumerate(labels) if l == person])
        samples  = X_pca[idx]
        centroid = samples.mean(axis=0)
        dists    = np.linalg.norm(samples - centroid, axis=1)
        cutoff   = np.percentile(dists, args.percentile)

        outliers = idx[dists > cutoff]
        keep_mask[outliers] = False
        total_removed += len(outliers)

        log.info(f"  {person}: {len(idx)} samples | "
                 f"cutoff dist: {cutoff:.3f} | "
                 f"removing {len(outliers)} outliers")

    kept = keep_mask.sum()
    log.info(f"\nKeeping {kept}/{len(rows)} samples "
             f"(removed {total_removed} outliers)\n")

    # ── backup + write cleaned data ───────────────────────────────────────────
    shutil.copy(args.data, BACKUP_FILE)
    log.info(f"Backup saved → {BACKUP_FILE}")

    with open(args.data, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, row in enumerate(rows):
            if keep_mask[i]:
                writer.writerow(row)

    log.info(f"Cleaned data saved → {args.data}")
    log.info("\nNow run train.py to retrain on the cleaned data.")


if __name__ == "__main__":
    main()
