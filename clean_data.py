"""
clean_data.py  —  Remove outlier samples from face_data.csv.

For each person, finds samples that are far from their own centroid
(in PCA space) and removes them. These are typically bad frames:
extreme angles, motion blur, partial occlusion.

Usage:
    python clean_data.py

Reads  : face_data.csv
Writes : face_data.csv  (cleaned, in-place)
         face_data_backup.csv  (original backup)
"""

import numpy as np
import csv
import joblib
import os
import shutil

DATA_FILE   = "face_data.csv"
MODEL_FILE  = "face_model.pkl"
BACKUP_FILE = "face_data_backup.csv"

# Remove samples beyond this percentile of distance from their centroid.
# 90 = remove the worst 10% of each person's samples.
OUTLIER_PERCENTILE = 90


def main():
    if not os.path.isfile(MODEL_FILE):
        print(f"ERROR: {MODEL_FILE} not found. Run train.py first.")
        return
    if not os.path.isfile(DATA_FILE):
        print(f"ERROR: {DATA_FILE} not found.")
        return

    bundle = joblib.load(MODEL_FILE)
    if bundle["mode"] != "multi_person":
        print("Only needed for multi-person models.")
        return

    # ── load data ─────────────────────────────────────────────────────────────
    with open(DATA_FILE, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows   = list(reader)

    labels = [r[0] for r in rows]
    X      = np.array([r[1:] for r in rows], dtype=np.float32)
    unique = list(dict.fromkeys(labels))

    print(f"Loaded {len(rows)} samples: {unique}")

    # ── apply same pipeline as train.py ──────────────────────────────────────
    X_w  = X.copy()
    X_w[:, -50:] *= bundle["spectral_weight"]
    Xs   = bundle["scaler"].transform(X_w)
    Xp   = bundle["pca"].transform(Xs)

    # ── find outliers per person ──────────────────────────────────────────────
    keep_mask = np.ones(len(rows), dtype=bool)
    total_removed = 0

    for person in unique:
        idx      = np.array([i for i, l in enumerate(labels) if l == person])
        samples  = Xp[idx]
        centroid = samples.mean(axis=0)
        dists    = np.linalg.norm(samples - centroid, axis=1)
        cutoff   = np.percentile(dists, OUTLIER_PERCENTILE)

        outliers = idx[dists > cutoff]
        keep_mask[outliers] = False
        total_removed += len(outliers)

        print(f"  {person}: {len(idx)} samples | "
              f"cutoff dist: {cutoff:.3f} | "
              f"removing {len(outliers)} outliers")

    kept = keep_mask.sum()
    print(f"\nKeeping {kept}/{len(rows)} samples "
          f"(removed {total_removed} outliers)\n")

    # ── backup + write cleaned data ───────────────────────────────────────────
    shutil.copy(DATA_FILE, BACKUP_FILE)
    print(f"Backup saved → {BACKUP_FILE}")

    with open(DATA_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, row in enumerate(rows):
            if keep_mask[i]:
                writer.writerow(row)

    print(f"Cleaned data saved → {DATA_FILE}")
    print("\nNow run train.py to retrain on the cleaned data.")


if __name__ == "__main__":
    main()
