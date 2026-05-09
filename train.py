import numpy as np
import csv
import joblib
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

DATA_FILE        = "face_data.csv"
MODEL_FILE       = "face_model.pkl"
N_PCA            = 150
SPECTRAL_WEIGHT  = 3.0
THRESHOLD_PCTILE = 95
THRESHOLD_FACTOR = 1.2


def apply_spectral_weighting(X, n_spectral=50):
    X = X.copy()
    X[:, -n_spectral:] *= SPECTRAL_WEIGHT
    return X


# ── single person: nearest-centroid ──────────────────────────────────────────
def train_one_person(X_scaled, pca, name):
    X_pca    = pca.fit_transform(X_scaled)
    centroid = X_pca.mean(axis=0)
    dists    = np.linalg.norm(X_pca - centroid, axis=1)
    thresh   = np.percentile(dists, THRESHOLD_PCTILE) * THRESHOLD_FACTOR

    print(f"  {name}: {len(X_pca)} samples | "
          f"median dist: {np.median(dists):.3f} | threshold: {thresh:.3f}")

    return {
        "mode":             "one_person",
        "pca":              pca,
        "centroid":         centroid,
        "threshold":        thresh,
        "name":             name,
    }


# ── multiple people: MLP ──────────────────────────────────────────────────────
def train_multi_person(X_scaled, labels, le, pca):
    X_pca = pca.fit_transform(X_scaled)
    y_enc = le.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    print(f"Training MLP on {X_pca.shape[1]}D PCA features...")
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        verbose=False,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n── Evaluation ──────────────────────────────────────────────")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    return {
        "mode":    "multi_person",
        "pca":     pca,
        "model":   model,
        "encoder": le,
    }


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    if not os.path.isfile(DATA_FILE):
        print(f"ERROR: {DATA_FILE} not found. Run enroll.py first.")
        return

    with open(DATA_FILE, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        rows = list(reader)

    labels        = np.array([r[0] for r in rows])
    X             = np.array([r[1:] for r in rows], dtype=np.float32)
    unique_people = list(dict.fromkeys(labels.tolist()))

    print(f"Loaded {len(X)} samples, {len(unique_people)} person(s): {unique_people}")
    print(f"Feature vector: {X.shape[1]}D\n")

    # weight + scale
    X_w      = apply_spectral_weighting(X)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_w)

    n_components = min(N_PCA, X_scaled.shape[0] - 1, X_scaled.shape[1])
    pca          = PCA(n_components=n_components, random_state=42)

    if len(unique_people) == 1:
        print("Single person — using nearest-centroid with PCA\n")
        bundle = train_one_person(X_scaled, pca, unique_people[0])
    else:
        print(f"Multiple people — using MLP neural network\n")
        le     = LabelEncoder()
        bundle = train_multi_person(X_scaled, labels, le, pca)

    bundle["scaler"]          = scaler
    bundle["spectral_weight"] = SPECTRAL_WEIGHT
    bundle["people"]          = unique_people

    explained = pca.explained_variance_ratio_.sum() * 100
    print(f"PCA: {X_scaled.shape[1]}D → {n_components}D  ({explained:.1f}% variance)\n")

    joblib.dump(bundle, MODEL_FILE)
    print(f"Model saved → {MODEL_FILE}")


if __name__ == "__main__":
    main()
