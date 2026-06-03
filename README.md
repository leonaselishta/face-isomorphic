# face-isomorphic

Face recognition with two backends:

- **Embedding backend (recommended):** InsightFace/ArcFace-style 512-D face
  embeddings with cosine thresholds.
- **Mesh backend:** MediaPipe's 478-landmark face mesh plus graph features.

## How it works

Each face is represented as a **1534-dimensional feature vector**:

| Slice | Content | Size |
|---|---|---|
| `[0:1434]` | Pose-normalised 3-D landmark coordinates (478 × 3) | 1434 |
| `[1434:1484]` | Distance ratio features (scale/rotation invariant) | 50 |
| `[1484:1534]` | Laplacian eigenvalues of the face mesh graph | 50 |

The mesh training pipeline is **StandardScaler → PCA → LDA → MLP**:
- **PCA** removes noise and reduces dimensionality (keeps 97% variance)
- **LDA** (Linear Discriminant Analysis) maximises the separation *between* enrolled people — this is what makes multi-face recognition accurate
- **MLP** is the final classifier

## Workflow

```
1. Enroll one or more people
   python enroll.py --name "Alice"
   python enroll.py --name "Bob"

2. Train the recommended embedding model
   python train.py --backend embedding

3. Run recognition
   python recognize.py
```

For the original graph/mesh model, train with optional outlier cleaning:

```
python train.py --backend mesh --clean-percentile 90
```

## Scripts

| Script | Purpose |
|---|---|
| `enroll.py` | Guided 5-pose enrollment with blur/lighting/pose quality gates |
| `train.py` | Train embedding thresholds or the PCA + LDA + MLP mesh pipeline |
| `recognize.py` | Live multi-face recognition with embedding or mesh backend |

## Controls

**enroll.py**
- `SPACE` — start/pause capturing
- `N` — skip to next pose
- `Q` — quit and save

**recognize.py**
- `Q` — quit

## Requirements

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

For the stronger embedding backend, install the optional dependencies too:

```
pip install -r requirements-embedding.txt
```

On Windows, `insightface` may require Microsoft C++ Build Tools. If that install
fails, the mesh backend still works with the base requirements.

## Re-enrolling after code changes

If `face_utils.py` changes (feature layout), you must re-enroll and retrain.
The schema version (`SCHEMA_VER`) is embedded in the CSV header and model bundle
so mismatches are detected automatically.

## Accuracy tips

- Enroll every person you want to distinguish; one-person data can only learn
  "known person" vs "Unknown".
- Use the embedding backend for best identity separation.
- Enroll each person in similar lighting to the recognition environment.
- Keep classes balanced: roughly the same number of accepted samples per person.
- Re-enroll when changing cameras, camera distance, or lighting significantly.
