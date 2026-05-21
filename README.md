# face-isomorphic

Graph-theory face recognition using MediaPipe's 478-landmark face mesh.

## How it works

Each face is represented as a **1534-dimensional feature vector**:

| Slice | Content | Size |
|---|---|---|
| `[0:1434]` | Pose-normalised 3-D landmark coordinates (478 × 3) | 1434 |
| `[1434:1484]` | Distance ratio features (scale/rotation invariant) | 50 |
| `[1484:1534]` | Laplacian eigenvalues of the face mesh graph | 50 |

The training pipeline is **StandardScaler → PCA → LDA → MLP**:
- **PCA** removes noise and reduces dimensionality (keeps 97% variance)
- **LDA** (Linear Discriminant Analysis) maximises the separation *between* enrolled people — this is what makes multi-face recognition accurate
- **MLP** is the final classifier

## Workflow

```
1. Enroll one or more people
   python enroll.py --name "Alice"
   python enroll.py --name "Bob"

2. (Optional) Remove outlier samples
   python clean_data.py

3. Train the model
   python train.py

4. Run recognition
   python recognize.py
```

## Scripts

| Script | Purpose |
|---|---|
| `enroll.py` | Guided 5-pose enrollment (300 samples + 2× augmentation = 900 rows per person) |
| `train.py` | Train PCA + LDA + MLP pipeline |
| `recognize.py` | Live multi-face recognition with background Laplacian computation |
| `clean_data.py` | Remove outlier samples (no model required — uses its own PCA) |
| `graph_analysis.py` | Live Laplacian spectrum visualiser |
| `visualize.py` | Neural network graph visualiser |

## Controls

**enroll.py**
- `SPACE` — start/pause capturing
- `N` — skip to next pose
- `Q` — quit and save

**recognize.py / graph_analysis.py**
- `Q` — quit

## Requirements

```
pip install -r requirements.txt
```

## Re-enrolling after code changes

If `face_utils.py` changes (feature layout), you must re-enroll and retrain.
The schema version (`SCHEMA_VER`) is embedded in the CSV header and model bundle
so mismatches are detected automatically.
