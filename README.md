# face-isomorphic

Face recognition from a webcam. The project can run in two modes:

- **Mesh backend:** uses MediaPipe Face Mesh landmarks, handcrafted geometry
  features, PCA, and a scikit-learn classifier. This is the default and the
  easiest mode to run on Windows.
- **Embedding backend:** removed. The project now uses the mesh backend
  (MediaPipe landmarks + MLP) by default.

## Project Flow

```text
enroll.py    -> collect face samples
train.py     -> train and save face_model.pkl
recognize.py -> run live webcam recognition
```

MediaPipe detects the face landmarks. The actual recognition/comparison is done
by the saved project model, `face_model.pkl`.

## Current Files

| File | Purpose |
|---|---|
| `enroll.py` | Opens the webcam and records face samples for a named person. It checks blur, lighting, face size, and pose before saving samples. |
| `train.py` | Trains `face_model.pkl` from the saved samples. Mesh mode uses StandardScaler, PCA, and MLP when there are 2+ people. LDA is optional. |
| `recognize.py` | Opens the webcam, detects faces, extracts features, and predicts a name or `Unknown`. |
| `face_utils.py` | Shared MediaPipe mesh feature extraction, pose estimation, quality checks, and graph features. |
| `neural_brain_viz.py` | Open3D 3D viewer for the model/pipeline nodes and sampled/strongest connections. |
| `face_data.csv` | Saved enrollment data for mesh mode. |
| `face_model.pkl` | Trained recognition model. |
| `shpjegim_kodi.txt` | Albanian explanation of the code. |

## Setup

Create and activate a virtual environment:

```powershell
python -m venv venv
venv\Scripts\activate
```

Install the base dependencies:

```powershell
pip install -r requirements.txt
```

This installs the mesh backend and Open3D visualization dependencies in one step.

## Mesh Backend Commands

Enroll at least two people if you want the model to tell faces apart:

```powershell
python enroll.py --name "Alice"
python enroll.py --name "Bob"
```

Train the mesh model:

```powershell
python train.py --clean-percentile 90
```

Run live recognition:

```powershell
python recognize.py
```

Press `Q` to quit the webcam window.

## Embedding Backend

InsightFace embedding support has been removed from this project. Use the
mesh backend (MediaPipe landmarks + MLP) which is easier to run on Windows.

## How The Mesh Model Works

For each detected face, `face_utils.py` creates a 1534-value vector:

| Slice | Content | Size |
|---|---|---|
| `0:1434` | Pose-normalized 3D coordinates from 478 MediaPipe landmarks | 1434 |
| `1434:1484` | Distance-ratio geometry features | 50 |
| `1484:1534` | Laplacian graph spectrum features | 50 |

Training then applies:

```text
feature weights -> StandardScaler -> PCA -> LDA -> MLPClassifier
```

The default is now:

```text
feature weights -> StandardScaler -> PCA -> MLPClassifier
```

The older LDA-compressed version is still available:

```powershell
python train.py --backend mesh --classifier lda-mlp --clean-percentile 90
```

Important behavior:

- With **one enrolled person**, the project does not train a neural network. It
  uses a centroid threshold: known person vs `Unknown`.
- With **two or more enrolled people**, `train.py` trains an MLP neural network.
  By default the MLP receives the richer PCA representation directly, avoiding
  LDA's `number_of_people - 1` compression limit. The hidden layers are fixed at
  `512 -> 256 -> 128 -> 64`.
- The output layer grows with the number of enrolled people.

## 3D Model Visualization

Run:

```powershell
python neural_brain_viz.py
```

The viewer shows all nodes in the displayed architecture. It does **not** show
all possible connections, because that can be hundreds of thousands of lines.
Instead:

- multi-person MLP mode shows the strongest learned connections
- one-person centroid mode shows sampled conceptual pipeline links
- one-person centroid mode shows sampled conceptual pipeline links

Useful options:

```powershell
python neural_brain_viz.py --edges-per-layer 150
python neural_brain_viz.py --node-radius 0.12
```

To preview the full multi-person neural network without enrolling another
person or changing `face_model.pkl`:

```powershell
python neural_brain_viz.py --demo-mlp --demo-people 2
```

## Accuracy Tips

- Enroll at least two people to train the neural-network classifier.
- Keep roughly the same number of samples per person.
- Enroll in good lighting and avoid motion blur.
- Re-train after adding new people.
- Use `--clean-percentile 90` in mesh mode to remove outlier samples.

## Notes

- `face_model.pkl` depends on the current feature layout. If `face_utils.py`
  changes, re-enroll and re-train.
- `venv/`, `__pycache__/`, generated visualizations, and backup CSVs are ignored
  by git.
