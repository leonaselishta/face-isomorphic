"""Open3D brain-style visualization for the face recognition model."""

import argparse
import os

import joblib
import numpy as np


MODEL_FILE = "face_model.pkl"
EDGES_PER_LAYER = 320
NODE_RADIUS = 0.18


def require_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise SystemExit(
            "Open3D is not installed. Install it with:\n"
            "  pip install open3d\n"
            "Then run this script again."
        ) from exc
    return o3d


def layer_offsets(layers):
    offsets = []
    total = 0
    for size in layers:
        offsets.append(total)
        total += size
    return offsets


def build_nodes(layers, z_spacing=4.0):
    nodes = []
    for li, size in enumerate(layers):
        side = int(np.ceil(np.sqrt(size)))
        x_center = (side - 1) / 2.0
        y_center = (side - 1) / 2.0
        z = li * z_spacing
        for i in range(size):
            nodes.append([i % side - x_center, i // side - y_center, z])
    return np.asarray(nodes, dtype=np.float64)


def build_strongest_edges(weights, layers, edges_per_layer):
    offsets = layer_offsets(layers)
    line_chunks = []
    color_chunks = []

    for li, weight_matrix in enumerate(weights):
        # scikit-learn MLP stores weights as (input_neurons, output_neurons).
        flat_abs = np.abs(weight_matrix).ravel()
        if flat_abs.size == 0:
            continue

        k = min(int(edges_per_layer), flat_abs.size)
        if k < flat_abs.size:
            pick = np.argpartition(-flat_abs, k - 1)[:k]
            pick = pick[np.argsort(-flat_abs[pick])]
        else:
            pick = np.argsort(-flat_abs)

        src, dst = np.unravel_index(pick, weight_matrix.shape)
        lines = np.column_stack([
            offsets[li] + src,
            offsets[li + 1] + dst,
        ])
        line_chunks.append(lines)

        signed = weight_matrix.ravel()[pick]
        colors = np.zeros((len(pick), 3), dtype=np.float64)
        colors[signed >= 0] = (0.22, 0.82, 0.98)
        colors[signed < 0] = (0.98, 0.48, 0.42)
        color_chunks.append(colors * 0.55)

    if not line_chunks:
        return (
            np.zeros((0, 2), dtype=np.int32),
            np.zeros((0, 3), dtype=np.float64),
        )

    return (
        np.vstack(line_chunks).astype(np.int32),
        np.vstack(color_chunks),
    )


def pseudo_edges(layers, edges_per_layer):
    offsets = layer_offsets(layers)
    rng = np.random.default_rng(42)
    chunks = []
    colors = []
    for li, (left, right) in enumerate(zip(layers[:-1], layers[1:])):
        k = min(edges_per_layer, left * right)
        src = rng.integers(0, left, size=k)
        dst = rng.integers(0, right, size=k)
        chunks.append(np.column_stack([offsets[li] + src, offsets[li + 1] + dst]))
        color = np.array([[0.35, 0.62, 0.90]], dtype=np.float64)
        colors.append(np.repeat(color, k, axis=0) * 0.45)
    return np.vstack(chunks).astype(np.int32), np.vstack(colors)


def architecture(bundle):
    if bundle.get("backend") == "embedding":
        people = bundle.get("people", [])
        return {
            "title": "Embedding centroid recognizer",
            "layers": [478, int(bundle.get("embedding_dim", 512)), len(people)],
            "weights": None,
            "note": "No MLP weights: showing embedding comparison pipeline.",
        }

    if bundle.get("mode") == "multi_person":
        model = bundle["model"]
        layers = [int(model.coefs_[0].shape[0])]
        layers.extend(int(c.shape[1]) for c in model.coefs_)
        return {
            "title": "Mesh MLP neural network",
            "layers": layers,
            "weights": model.coefs_,
            "note": "Showing strongest learned MLP connections.",
        }

    feat_dim = int(bundle.get("feat_dim", 1534))
    pca_dim = int(bundle["centroid"].shape[0])
    return {
        "title": "Single-person centroid pipeline",
        "layers": [feat_dim, pca_dim, 1],
        "weights": None,
        "note": (
            "No neural network exists for one-person mode. Showing pipeline "
            "nodes with sampled conceptual links."
        ),
    }


def node_colors_for_layers(layers):
    colors = []
    denom = max(len(layers) - 1, 1)
    for li, size in enumerate(layers):
        t = li / denom
        color = [0.18 + t * 0.55, 0.52, 1.0 - t * 0.45]
        colors.extend([color] * size)
    return np.asarray(colors, dtype=np.float64)


def make_spheres(o3d, nodes, colors, radius):
    spheres = []
    for point, color in zip(nodes, colors):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.compute_vertex_normals()
        sphere.translate(point)
        sphere.paint_uniform_color(color)
        spheres.append(sphere)
    return spheres


def print_summary(info, edges_per_layer):
    layers = info["layers"]
    links = [a * b for a, b in zip(layers[:-1], layers[1:])]
    print("\n" + info["title"])
    print(info["note"])
    print("Layers:", " -> ".join(f"{n:,}" for n in layers))
    print("Total nodes:", f"{sum(layers):,}")
    print("Total possible links:", f"{sum(links):,}")
    print("Visible links per layer pair:", f"{edges_per_layer:,}")
    print("Controls: use the Open3D window to rotate, zoom, and pan.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_FILE)
    parser.add_argument("--edges-per-layer", type=int, default=EDGES_PER_LAYER)
    parser.add_argument("--node-radius", type=float, default=NODE_RADIUS)
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        raise SystemExit(f"{args.model} not found. Run train.py first.")

    o3d = require_open3d()
    bundle = joblib.load(args.model)
    info = architecture(bundle)
    layers = info["layers"]
    nodes = build_nodes(layers)
    node_colors = node_colors_for_layers(layers)

    if info["weights"] is None:
        lines, line_colors = pseudo_edges(layers, args.edges_per_layer)
    else:
        lines, line_colors = build_strongest_edges(
            info["weights"], layers, args.edges_per_layer)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(nodes)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(np.clip(line_colors, 0.0, 1.0))

    spheres = make_spheres(o3d, nodes, node_colors, args.node_radius)
    print_summary(info, args.edges_per_layer)
    o3d.visualization.draw_geometries(
        spheres + [line_set],
        window_name="Face Model Brain Visualization",
    )


if __name__ == "__main__":
    main()
