"""Open3D brain-style visualization for the face recognition model.

Controls (shown in console on launch):
  L  — toggle edge/link lines on or off
  Q  — close the window
"""

import argparse
import os

import joblib
import numpy as np


MODEL_FILE      = "face_model.pkl"
EDGES_PER_LAYER = 320
NODE_RADIUS     = 0.22
Z_SPACING       = 4.0
SPHERE_RES      = 1    # 1 = 80 tri/sphere (fast);  2 = 320 (smoother)


# ── Open3D import ─────────────────────────────────────────────────────────────
def require_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise SystemExit(
            "Open3D is not installed.\n  pip install open3d"
        ) from exc
    return o3d


# ── node positions ────────────────────────────────────────────────────────────
def layer_offsets(layers):
    offsets, total = [], 0
    for size in layers:
        offsets.append(total)
        total += size
    return offsets


def build_nodes(layers, z_spacing=Z_SPACING):
    nodes = []
    for li, size in enumerate(layers):
        side     = int(np.ceil(np.sqrt(size)))
        x_center = (side - 1) / 2.0
        y_center = (side - 1) / 2.0
        z        = li * z_spacing
        for i in range(size):
            nodes.append([i % side - x_center, i // side - y_center, z])
    return np.asarray(nodes, dtype=np.float64)


# ── node colors — professional cool-to-warm gradient ─────────────────────────
# Deep navy (input) → steel blue → slate → warm silver (output).
# Desaturated, muted tones that look clean against a dark background.
_LAYER_PALETTE = [
    (0.18, 0.35, 0.62),   # deep navy blue    — input
    (0.25, 0.52, 0.78),   # steel blue
    (0.35, 0.65, 0.82),   # sky blue
    (0.45, 0.72, 0.76),   # teal-blue
    (0.58, 0.76, 0.70),   # sage
    (0.72, 0.78, 0.65),   # warm sage
    (0.86, 0.80, 0.58),   # warm sand
    (0.92, 0.72, 0.48),   # soft amber       — output
]


def node_colors_for_layers(layers):
    """
    Interpolate smoothly through _LAYER_PALETTE across all layers.
    Every node in a layer shares the same color.
    """
    n      = len(layers)
    pal    = np.asarray(_LAYER_PALETTE, dtype=np.float64)
    colors = []
    for li, size in enumerate(layers):
        t     = li / max(n - 1, 1)                # 0.0 … 1.0
        idx_f = t * (len(pal) - 1)
        lo    = int(idx_f)
        hi    = min(lo + 1, len(pal) - 1)
        frac  = idx_f - lo
        color = (1.0 - frac) * pal[lo] + frac * pal[hi]
        colors.extend([color.tolist()] * size)
    return np.asarray(colors, dtype=np.float64)


# ── sphere mesh (single merged mesh for speed) ────────────────────────────────
def make_sphere_mesh(o3d, nodes, colors, radius, subdivision=SPHERE_RES):
    template = o3d.geometry.TriangleMesh.create_icosahedron(radius=radius)
    template = template.subdivide_midpoint(number_of_iterations=subdivision)
    t_verts  = np.asarray(template.vertices)
    t_tris   = np.asarray(template.triangles)
    nv, nt   = len(t_verts), len(t_tris)
    nn       = len(nodes)

    all_verts  = np.empty((nn * nv, 3), dtype=np.float64)
    all_tris   = np.empty((nn * nt, 3), dtype=np.int32)
    all_colors = np.empty((nn * nv, 3), dtype=np.float64)

    for i, (center, color) in enumerate(zip(nodes, colors)):
        v0 = i * nv;  t0 = i * nt
        all_verts [v0:v0+nv] = t_verts + center
        all_tris  [t0:t0+nt] = t_tris  + v0
        all_colors[v0:v0+nv] = color

    mesh             = o3d.geometry.TriangleMesh()
    mesh.vertices    = o3d.utility.Vector3dVector(all_verts)
    mesh.triangles   = o3d.utility.Vector3iVector(all_tris)
    mesh.vertex_colors = o3d.utility.Vector3dVector(
        np.clip(all_colors, 0.0, 1.0))
    mesh.compute_vertex_normals()
    return mesh


# ── edge helpers ──────────────────────────────────────────────────────────────
def build_strongest_edges(weights, mlp_layers, edges_per_layer):
    """
    Build edges for the MLP portion only.

    mlp_layers : the layer sizes that the MLP weight matrices span,
                 i.e. [mlp_input, h1, h2, ..., output].
                 len(mlp_layers) == len(weights) + 1
    """
    offsets      = layer_offsets(mlp_layers)
    line_chunks  = []
    color_chunks = []

    for li, weight_matrix in enumerate(weights):
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
        line_chunks.append(np.column_stack([
            offsets[li]     + src,
            offsets[li + 1] + dst,
        ]))
        signed = weight_matrix.ravel()[pick]
        c = np.zeros((len(pick), 3), dtype=np.float64)
        c[signed >= 0] = (0.45, 0.65, 0.90)
        c[signed <  0] = (0.90, 0.45, 0.38)
        color_chunks.append(c * 0.50)

    if not line_chunks:
        return (np.zeros((0, 2), dtype=np.int32),
                np.zeros((0, 3), dtype=np.float64))
    return (np.vstack(line_chunks).astype(np.int32),
            np.vstack(color_chunks))


def pseudo_edges(layers, edges_per_layer):
    offsets = layer_offsets(layers)
    rng     = np.random.default_rng(42)
    chunks, colors = [], []
    for li, (left, right) in enumerate(zip(layers[:-1], layers[1:])):
        k   = min(edges_per_layer, left * right)
        src = rng.integers(0, left,  size=k)
        dst = rng.integers(0, right, size=k)
        chunks.append(
            np.column_stack([offsets[li] + src, offsets[li + 1] + dst]))
        colors.append(np.full((k, 3), [0.30, 0.45, 0.62], dtype=np.float64))
    return np.vstack(chunks).astype(np.int32), np.vstack(colors)


def pseudo_edges_for_range(layers, edges_per_layer, start_layer, end_layer):
    offsets = layer_offsets(layers)
    rng     = np.random.default_rng(42)
    chunks, colors = [], []
    for li in range(start_layer, end_layer):
        left, right = layers[li], layers[li + 1]
        k   = min(edges_per_layer, left * right)
        src = rng.integers(0, left,  size=k)
        dst = rng.integers(0, right, size=k)
        chunks.append(
            np.column_stack([offsets[li] + src, offsets[li + 1] + dst]))
        colors.append(np.full((k, 3), [0.28, 0.40, 0.58], dtype=np.float64))
    if not chunks:
        return (np.zeros((0, 2), dtype=np.int32),
                np.zeros((0, 3), dtype=np.float64))
    return np.vstack(chunks).astype(np.int32), np.vstack(colors)


def build_lineset(o3d, nodes, lines, line_colors):
    ls        = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(nodes)
    ls.lines  = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(
        np.clip(line_colors * 0.28, 0.0, 1.0))
    return ls


# ── architecture helpers ──────────────────────────────────────────────────────
def architecture(bundle):
    if bundle.get("mode") == "multi_person":
        model = bundle["model"]
        feat_dim = int(bundle.get("feat_dim", 1534))

        # robustly get PCA output dimension if available
        pca_obj = bundle.get("pca")
        if pca_obj is not None:
            if hasattr(pca_obj, "n_components_"):
                pca_dim = int(pca_obj.n_components_)
            elif hasattr(pca_obj, "n_components"):
                pca_dim = int(pca_obj.n_components)
            else:
                pca_dim = feat_dim
        else:
            pca_dim = feat_dim

        # If the model is an MLPClassifier it exposes `coefs_` — use those
        mlp_coefs = getattr(model, "coefs_", None)
        n_classes = len(bundle["encoder"].classes_)

        if mlp_coefs is not None:
            layers = [feat_dim, pca_dim]
            use_lda = bundle.get("use_lda", False)
            if use_lda and "lda" in bundle:
                lda_out = int(bundle["lda"].n_components)
                layers.append(lda_out)
                weight_start = len(layers) - 1
            else:
                weight_start = len(layers) - 1

            # MLP hidden layers from coefs (skip last coef's output — use encoder count)
            for c in mlp_coefs[:-1]:
                layers.append(int(c.shape[1]))
            layers.append(n_classes)   # true output count, not coefs[-1].shape[1]

            path = "PCA + LDA + MLP" if use_lda else "PCA + MLP"
            return {
                "title": f"Mesh MLP — {n_classes} people  ({', '.join(bundle['encoder'].classes_)})",
                "layers": layers,
                "weights": mlp_coefs,
                "weight_start_layer": weight_start,
                "note": f"{path}. Showing strongest learned MLP connections.",
            }

        # Non-MLP model (for example a CalibratedClassifierCV wrapping an SVM).
        # No `coefs_` to visualize; show PCA(+optional LDA) → classifier pipeline.
        use_lda = bundle.get("use_lda", False)
        if use_lda and "lda" in bundle:
            lda_out = int(bundle["lda"].n_components)
            layers = [feat_dim, pca_dim, lda_out, n_classes]
            weight_start = 2
            path = "PCA + LDA + classifier"
        else:
            layers = [feat_dim, pca_dim, n_classes]
            weight_start = 1
            path = "PCA + classifier"

        return {
            "title": f"Mesh classifier — {n_classes} people  ({', '.join(bundle['encoder'].classes_)})",
            "layers": layers,
            "weights": None,
            "weight_start_layer": weight_start,
            "note": f"{path}. No MLP weights available; showing conceptual links.",
        }
    feat_dim = int(bundle.get("feat_dim", 1534))
    pca_dim  = int(bundle["centroid"].shape[0])
    return {
        "title": "Single-person centroid pipeline",
        "layers": [feat_dim, pca_dim, 1],
        "weights": None,
        "note": "No neural network: showing conceptual pipeline links.",
    }


def demo_architecture(num_people=2):
    return {
        "title": "Demo multi-person mesh neural network",
        "layers": [1534, 200, 512, 256, 128, 64, int(num_people)],
        "weights": None,
        "note": "Demo only — links are conceptual, not trained weights.",
    }


# ── console summary ───────────────────────────────────────────────────────────
def print_summary(info, edges_per_layer):
    layers = info["layers"]
    links  = [a * b for a, b in zip(layers[:-1], layers[1:])]
    print()
    print("=" * 55)
    print(info["title"])
    print(info["note"])
    print("-" * 55)
    print("Layers      :", " -> ".join(f"{n:,}" for n in layers))
    print("Total nodes :", f"{sum(layers):,}")
    print("Total links :", f"{sum(links):,}")
    print("Shown/layer :", f"{edges_per_layer:,}")
    print("-" * 55)
    print("L — toggle edges    Q — quit")
    print("Mouse: rotate / scroll-zoom / right-drag pan")
    print("=" * 55)
    print()


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",           default=MODEL_FILE)
    parser.add_argument("--edges-per-layer", type=int,   default=EDGES_PER_LAYER)
    parser.add_argument("--node-radius",     type=float, default=NODE_RADIUS)
    parser.add_argument("--sphere-res",      type=int,   default=SPHERE_RES,
                        help="Ico-sphere subdivisions: 1=fast (default)  2=smooth")
    parser.add_argument("--demo-mlp",        action="store_true")
    parser.add_argument("--demo-people",     type=int,   default=2)
    args = parser.parse_args()

    if not args.demo_mlp and not os.path.isfile(args.model):
        raise SystemExit(f"{args.model} not found.  Run train.py first.")

    o3d = require_open3d()

    if args.demo_mlp:
        info = demo_architecture(args.demo_people)
    else:
        bundle = joblib.load(args.model)
        info   = architecture(bundle)

    layers      = info["layers"]
    nodes       = build_nodes(layers)
    node_colors = node_colors_for_layers(layers)

    if info["weights"] is None:
        lines, line_colors = pseudo_edges(layers, args.edges_per_layer)
    else:
        # pre-MLP layers (feat → PCA → optional LDA) get pseudo edges
        # MLP layers get real weight-based edges
        start  = int(info.get("weight_start_layer", len(layers) - len(info["weights"]) - 1))
        pl, pc = pseudo_edges_for_range(layers, args.edges_per_layer, 0, start)
        # mlp_layers is the slice of layers that the MLP weights span
        mlp_layers = layers[start:]
        ml, mc = build_strongest_edges(
            info["weights"], mlp_layers, args.edges_per_layer)
        # ml indices are relative to mlp_layers; shift them by the node offset of start
        node_offset = sum(layers[:start])
        ml = ml + node_offset
        lines       = np.vstack([pl, ml])
        line_colors = np.vstack([pc, mc])

    sphere_mesh  = make_sphere_mesh(
        o3d, nodes, node_colors, args.node_radius, args.sphere_res)
    edge_lineset = build_lineset(o3d, nodes, lines, line_colors)

    print_summary(info, args.edges_per_layer)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Face Model Brain Visualization",
                      width=1400, height=900)

    vis.add_geometry(edge_lineset)
    vis.add_geometry(sphere_mesh)

    opt = vis.get_render_option()
    opt.background_color    = np.array([0.05, 0.06, 0.10])  # dark charcoal-blue
    opt.light_on            = True
    opt.mesh_show_back_face = True

    state = {"edges_visible": True}

    def toggle_edges(vis_ref):
        state["edges_visible"] = not state["edges_visible"]
        if state["edges_visible"]:
            vis_ref.add_geometry(edge_lineset, reset_bounding_box=False)
            print("Edges: ON")
        else:
            vis_ref.remove_geometry(edge_lineset, reset_bounding_box=False)
            print("Edges: OFF")
        vis_ref.update_renderer()
        return False

    vis.register_key_callback(76, toggle_edges)   # L

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
