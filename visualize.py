"""
visualize.py  —  Visualize the trained neural network and its training history.

The neural network IS a directed weighted graph:
  Nodes  = neurons (grouped by layer)
  Edges  = weighted connections between neurons (the learned weights)
  Direction = input → hidden layers → output

Works for both one_person (centroid) and multi_person (MLP + LDA) models.

Usage:
    python visualize.py

Shows 5 plots:
    1. Network as a directed weighted graph (graph theory view)
    2. Network architecture diagram (classic NN view)
    3. Training loss curve
    4. Weight heatmaps for each layer
    5. Per-class output activations
"""

import numpy as np
import joblib
import os
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

MODEL_FILE = "face_model.pkl"


def load_bundle():
    if not os.path.isfile(MODEL_FILE):
        print(f"ERROR: {MODEL_FILE} not found. Run train.py first.")
        exit(1)
    return joblib.load(MODEL_FILE)


def check_multi_person(bundle):
    """Return True if the bundle contains a trained MLP, False otherwise."""
    if bundle.get("mode") != "multi_person":
        print(
            f"Model is in '{bundle.get('mode')}' mode (only one person enrolled).\n"
            "Skipping MLP-specific plots (loss curve, weight heatmaps, activations).\n"
            "Enroll at least two people and retrain to see the full visualisation."
        )
        return False
    return True


# ── 0. Neural network as a directed weighted graph ───────────────────────────
def plot_nn_as_graph(ax, model, encoder):
    """
    Build a NetworkX DiGraph where:
      - Each neuron is a node  (named layer_neuronIndex)
      - Each weight is a directed edge with the weight value
    Only a sample of neurons/edges is drawn to keep it readable.
    """
    n_inputs    = model.coefs_[0].shape[0]
    hidden      = list(model.hidden_layer_sizes)
    n_outputs   = model.coefs_[-1].shape[1]
    layer_sizes = [n_inputs] + hidden + [n_outputs]

    MAX_PER_LAYER = 8

    DG = nx.DiGraph()

    sampled = []
    for size in layer_sizes:
        idx = np.linspace(0, size - 1, min(size, MAX_PER_LAYER), dtype=int)
        sampled.append(idx)

    n_layers = len(layer_sizes)
    for li, indices in enumerate(sampled):
        x  = li / (n_layers - 1)
        ys = np.linspace(0.1, 0.9, len(indices))
        for orig_idx, y in zip(indices, ys):
            node_id = f"L{li}_N{orig_idx}"
            DG.add_node(node_id, pos=(x, y), layer=li)

    for li in range(len(layer_sizes) - 1):
        coef = model.coefs_[li]
        for src_orig in sampled[li]:
            for dst_orig in sampled[li + 1]:
                w = coef[src_orig, dst_orig]
                DG.add_edge(f"L{li}_N{src_orig}",
                            f"L{li+1}_N{dst_orig}",
                            weight=w)

    pos    = nx.get_node_attributes(DG, "pos")
    layers = nx.get_node_attributes(DG, "layer")

    layer_colors = plt.cm.plasma(np.linspace(0.1, 0.9, n_layers))
    node_colors  = [layer_colors[layers[n]] for n in DG.nodes()]

    weights     = [DG[u][v]["weight"] for u, v in DG.edges()]
    wmax        = max(abs(w) for w in weights) if weights else 1
    edge_colors = [plt.cm.RdBu_r((w / wmax + 1) / 2) for w in weights]
    edge_widths = [0.5 + 1.5 * abs(w) / wmax for w in weights]

    nx.draw_networkx_nodes(DG, pos, ax=ax, node_color=node_colors,
                           node_size=120, alpha=0.95)
    nx.draw_networkx_edges(DG, pos, ax=ax, edge_color=edge_colors,
                           width=edge_widths, alpha=0.6,
                           arrows=True, arrowsize=8,
                           connectionstyle="arc3,rad=0.05")

    layer_names = (["Input"] +
                   [f"H{i+1}" for i in range(len(hidden))] +
                   ["Output"])
    for li, name in enumerate(layer_names):
        x = li / (n_layers - 1)
        ax.text(x, 0.02, name, ha="center", va="bottom",
                color="white", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="#333", ec="none"))

    n_nodes = DG.number_of_nodes()
    n_edges = DG.number_of_edges()
    density = nx.density(DG)
    ax.set_title(
        f"Neural Network as Directed Weighted Graph\n"
        f"Nodes: {n_nodes}  |  Edges: {n_edges}  |  Density: {density:.3f}  "
        f"(sampled {MAX_PER_LAYER} neurons/layer)",
        color="white", fontsize=10, pad=8
    )
    ax.set_facecolor("#1e1e1e")
    ax.axis("off")


def plot_architecture(ax, model, encoder):
    """Draw circles for neurons in each layer, connected by lines."""
    n_inputs    = model.coefs_[0].shape[0]
    hidden      = list(model.hidden_layer_sizes)
    n_outputs   = model.coefs_[-1].shape[1]
    layer_sizes = [n_inputs] + hidden + [n_outputs]
    layer_names = (
        [f"Input\n({n_inputs})"]
        + [f"Hidden {i+1}\n({s})" for i, s in enumerate(hidden)]
        + [f"Output\n({n_outputs})"]
    )

    max_display = 12
    n_layers    = len(layer_sizes)
    x_positions = np.linspace(0.05, 0.95, n_layers)

    neuron_positions = []

    for li, (x, size) in enumerate(zip(x_positions, layer_sizes)):
        display = min(size, max_display)
        ys = np.linspace(0.1, 0.9, display)
        neuron_positions.append((x, ys))

        if li > 0:
            px, pys = neuron_positions[li - 1]
            for py in pys:
                for y in ys:
                    ax.plot([px, x], [py, y], color="#cccccc", lw=0.4, zorder=1)

        for y in ys:
            circle = plt.Circle((x, y), 0.025, color="#4C9BE8",
                                 ec="white", lw=1.2, zorder=3)
            ax.add_patch(circle)

        if size > max_display:
            ax.text(x, 0.04, "...", ha="center", va="center",
                    fontsize=11, color="#888888")

        ax.text(x, 0.97, layer_names[li], ha="center", va="top",
                fontsize=8, color="white",
                bbox=dict(boxstyle="round,pad=0.3", fc="#333333", ec="none"))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_facecolor("#1e1e1e")
    ax.axis("off")
    ax.set_title("Network Architecture", color="white", fontsize=13, pad=10)


# ── 2. Training loss curve ────────────────────────────────────────────────────
def plot_loss(ax, model):
    if not hasattr(model, "loss_curve_"):
        ax.text(0.5, 0.5, "No loss curve available", ha="center",
                va="center", color="white", transform=ax.transAxes)
        return

    ax.plot(model.loss_curve_, color="#4C9BE8", lw=2, label="Training loss")

    if hasattr(model, "best_loss_") and model.best_loss_ is not None:
        ax.axhline(model.best_loss_, color="#E87B4C", lw=1.5,
                   linestyle="--", label=f"Best loss: {model.best_loss_:.4f}")

    ax.set_xlabel("Epoch", color="white")
    ax.set_ylabel("Loss", color="white")
    ax.set_title("Training Loss Curve", color="white", fontsize=13)
    ax.legend(facecolor="#2a2a2a", labelcolor="white")
    ax.tick_params(colors="white")
    ax.set_facecolor("#1e1e1e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#555555")


# ── 3. Weight heatmaps ────────────────────────────────────────────────────────
def plot_weights(axes, model):
    """One heatmap per layer showing the weight matrix (sampled if large)."""
    for i, (coef, ax) in enumerate(zip(model.coefs_, axes)):
        max_show = 60
        rows     = coef.shape[0]
        cols     = coef.shape[1]
        row_idx  = np.linspace(0, rows - 1, min(rows, max_show), dtype=int)
        col_idx  = np.linspace(0, cols - 1, min(cols, max_show), dtype=int)
        sample   = coef[np.ix_(row_idx, col_idx)]

        vmax = np.abs(sample).max()
        im   = ax.imshow(sample, aspect="auto", cmap="RdBu_r",
                         vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        label = ("Input→H1" if i == 0
                 else f"H{i}→H{i+1}" if i < len(model.coefs_) - 1
                 else f"H{i}→Output")
        ax.set_title(f"Weights: {label}\n({rows}×{cols})",
                     color="white", fontsize=10)
        ax.set_xlabel("Neurons out", color="white", fontsize=8)
        ax.set_ylabel("Neurons in",  color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=7)
        ax.set_facecolor("#1e1e1e")
        for spine in ax.spines.values():
            spine.set_edgecolor("#555555")


# ── 4. Per-class activation bars ──────────────────────────────────────────────
def plot_activations(ax, model, encoder, scaler):
    """
    Feed a zero vector through the network and show the output probabilities.
    This is a baseline — replace with a real sample for meaningful output.
    """
    zero_input = np.zeros((1, model.coefs_[0].shape[0]))
    proba      = model.predict_proba(zero_input)[0]
    classes    = encoder.classes_

    colors = ["#4C9BE8" if p < 0.5 else "#E84C4C" for p in proba]
    bars   = ax.barh(classes, proba, color=colors, edgecolor="#555555")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability", color="white")
    ax.set_title("Output Layer Activations\n(zero input baseline)",
                 color="white", fontsize=13)
    ax.tick_params(colors="white")
    ax.set_facecolor("#1e1e1e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#555555")

    for bar, p in zip(bars, proba):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{p*100:.1f}%", va="center", color="white", fontsize=9)


# ── one-person centroid visualisation ─────────────────────────────────────────
def plot_centroid_info(fig, bundle):
    """Simple summary plot for single-person centroid models."""
    ax = fig.add_subplot(111)
    ax.set_facecolor("#1e1e1e")
    ax.axis("off")

    centroid  = bundle["centroid"]
    threshold = bundle["threshold"]
    name      = bundle["name"]

    info = (
        f"Mode: Single-person centroid\n\n"
        f"Enrolled person: {name}\n"
        f"Centroid dimensions: {centroid.shape[0]}\n"
        f"Recognition threshold: {threshold:.4f}\n\n"
        f"Enroll at least two people and retrain\n"
        f"to see the full MLP + LDA visualisation."
    )
    ax.text(0.5, 0.5, info, ha="center", va="center",
            color="white", fontsize=14, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.6", fc="#2a2a2a", ec="#555"))
    ax.set_title("Face Recognition Model — Single Person",
                 color="white", fontsize=14)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    bundle = load_bundle()

    plt.style.use("dark_background")

    if not check_multi_person(bundle):
        # one-person mode: show a simple summary
        fig = plt.figure(figsize=(10, 6), facecolor="#121212")
        fig.suptitle("Neural Network Visualizer — Face Recognition",
                     color="white", fontsize=14, y=0.99)
        plot_centroid_info(fig, bundle)
        plt.tight_layout()
        plt.savefig("nn_visualization.png", dpi=150, bbox_inches="tight",
                    facecolor="#121212")
        print("Saved → nn_visualization.png")
        plt.show()
        return

    model   = bundle["model"]
    scaler  = bundle["scaler"]
    encoder = bundle["encoder"]

    n_weight_plots = len(model.coefs_)

    fig = plt.figure(figsize=(20, 12), facecolor="#121212")
    fig.suptitle(
        "Neural Network Visualizer — Face Recognition  |  PCA + LDA + MLP",
        color="white", fontsize=14, y=0.99
    )

    gs0 = gridspec.GridSpec(1, 1, figure=fig,
                            left=0.04, right=0.96,
                            top=0.93, bottom=0.67, wspace=0.3)
    gs1 = gridspec.GridSpec(1, 2, figure=fig,
                            left=0.04, right=0.96,
                            top=0.62, bottom=0.38, wspace=0.3)
    gs2 = gridspec.GridSpec(1, n_weight_plots + 1, figure=fig,
                            left=0.04, right=0.96,
                            top=0.33, bottom=0.05, wspace=0.4)

    ax_graph   = fig.add_subplot(gs0[0])
    ax_arch    = fig.add_subplot(gs1[0])
    ax_loss    = fig.add_subplot(gs1[1])
    ax_weights = [fig.add_subplot(gs2[i]) for i in range(n_weight_plots)]
    ax_act     = fig.add_subplot(gs2[n_weight_plots])

    plot_nn_as_graph(ax_graph, model, encoder)
    plot_architecture(ax_arch, model, encoder)
    plot_loss(ax_loss, model)
    plot_weights(ax_weights, model)
    plot_activations(ax_act, model, encoder, scaler)

    plt.savefig("nn_visualization.png", dpi=150, bbox_inches="tight",
                facecolor="#121212")
    print("Saved → nn_visualization.png")
    plt.show()


if __name__ == "__main__":
    main()
