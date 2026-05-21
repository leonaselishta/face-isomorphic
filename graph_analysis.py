"""
graph_analysis.py  —  Live Laplacian spectrum analysis of the face graph.

The face mesh is modelled as a weighted graph G = (V, E):
  V = 478 MediaPipe landmarks (nodes)
  E = FACEMESH_TESSELATION connections (edges, weighted by 3-D distance)

The normalised Laplacian is:  L = D^{-1/2} (D - A) D^{-1/2}
  where D = degree matrix, A = weighted adjacency matrix.

Its eigenvalues  0 = λ₀ ≤ λ₁ ≤ … ≤ λₙ₋₁  form the SPECTRAL SIGNATURE
of the graph.  Key properties:
  - Invariant to node relabelling
  - Captures global topology (connectivity, clusters, symmetry)
  - Two faces with similar geometry → similar spectra
  - The spectral gap λ₁ indicates how well-connected the graph is

This script shows:
  Left panel  — live webcam with face mesh
  Right panel — live plot of the Laplacian eigenvalue spectrum
                + key graph metrics updated every frame

Controls:
    Q  – quit
    S  – save current spectrum to spectrum_snapshot.png
"""

import cv2
import mediapipe as mp
import numpy as np
import networkx as nx
import logging
import matplotlib
matplotlib.use("Agg")          # render to image buffer, not a GUI window
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import io

from face_utils import build_graph, laplacian_spectrum, N_SPECTRAL

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

mp_face_mesh   = mp.solutions.face_mesh
mp_drawing     = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles

PLOT_W, PLOT_H = 640, 480    # size of the matplotlib panel in pixels


def render_spectrum_plot(eigenvalues, graph_stats):
    """
    Render the eigenvalue spectrum + graph stats to a numpy image (BGR).
    """
    fig = plt.figure(figsize=(PLOT_W / 100, PLOT_H / 100),
                     facecolor="#0d0d0d", dpi=100)
    gs  = gridspec.GridSpec(2, 1, figure=fig,
                            top=0.88, bottom=0.12,
                            hspace=0.55)

    # ── top: full spectrum bar chart ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    x   = np.arange(len(eigenvalues))
    colors = plt.cm.plasma(eigenvalues / (eigenvalues.max() + 1e-9))
    ax1.bar(x, eigenvalues, color=colors, width=0.8)
    ax1.set_facecolor("#1a1a1a")
    ax1.set_xlabel("Eigenvalue index  k", color="white", fontsize=8)
    ax1.set_ylabel("λₖ", color="white", fontsize=9)
    ax1.set_title("Laplacian Spectrum  (face mesh graph)", color="white", fontsize=10)
    ax1.tick_params(colors="white", labelsize=7)
    for sp in ax1.spines.values():
        sp.set_edgecolor("#444")

    # annotate spectral gap λ₁
    if len(eigenvalues) > 1:
        ax1.annotate(
            f"λ₁ = {eigenvalues[1]:.4f}\n(spectral gap)",
            xy=(1, eigenvalues[1]),
            xytext=(len(eigenvalues) * 0.3, eigenvalues[1] + 0.05),
            color="#FFD700", fontsize=7,
            arrowprops=dict(arrowstyle="->", color="#FFD700", lw=1),
        )

    # ── bottom: graph metrics table ──────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#1a1a1a")
    ax2.axis("off")

    rows = [
        ["Metric", "Value", "Meaning"],
        ["Nodes  |V|",       str(graph_stats["nodes"]),
         "478 face landmarks"],
        ["Edges  |E|",       str(graph_stats["edges"]),
         "Tesselation connections"],
        ["Avg degree",       f"{graph_stats['avg_deg']:.2f}",
         "Avg connections per landmark"],
        ["Spectral gap λ₁",  f"{graph_stats['lambda1']:.4f}",
         "Graph connectivity (higher = better connected)"],
        ["Algebraic conn.",  f"{graph_stats['lambda1']:.4f}",
         "Fiedler value — measures robustness"],
        ["λ_max",            f"{graph_stats['lambda_max']:.4f}",
         "Largest eigenvalue (≤ 2 for norm. Laplacian)"],
        ["Spectral spread",  f"{graph_stats['lambda_max'] - graph_stats['lambda1']:.4f}",
         "λ_max − λ₁"],
    ]

    col_widths = [0.22, 0.18, 0.60]
    col_x      = [0.01, 0.24, 0.43]
    row_h      = 0.115
    for ri, row in enumerate(rows):
        y = 1.0 - ri * row_h
        for ci, (cell, _cw) in enumerate(zip(row, col_widths)):
            style = dict(color="white", fontsize=7.5, va="top")
            if ri == 0:
                style["fontweight"] = "bold"
                style["color"]      = "#FFD700"
            ax2.text(col_x[ci], y, cell, transform=ax2.transAxes, **style)

    fig.suptitle("Graph Theory — Face Mesh Analysis", color="white",
                 fontsize=11, y=0.97)

    # render to numpy array
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor="#0d0d0d")
    plt.close(fig)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img     = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    buf.close()
    return cv2.resize(img, (PLOT_W, PLOT_H))


def compute_graph_stats(G, eigenvalues):
    n       = G.number_of_nodes()
    degrees = [d for _, d in G.degree()]
    return {
        "nodes":      n,
        "edges":      G.number_of_edges(),
        "avg_deg":    sum(degrees) / n if n else 0,
        "lambda1":    float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0,
        "lambda_max": float(eigenvalues[-1]) if len(eigenvalues) > 0 else 0.0,
    }


def main():
    cap = cv2.VideoCapture(0)

    # placeholder plot shown before a face is detected
    placeholder = np.zeros((PLOT_H, PLOT_W, 3), dtype=np.uint8)
    cv2.putText(placeholder, "Waiting for face...", (120, PLOT_H // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 100), 2)
    plot_img = placeholder.copy()

    RECOMPUTE_EVERY = 5    # recompute spectrum every N frames (it's O(n³))
    frame_count     = 0
    save_next       = False

    # initialise stats and eigenvalues so they're always defined
    stats       = None
    eigenvalues = None

    log.info("Laplacian Spectrum Analyser")
    log.info("  S = save snapshot  |  Q = quit\n")

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = face_mesh.process(rgb)
            rgb.flags.writeable = True
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            if results.multi_face_landmarks:
                lms = results.multi_face_landmarks[0]

                # draw mesh on webcam frame
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=lms,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_draw_styles
                        .get_default_face_mesh_tesselation_style(),
                )
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=lms,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_draw_styles
                        .get_default_face_mesh_contours_style(),
                )

                # recompute spectrum periodically
                if frame_count % RECOMPUTE_EVERY == 0:
                    try:
                        G           = build_graph(lms)
                        eigenvalues = laplacian_spectrum(G, k=N_SPECTRAL)
                        stats       = compute_graph_stats(G, eigenvalues)
                        plot_img    = render_spectrum_plot(eigenvalues, stats)

                        if save_next:
                            cv2.imwrite("spectrum_snapshot.png", plot_img)
                            log.info("Saved → spectrum_snapshot.png")
                            save_next = False
                    except Exception as exc:
                        log.warning("Spectrum computation error: %s", exc)

                # overlay λ₁ on webcam frame — only if stats have been computed
                if stats is not None:
                    lam1_str = f"Spectral gap λ₁ = {stats['lambda1']:.4f}"
                    cv2.putText(frame, lam1_str, (10, h - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 230, 150), 1)

            # resize webcam frame to match plot height
            cam_resized = cv2.resize(frame, (PLOT_W, PLOT_H))

            # side-by-side display
            combined = np.hstack([cam_resized, plot_img])
            cv2.imshow("Laplacian Spectrum — Graph Theory Face Analysis", combined)
            frame_count += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                save_next = True

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
