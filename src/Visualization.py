"""
Brain Connectivity Visualization Tools
========================================

This module provides four types of visualizations:

1. Connectivity matrix heatmap
   Shows pairwise correlations as a colored grid.
   Red = positive correlation, Blue = negative correlation.

2. Glass brain plot
   Projects connections onto a transparent 3D brain rendering,
   shown from three angles (axial, sagittal, coronal).

3. Edge weight distribution
   Histogram of all connectivity values — useful for sanity checking.

4. DMN chord diagram
   Circular diagram showing the Default Mode Network's internal
   connectivity. Only useful for small subnetworks (5-15 regions).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns


# ═══════════════════════════════════════════════════════════════════════
# 1. CONNECTIVITY MATRIX HEATMAP
# ═══════════════════════════════════════════════════════════════════════

def plot_connectivity_matrix(
    matrix,
    labels=None,
    threshold=None,
    title="Functional Connectivity Matrix",
    cmap="RdBu_r",
    vmin=-1,
    vmax=1,
    figsize=(12, 10),
    save_path=None,
):
    """
    Plot the connectivity matrix as a colored heatmap.

    HOW TO READ THIS PLOT
    ----------------------
    • Each row and column represents one brain region
    • Cell color = correlation between row region and column region
    • Dark red   → strong positive correlation (regions co-activate)
    • Dark blue  → strong negative correlation (anti-correlation)
    • White/light → weak or no correlation
    • Diagonal is always dark red (self-correlation = 1.0)
    • Matrix is symmetric (upper triangle = lower triangle, mirrored)

    WHAT TO LOOK FOR
    -----------------
    • Block patterns: groups of regions correlating strongly with each
      other → these are functional networks (DMN, visual, motor, etc.)
    • Anti-correlated blocks: when one network activates, another
      deactivates (e.g., DMN vs. task-positive network)
    • Hub regions: single rows/columns with many strong connections
      → these are network hubs (e.g., posterior cingulate cortex)

    Parameters
    ----------
    matrix : numpy.ndarray, shape (n_regions, n_regions)
        Symmetric connectivity matrix from compute_connectivity().
    labels : list of str, optional
        Region names for axis ticks (only shown if ≤30 regions).
    threshold : float, optional
        Zero out connections with |r| < threshold for visual clarity.
    title : str
        Plot title.
    cmap : str
        Matplotlib colormap. "RdBu_r" = red-white-blue (reversed).
    vmin, vmax : float
        Color scale limits. Default [-1, 1] = full correlation range.
    figsize : tuple
        Figure dimensions in inches.
    save_path : str, optional
        File path to save the figure (e.g., "figures/conn_matrix.png").

    Returns
    -------
    matplotlib.figure.Figure
    """
    plot_matrix = matrix.copy()

    # Optional thresholding for visual clarity
    if threshold is not None:
        plot_matrix[np.abs(plot_matrix) < threshold] = 0

    fig, ax = plt.subplots(figsize=figsize)

    # Use seaborn — nicer defaults than raw matplotlib imshow
    sns.heatmap(
        plot_matrix,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=0,                     # white at zero
        square=True,                  # square cells (matrix is symmetric)
        linewidths=0.1,
        linecolor="gray",
        cbar_kws={"label": "Correlation (r)", "shrink": 0.8},
        ax=ax,
    )

    # Show region labels only if there aren't too many (otherwise unreadable)
    if labels is not None and len(labels) <= 30:
        ax.set_xticks(np.arange(len(labels)) + 0.5)
        ax.set_yticks(np.arange(len(labels)) + 0.5)
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticklabels(labels, rotation=0, fontsize=7)
    else:
        ax.set_xlabel("Brain Region Index")
        ax.set_ylabel("Brain Region Index")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Figure saved to {save_path}")

    return fig


# ═══════════════════════════════════════════════════════════════════════
# 2. GLASS BRAIN PLOT
# ═══════════════════════════════════════════════════════════════════════

def plot_glass_brain(
    matrix,
    coords,
    threshold=0.3,
    title="Functional Connectivity — Glass Brain",
    node_size="auto",
    edge_cmap="RdYlBu_r",
    save_path=None,
):
    """
    Plot connections on a transparent brain rendering.

    WHAT IS A GLASS BRAIN?
    -----------------------
    A semi-transparent 3D rendering of the brain that lets you see
    through it.  Brain regions appear as dots; connections appear as
    lines between dots.  Standard for visualizing brain networks
    because you can see all connections at once without occlusion.

    THREE VIEWS
    -----------
    • Axial (top-down):    z-axis projection — see left/right and front/back
    • Sagittal (side):     x-axis projection — see front/back and top/bottom
    • Coronal (front):     y-axis projection — see left/right and top/bottom

    NODE SIZE = DEGREE
    -------------------
    In "auto" mode, each node's size is proportional to how many strong
    connections it has (its "degree" in graph theory).

    Highly connected regions are called HUBS.  In the brain, common
    hubs include:
      • Posterior Cingulate Cortex (PCC) — center of the DMN
      • Precuneus — multimodal integration
      • Anterior Insula — switching between networks

    Hubs appear as the largest nodes in the plot.

    Parameters
    ----------
    matrix : numpy.ndarray
        Connectivity matrix.
    coords : numpy.ndarray, shape (n_regions, 3)
        MNI coordinates (x, y, z) for each region.
    threshold : float
        Show only edges with |correlation| above this value.
    title : str
        Main figure title.
    node_size : str or array-like
        "auto" → scaled by degree; or pass explicit sizes.
    edge_cmap : str
        Colormap for edges (red-yellow-blue reversed by default).
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from nilearn import plotting

    # Compute node degrees if "auto"
    if node_size == "auto":
        adj = np.abs(matrix) > threshold
        degree = adj.sum(axis=0)
        node_size = 20 + degree * 8   # base size 20, +8 per connection

    fig = plt.figure(figsize=(14, 5))

    # Plot all three views side by side
    for i, (display_mode, title_suffix) in enumerate([
        ("z", "Axial"),
        ("x", "Sagittal"),
        ("y", "Coronal"),
    ]):
        ax = fig.add_subplot(1, 3, i + 1)
        plotting.plot_connectome(
            matrix,
            coords,
            edge_threshold=f"{threshold * 100:.0f}%",
            node_size=node_size,
            edge_cmap=edge_cmap,
            display_mode=display_mode,
            colorbar=(i == 2),         # only show colorbar on last subplot
            title=title_suffix,
            axes=ax,
        )

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Figure saved to {save_path}")

    return fig


# ═══════════════════════════════════════════════════════════════════════
# 3. EDGE DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════

def plot_connectivity_distribution(matrix, title="Edge Weight Distribution", save_path=None):
    """
    Plot the distribution of connectivity values.

    WHY LOOK AT THIS?
    ------------------
    The shape of the distribution tells you a lot about your data:

    • Roughly normal, centered near 0 → healthy resting-state pattern
    • Heavily right-skewed (mostly positive) → motion artifacts likely
    • Bimodal (two peaks) → distinct subnetworks emerging
    • Very narrow → over-cleaning / signal lost
    • Very wide → under-cleaning / noise dominant

    THE PLOT HAS TWO PANELS
    ------------------------
    Left:  Histogram of correlation values.
    Right: Sorted edge strengths (largest first).
           Look for an "elbow" — point where strong edges end.

    Parameters
    ----------
    matrix : numpy.ndarray
        Connectivity matrix.
    title : str
        Figure title.
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Extract upper triangle (matrix is symmetric)
    upper = matrix[np.triu_indices_from(matrix, k=1)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left panel: histogram
    axes[0].hist(upper, bins=60, color="#2196F3", alpha=0.8, edgecolor="white")
    axes[0].axvline(0, color="red", linestyle="--", alpha=0.5)   # zero reference
    axes[0].set_xlabel("Correlation (r)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of Connectivity Weights")

    # Right panel: ranked edge strengths
    sorted_vals = np.sort(np.abs(upper))[::-1]   # descending
    axes[1].plot(sorted_vals, color="#FF5722", linewidth=1.5)
    axes[1].set_xlabel("Edge Rank")
    axes[1].set_ylabel("|Correlation|")
    axes[1].set_title("Ranked Edge Strengths")
    axes[1].axhline(0.3, color="gray", linestyle="--", alpha=0.5, label="r = 0.3")
    axes[1].legend()

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# ═══════════════════════════════════════════════════════════════════════
# 4. DMN CHORD DIAGRAM
# ═══════════════════════════════════════════════════════════════════════

def plot_dmn_chord(dmn_matrix, dmn_labels, threshold=0.2, save_path=None):
    """
    Plot DMN connectivity as a circular chord-style diagram.

    WHAT IS A CHORD DIAGRAM?
    -------------------------
    Nodes are placed evenly around a circle.  Connections between nodes
    are drawn as chords (lines crossing the interior).

    Visual encoding:
      • Line color    → red (positive r) or blue (negative r)
      • Line width    → proportional to |r|
      • Line opacity  → also proportional to |r|

    WHEN TO USE THIS?
    ------------------
    Best for SMALL networks (5-15 regions).  With more regions the
    diagram becomes a tangled mess — use the heatmap or glass brain
    instead.

    Parameters
    ----------
    dmn_matrix : numpy.ndarray, shape (n_dmn, n_dmn)
        Connectivity submatrix for DMN regions.
    dmn_labels : list of str
        Region names for the DMN.
    threshold : float
        Minimum |r| to draw a connection.
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n = len(dmn_labels)

    # Place nodes evenly around the circle (0 to 2π, n points)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})

    # Draw nodes (blue dots)
    ax.scatter(
        angles, np.ones(n),
        s=200, c="#1E88E5", zorder=5,
        edgecolors="white", linewidth=2,
    )

    # Draw edges (chord lines) for connections above threshold
    for i in range(n):
        for j in range(i + 1, n):    # upper triangle only
            weight = dmn_matrix[i, j]
            if abs(weight) > threshold:
                color = "#E53935" if weight > 0 else "#1565C0"
                alpha = min(abs(weight), 1.0) * 0.7
                ax.plot(
                    [angles[i], angles[j]],
                    [1, 1],
                    color=color,
                    alpha=alpha,
                    linewidth=abs(weight) * 4,
                )

    # Place region labels around the outside of the circle
    for angle, label in zip(angles, dmn_labels):
        rotation = np.degrees(angle)
        # Flip labels in lower half for readability
        if 90 < rotation < 270:
            rotation += 180
            ha = "right"
        else:
            ha = "left"
        ax.text(
            angle, 1.15, label,
            rotation=rotation,
            ha=ha, va="center",
            fontsize=8, fontweight="bold",
        )

    # Clean up polar plot styling
    ax.set_ylim(0, 1.3)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.grid(False)
    ax.set_title(
        "Default Mode Network Connectivity",
        fontsize=13, fontweight="bold", pad=30,
    )

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
