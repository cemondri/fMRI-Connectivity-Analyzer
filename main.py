"""
Main analysis script for fMRI resting-state connectivity.

Run from the project root:
    python main.py

Output:
    figures/connectivity_matrix.png
    figures/glass_brain.png
    figures/dmn_connectivity.png
    figures/edge_distribution.png
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")   # use non-interactive backend (saves figures, no display)
import matplotlib.pyplot as plt

# Make src importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.connectivity import ConnectivityAnalyzer
from src.visualization import plot_connectivity_distribution, plot_dmn_chord


def main():
    # Make sure figures/ directory exists
    os.makedirs("figures", exist_ok=True)

    print("\n" + "=" * 60)
    print("  fMRI RESTING-STATE CONNECTIVITY ANALYSIS")
    print("=" * 60 + "\n")

    # ── 1. Initialize ─────────────────────────────────────────────────
    print("[STEP 1/6] Initializing analyzer with Harvard-Oxford atlas...")
    analyzer = ConnectivityAnalyzer(
        atlas="harvard-oxford",
        low_pass=0.1,
        high_pass=0.01,
    )

    # ── 2. Fetch data ─────────────────────────────────────────────────
    print("\n[STEP 2/6] Fetching fMRI data (this may take 3-5 minutes)...")
    analyzer.fetch_data(n_subjects=1)

    # ── 3. Extract time series ────────────────────────────────────────
    print("\n[STEP 3/6] Extracting regional time series...")
    analyzer.extract_time_series()

    # ── 4. Compute connectivity ───────────────────────────────────────
    print("\n[STEP 4/6] Computing connectivity matrix...")
    conn_matrix = analyzer.compute_connectivity(method="correlation")
    analyzer.summary(conn_matrix)

    # ── 5. Visualize ──────────────────────────────────────────────────
    print("\n[STEP 5/6] Generating figures...")

    # 5a. Connectivity matrix heatmap
    print("  → Connectivity matrix heatmap...")
    fig1 = analyzer.plot_matrix(
        conn_matrix,
        title="Resting-State Functional Connectivity (Pearson r)",
        save_path="figures/connectivity_matrix.png",
    )
    plt.close(fig1)

    # 5b. Glass brain
    print("  → Glass brain plot...")
    try:
        fig2 = analyzer.plot_glass_brain(conn_matrix, threshold=0.4)
        fig2.savefig("figures/glass_brain.png", dpi=300, bbox_inches="tight")
        plt.close(fig2)
    except Exception as e:
        print(f"  [WARN] Glass brain failed: {e}")

    # 5c. Edge distribution
    print("  → Edge weight distribution...")
    fig3 = plot_connectivity_distribution(
        conn_matrix,
        save_path="figures/edge_distribution.png",
    )
    plt.close(fig3)

    # ── 6. DMN analysis ───────────────────────────────────────────────
    print("\n[STEP 6/6] Extracting Default Mode Network...")
    dmn_matrix, dmn_labels, dmn_indices = analyzer.extract_dmn(conn_matrix)

    if dmn_matrix is not None and len(dmn_labels) > 0:
        print(f"  DMN regions found: {len(dmn_labels)}")
        for label in dmn_labels:
            print(f"    • {label}")

        fig4 = analyzer.plot_matrix(
            dmn_matrix,
            labels=dmn_labels,
            title="Default Mode Network — Internal Connectivity",
            figsize=(8, 7),
            save_path="figures/dmn_connectivity.png",
        )
        plt.close(fig4)
    else:
        print("  [WARN] No DMN regions matched.")

    # ── Done ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ANALYSIS COMPLETE")
    print("=" * 60)
    print("\n  Figures saved to: figures/")
    print("    • connectivity_matrix.png")
    print("    • glass_brain.png")
    print("    • edge_distribution.png")
    print("    • dmn_connectivity.png")
    print()


if __name__ == "__main__":
    main()
