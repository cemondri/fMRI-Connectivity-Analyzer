"""
fMRI Resting-State Connectivity Analyzer
=========================================

This package provides tools for analyzing resting-state fMRI data:
  - connectivity.py  : Core analysis engine (atlas loading, time series
                        extraction, connectivity matrix computation,
                        thresholding, and DMN extraction)
  - preprocessing.py : Signal cleaning utilities (confound regression,
                        bandpass filtering, motion metrics)
  - visualization.py : Plotting functions (heatmaps, glass brain,
                        distribution plots, chord diagrams)

Typical usage
-------------
    from src.connectivity import ConnectivityAnalyzer

    analyzer = ConnectivityAnalyzer(atlas="harvard-oxford")
    analyzer.fetch_data(n_subjects=1)
    analyzer.extract_time_series()
    conn = analyzer.compute_connectivity(method="correlation")
    analyzer.plot_matrix(conn, threshold=0.3)
"""

from .connectivity import ConnectivityAnalyzer
from .preprocessing import preprocess_fmri
from .visualization import plot_connectivity_matrix, plot_glass_brain

__version__ = "1.0.0"
__all__ = [
    "ConnectivityAnalyzer",
    "preprocess_fmri",
    "plot_connectivity_matrix",
    "plot_glass_brain",
]
