"""
Unit tests for the connectivity analysis engine.

Tests core functionality with synthetic data so they run fast
and don't require downloading real fMRI data.

Run with:
    python -m unittest tests.test_connectivity
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestConnectivityMatrix(unittest.TestCase):
    """Test connectivity computations on synthetic time series."""

    def setUp(self):
        """Create deterministic synthetic data."""
        np.random.seed(42)
        self.n_timepoints = 200
        self.n_regions = 10

        # Generate time series with a shared component (will create correlations)
        base_signal = np.random.randn(self.n_timepoints)
        self.time_series = np.column_stack([
            base_signal + np.random.randn(self.n_timepoints) * noise_level
            for noise_level in np.linspace(0.1, 2.0, self.n_regions)
        ])

    def test_correlation_matrix_shape(self):
        """Output should be square: (n_regions, n_regions)."""
        matrix = np.corrcoef(self.time_series.T)
        self.assertEqual(matrix.shape, (self.n_regions, self.n_regions))

    def test_correlation_matrix_symmetry(self):
        """Connectivity matrix must be symmetric."""
        matrix = np.corrcoef(self.time_series.T)
        np.testing.assert_array_almost_equal(matrix, matrix.T)

    def test_correlation_diagonal_is_one(self):
        """Self-correlation must equal 1.0 on the diagonal."""
        matrix = np.corrcoef(self.time_series.T)
        np.testing.assert_array_almost_equal(np.diag(matrix), np.ones(self.n_regions))

    def test_correlation_values_in_range(self):
        """All correlations must be in [-1, 1]."""
        matrix = np.corrcoef(self.time_series.T)
        self.assertTrue(np.all(matrix >= -1.0))
        self.assertTrue(np.all(matrix <= 1.0))

    def test_highly_correlated_regions(self):
        """Regions with shared signal should correlate highly."""
        matrix = np.corrcoef(self.time_series.T)
        # First two regions have lowest noise → strongest shared signal
        self.assertGreater(matrix[0, 1], 0.5)

    def test_thresholding_zeros_weak_connections(self):
        """Thresholding should zero out connections below cutoff."""
        matrix = np.corrcoef(self.time_series.T)
        threshold = 0.5
        thresholded = matrix.copy()
        thresholded[np.abs(thresholded) < threshold] = 0

        weak_mask = np.abs(matrix) < threshold
        self.assertTrue(np.all(thresholded[weak_mask] == 0))


class TestDMNExtraction(unittest.TestCase):
    """Test Default Mode Network keyword matching."""

    def test_dmn_keyword_matching(self):
        """DMN keywords should correctly identify expected regions."""
        labels = [
            "Frontal Pole",
            "Cingulate Gyrus, anterior",
            "Precuneous Cortex",
            "Angular Gyrus",
            "Precentral Gyrus",
            "Superior Temporal Gyrus",
            "Parahippocampal Gyrus",
        ]

        dmn_keywords = [
            "cingulate", "precuneous", "angular",
            "frontal medial", "parahippocampal",
        ]

        dmn_indices = [
            i for i, label in enumerate(labels)
            if any(kw in label.lower() for kw in dmn_keywords)
        ]

        # Should match: Cingulate (1), Precuneous (2), Angular (3), Parahippocampal (6)
        self.assertIn(1, dmn_indices)
        self.assertIn(2, dmn_indices)
        self.assertIn(3, dmn_indices)
        self.assertIn(6, dmn_indices)
        # Should NOT match: Frontal Pole, Precentral, Temporal
        self.assertNotIn(0, dmn_indices)
        self.assertNotIn(4, dmn_indices)

    def test_submatrix_extraction(self):
        """np.ix_ should correctly extract a square submatrix."""
        n = 10
        matrix = np.random.rand(n, n)
        indices = [1, 3, 5]

        submatrix = matrix[np.ix_(indices, indices)]
        self.assertEqual(submatrix.shape, (3, 3))


class TestPreprocessing(unittest.TestCase):
    """Test signal preprocessing utilities."""

    def test_bandpass_filter_preserves_shape(self):
        """Filtered output should have the same shape as input."""
        from scipy.signal import butter, filtfilt

        signal = np.random.randn(200, 5)
        t_r = 2.0
        nyquist = 1 / (2 * t_r)
        b, a = butter(5, [0.01 / nyquist, 0.1 / nyquist], btype="band")
        filtered = filtfilt(b, a, signal, axis=0)

        self.assertEqual(filtered.shape, signal.shape)

    def test_zscore_normalization(self):
        """Z-scored signal should have mean ≈ 0 and std ≈ 1."""
        signal = np.random.randn(200, 5) * 10 + 50
        z_scored = (signal - signal.mean(axis=0)) / signal.std(axis=0)

        np.testing.assert_array_almost_equal(z_scored.mean(axis=0), np.zeros(5), decimal=10)
        np.testing.assert_array_almost_equal(z_scored.std(axis=0), np.ones(5), decimal=10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
