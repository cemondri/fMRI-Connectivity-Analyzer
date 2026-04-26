"""
Functional Connectivity Analysis Engine
=========================================

This is the CORE of the entire project.

WHAT IS FUNCTIONAL CONNECTIVITY?
----------------------------------
It measures statistical dependencies between the time series of
different brain regions.  If two regions' BOLD signals rise and fall
together over time, we say they are "functionally connected."

Important: this is a STATISTICAL relationship, not a physical wire.
Two regions can be functionally connected even without a direct axonal
pathway between them (they might share a common input, for example).

Analogy: two people on opposite sides of a stadium doing the wave
at the same time.  They're "connected" in behavior, but there's no
rope between them.

THE FULL PIPELINE
------------------
1. Load a brain atlas        → divide the brain into labeled regions
2. Download fMRI data        → get resting-state scans from Nilearn
3. Extract time series       → average BOLD signal per region
4. Compute connectivity      → correlate every pair of regions
5. Statistical thresholding  → keep only significant connections
6. DMN extraction            → zoom into the Default Mode Network
7. Visualize                 → heatmaps, glass brain plots

WHAT DOES THE OUTPUT LOOK LIKE?
---------------------------------
A symmetric N × N matrix (N = number of brain regions).
Entry [i, j] = correlation between region i and region j.

    Region   |  FrontalPole  Cingulate  Angular  ...
    ---------|--------------------------------------
    FrontalPole  |    1.00       0.45     0.12
    Cingulate    |    0.45       1.00     0.67
    Angular      |    0.12       0.67     1.00
    ...
"""

import numpy as np
from nilearn import datasets, connectome
from nilearn.maskers import NiftiLabelsMasker
from sklearn.covariance import LedoitWolf
from scipy import stats

from .preprocessing import preprocess_fmri
from .visualization import plot_connectivity_matrix, plot_glass_brain


class ConnectivityAnalyzer:
    """
    Main analysis class.  Holds the entire state of an analysis session.

    WHY A CLASS INSTEAD OF STANDALONE FUNCTIONS?
    ----------------------------------------------
    Because analysis steps are sequential and interdependent:
      atlas → data → time_series → connectivity_matrix → thresholding → DMN

    Each step's output feeds the next.  A class keeps all intermediate
    results together so you don't have to pass 10 variables around.
    It also lets you go back and re-run a step with different parameters
    without starting over (e.g., try correlation then partial correlation
    on the same time series).

    Parameters
    ----------
    atlas : str
        Which brain atlas to use for parcellation.

        'harvard-oxford' (recommended for beginners):
            48 cortical + 21 subcortical = 69 regions.
            Cortical = cerebral cortex (thinking, perception, motor).
            Subcortical = below the cortex (emotion, memory, autonomic).
            This is the most widely used atlas in resting-state studies.

        'destrieux':
            148 regions.  Finer parcellation → more detail, but also
            more pairwise comparisons → more multiple-testing burden.

    low_pass : float
        Cut all frequencies ABOVE this value (Hz).
        Default 0.1 Hz → removes cardiac and respiratory noise.

    high_pass : float
        Cut all frequencies BELOW this value (Hz).
        Default 0.01 Hz → removes scanner drift.

    standardize : bool
        If True, z-score each region's time series (mean=0, std=1).
        This makes correlations comparable across regions.
    """

    SUPPORTED_ATLASES = {
        "harvard-oxford": "Harvard-Oxford cortical and subcortical atlas (69 regions)",
        "aal": "Automated Anatomical Labeling atlas",
        "destrieux": "Destrieux cortical atlas (148 regions)",
    }

    def __init__(
        self,
        atlas="harvard-oxford",
        low_pass=0.1,
        high_pass=0.01,
        standardize=True,
    ):
        if atlas not in self.SUPPORTED_ATLASES:
            raise ValueError(
                f"Unsupported atlas '{atlas}'. "
                f"Choose from: {list(self.SUPPORTED_ATLASES.keys())}"
            )

        # Store parameters
        self.atlas_name = atlas
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.standardize = standardize

        # ── State variables ──
        # These start as None and get filled as you run each step.
        # Using underscore prefix (_) signals "private" — access through
        # the public methods / properties instead.
        self._atlas_data = None           # 3D NIfTI image of the atlas
        self._labels = None               # list of region names
        self._masker = None               # NiftiLabelsMasker object
        self._fmri_data = None            # downloaded fMRI file paths
        self._confounds = None            # confound file paths
        self._time_series = None          # extracted time series matrix
        self._connectivity_matrix = None  # computed connectivity matrix

        # Load the atlas immediately upon creation
        self._load_atlas()

    # ══════════════════════════════════════════════════════════════════
    # ATLAS LOADING
    # ══════════════════════════════════════════════════════════════════

    def _load_atlas(self):
        """
        Download and configure the brain atlas.

        WHAT IS A BRAIN ATLAS?
        -----------------------
        A 3D image where every voxel (3D pixel) carries an integer label
        indicating which brain region it belongs to.

        Example (Harvard-Oxford):
          Voxel at (34, 56, 42) → label 7  → "Angular Gyrus"
          Voxel at (50, 60, 30) → label 12 → "Frontal Pole"
          Voxel at (20, 30, 25) → label 0  → "Background" (not brain)

        WHAT IS A MASKER?
        ------------------
        NiftiLabelsMasker takes a 4D fMRI image + a 3D atlas and returns
        a 2D matrix (time × regions).  For each region, it:
          1. Finds all voxels with that label
          2. Averages their signals at each time point
          → One time series per region

        Why average?  Voxels within the same region show similar activity.
        Averaging boosts signal-to-noise ratio (SNR) — the random noise
        in individual voxels cancels out, while the shared neural signal
        adds up.
        """
        if self.atlas_name == "harvard-oxford":
            # Nilearn downloads from the internet and caches locally.
            # "cort-maxprob-thr25-2mm" = cortical, maximum-probability,
            # 25% threshold, 2mm resolution.
            atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
            self._atlas_data = atlas.maps       # the 3D label image
            self._labels = atlas.labels[1:]     # skip "Background" at index 0

        elif self.atlas_name == "destrieux":
            atlas = datasets.fetch_atlas_destrieux_2009()
            self._atlas_data = atlas.maps
            # Destrieux labels may be bytes — decode to string
            self._labels = [
                label.decode() if isinstance(label, bytes) else label
                for label in atlas.labels[1:]
            ]

        # Create the masker — this object will extract regional time series.
        # It handles resampling, filtering, and confound regression internally.
        self._masker = NiftiLabelsMasker(
            labels_img=self._atlas_data,
            standardize=self.standardize,
            low_pass=self.low_pass,
            high_pass=self.high_pass,
            t_r=2.0,
            memory="nilearn_cache",   # cache to disk → faster re-runs
            verbose=0,
        )

        print(f"[INFO] Loaded {self.atlas_name} atlas with {len(self._labels)} regions.")

    # ── Properties ────────────────────────────────────────────────────

    @property
    def labels(self):
        """Return the list of region names (read-only)."""
        return self._labels

    @property
    def n_regions(self):
        """Return the number of regions in the atlas."""
        return len(self._labels) if self._labels else 0

    # ══════════════════════════════════════════════════════════════════
    # STEP 1: DOWNLOAD DATA
    # ══════════════════════════════════════════════════════════════════

    def fetch_data(self, n_subjects=1):
        """
        Download sample resting-state fMRI data from Nilearn.

        WHERE DOES THE DATA COME FROM?
        --------------------------------
        Nilearn hosts several open-access neuroimaging datasets.
        We use the "development" dataset: resting-state scans from
        healthy individuals, already preprocessed (motion-corrected,
        registered to standard space).

        DATA FORMAT: NIfTI (.nii.gz)
        -----------------------------
        A 4D matrix: (x, y, z, time)
          x, y, z = spatial dimensions (voxels in the brain)
          time    = successive scans, one every TR seconds

        Typical dimensions: 64 × 64 × 33 × 168
          → 64 × 64 × 33 = 135,168 voxels per volume
          → 168 time points × 2s TR = 336 seconds ≈ 5.6 minutes of scanning

        Parameters
        ----------
        n_subjects : int
            How many subjects to download (start with 1 — it's ~270 MB).

        Returns
        -------
        list of str : file paths to the downloaded NIfTI images
        """
        print(f"[INFO] Downloading resting-state fMRI for {n_subjects} subject(s)...")
        dataset = datasets.fetch_development_fmri(n_subjects=n_subjects)
        self._fmri_data = dataset.func        # list of fMRI file paths
        self._confounds = dataset.confounds    # list of confound file paths
        print(f"[INFO] Downloaded {len(self._fmri_data)} functional image(s).")
        return self._fmri_data

    # ══════════════════════════════════════════════════════════════════
    # STEP 2: EXTRACT TIME SERIES
    # ══════════════════════════════════════════════════════════════════

    def extract_time_series(self, fmri_img=None, confounds=None):
        """
        Extract region-averaged time series from fMRI data.

        WHAT HAPPENS HERE?
        --------------------
        Input:  4D fMRI volume  (x, y, z, time)  ~135,000 voxels × 168 time points
        Output: 2D matrix       (time, regions)   168 time points × 48 regions

        This is a MASSIVE dimensionality reduction.  Instead of tracking
        135,000 individual voxels, we summarize the brain into 48 regions.

        HOW THE MASKER WORKS (fit_transform):
        ----------------------------------------
        fit():
          - Resamples the atlas to match the fMRI resolution (if different)
          - Identifies which voxels belong to which region

        transform():
          - For each time point:
              for each region:
                  collect all voxels with that region's label
                  compute their mean signal
          - Applies confound regression (remove motion, WM, CSF)
          - Applies bandpass filter (keep 0.01–0.1 Hz)
          - Z-scores each region

        Result: a clean (n_timepoints × n_regions) matrix.
        Each column is one brain region's "activity timeline."

        Parameters
        ----------
        fmri_img : str or Nifti1Image, optional
            Path to fMRI image.  If None, uses the first downloaded image.
        confounds : str or array, optional
            Confound regressors.  If None, uses the matching confounds file.

        Returns
        -------
        numpy.ndarray, shape (n_timepoints, n_regions)
        """
        if fmri_img is None:
            if self._fmri_data is None:
                raise RuntimeError("No data loaded.  Call fetch_data() first.")
            fmri_img = self._fmri_data[0]
            confounds = self._confounds[0] if self._confounds else None

        print("[INFO] Extracting regional time series...")
        self._time_series = self._masker.fit_transform(fmri_img, confounds=confounds)

        n_timepoints, n_regions = self._time_series.shape
        print(f"[INFO] Extracted {n_timepoints} time points × {n_regions} regions.")
        return self._time_series

    # ══════════════════════════════════════════════════════════════════
    # STEP 3: COMPUTE CONNECTIVITY MATRIX
    # ══════════════════════════════════════════════════════════════════

    def compute_connectivity(self, method="correlation", time_series=None):
        """
        Compute the functional connectivity matrix.

        THIS IS THE MOST IMPORTANT STEP IN THE WHOLE PROJECT.

        We take two brain regions' time series and ask: "Do they go up
        and down together?"  If yes → they are functionally connected.

        THREE METHODS AVAILABLE:
        =========================

        1. PEARSON CORRELATION  (method="correlation")
        ───────────────────────────────────────────────
        The simplest and most common approach.

        Formula:  r = Σ(x_i - x̄)(y_i - ȳ) / √[Σ(x_i - x̄)² × Σ(y_i - ȳ)²]

        Interpretation:
          r = +1.0 → perfect positive correlation (rise together)
          r =  0.0 → no linear relationship
          r = -1.0 → perfect negative correlation (one rises, other falls)

        In resting-state fMRI, typical values:
          r > 0.5  → strong connection (same network)
          r = 0.2–0.5 → moderate connection
          r < 0.2  → weak or no connection
          r < 0    → anti-correlation (competing networks)

        PROBLEM: indirect connections.
        If A → B → C, then A and C will also show correlation — but they
        might not be DIRECTLY connected.  Pearson can't tell the difference.

        2. PARTIAL CORRELATION  (method="partial_correlation")
        ─────────────────────────────────────────────────────
        Removes indirect connections.  Shows only DIRECT relationships.

        How?  When computing the correlation between regions A and C,
        it "controls for" ALL other regions (B, D, E, ...).
        If A and C are only correlated because of B, partial correlation
        removes B's influence and the A–C connection disappears.

        Technical implementation:
          Step 1: Estimate the covariance matrix of all regions
          Step 2: Invert it → this gives the "precision matrix"
          Step 3: Normalize the precision matrix → partial correlations

        We use the Ledoit-Wolf estimator for Step 1 because:
          - Standard covariance estimation fails when n_regions > n_timepoints
            (the matrix becomes "singular" — can't be inverted)
          - Ledoit-Wolf applies "shrinkage": it blends the sample covariance
            with a structured estimate (identity matrix), producing a stable
            invertible matrix even with limited data

        Result: a sparser matrix — fewer connections, but more trustworthy.

        3. COVARIANCE  (method="covariance")
        ────────────────────────────────────
        Raw covariance (not normalized).  Values depend on the scale of
        the signals, so they're harder to interpret.  Rarely used for
        visualization — correlation is preferred because it's bounded [-1, 1].

        Parameters
        ----------
        method : str
            "correlation", "partial_correlation", or "covariance"
        time_series : numpy.ndarray, optional
            If None, uses the time series from extract_time_series().

        Returns
        -------
        numpy.ndarray, shape (n_regions, n_regions) — symmetric matrix
        """
        if time_series is None:
            if self._time_series is None:
                raise RuntimeError("No time series.  Call extract_time_series() first.")
            time_series = self._time_series

        valid_methods = ["correlation", "partial_correlation", "covariance"]
        if method not in valid_methods:
            raise ValueError(f"Unknown method '{method}'.  Choose from: {valid_methods}")

        print(f"[INFO] Computing connectivity using '{method}'...")

        if method == "correlation":
            # np.corrcoef computes Pearson r for all row pairs.
            # We transpose (.T) because corrcoef treats ROWS as variables,
            # but our variables (regions) are in COLUMNS.
            self._connectivity_matrix = np.corrcoef(time_series.T)

        elif method == "partial_correlation":
            # ── Step 1: Ledoit-Wolf covariance estimation ──
            estimator = LedoitWolf()
            estimator.fit(time_series)

            # ── Step 2: Get the precision matrix (inverse covariance) ──
            # The precision matrix Θ has a useful property:
            #   Θ[i,j] = 0  ↔  regions i and j are conditionally independent
            #                   (given all other regions)
            precision = estimator.precision_

            # ── Step 3: Convert precision → partial correlation ──
            # Formula: ρ_ij = -Θ_ij / √(Θ_ii × Θ_jj)
            # The negative sign is because the precision matrix encodes
            # the INVERSE relationship.
            diag = np.sqrt(np.diag(precision))
            partial_corr = -precision / np.outer(diag, diag)
            np.fill_diagonal(partial_corr, 1.0)   # self-correlation = 1
            self._connectivity_matrix = partial_corr

        elif method == "covariance":
            self._connectivity_matrix = np.cov(time_series.T)

        print(f"[INFO] Connectivity matrix shape: {self._connectivity_matrix.shape}")
        return self._connectivity_matrix

    # ══════════════════════════════════════════════════════════════════
    # STEP 4: STATISTICAL THRESHOLDING
    # ══════════════════════════════════════════════════════════════════

    def threshold_connections(self, matrix=None, method="fdr", alpha=0.05):
        """
        Apply statistical thresholding to the connectivity matrix.

        WHY DO WE NEED THRESHOLDING?
        ------------------------------
        With 48 regions, there are 48 × 47 / 2 = 1,128 pairwise connections.
        Some of these correlations will appear "significant" purely by
        chance — this is the multiple comparisons problem.

        If you test 1,128 connections at α = 0.05, you'd expect
        ~56 false positives even if there were NO real connections at all.

        THREE THRESHOLDING METHODS:
        ============================

        1. FDR — False Discovery Rate (Benjamini-Hochberg)
        ──────────────────────────────────────────────────
        Controls the PROPORTION of false positives among discoveries.
        "Of the connections I declare significant, at most 5% are wrong."

        This is the recommended method — it's a good balance between
        being too strict (missing real connections) and too lenient
        (reporting false connections).

        How it works:
          a) Compute a p-value for each connection
          b) Sort all p-values from smallest to largest
          c) For the i-th smallest p-value, the threshold is: i/m × α
             (m = total number of tests)
          d) Find the largest p-value that's still below its threshold
          e) All connections with p-values ≤ that cutoff are significant

        2. BONFERRONI
        ─────────────
        The strictest method.  Divides α by the number of tests.
        Threshold = 0.05 / 1128 = 0.0000443

        Very conservative — few false positives, but misses many
        real connections (low statistical power).

        3. PERCENTILE
        ─────────────
        Not a statistical test — just keeps the top X% strongest
        connections.  Useful for exploratory analysis when you want
        to see the network backbone regardless of significance.

        HOW DO WE GET P-VALUES FROM CORRELATIONS?
        -------------------------------------------
        We use the Fisher z-transform:
          z = arctanh(r) × √(n - 3)

        This converts a correlation coefficient r into a z-score that
        follows a standard normal distribution (under H₀: r = 0).
        Then we compute a two-tailed p-value from the z-score.

        Parameters
        ----------
        matrix : numpy.ndarray, optional
        method : str — "fdr", "bonferroni", or "percentile"
        alpha  : float — significance level (default 0.05)

        Returns
        -------
        numpy.ndarray — thresholded matrix (non-significant entries = 0)
        """
        if matrix is None:
            matrix = self._connectivity_matrix
        if matrix is None:
            raise RuntimeError("No connectivity matrix.  Call compute_connectivity() first.")

        n = matrix.shape[0]
        thresholded = matrix.copy()

        if method in ("fdr", "bonferroni"):
            # ── Fisher z-transform ──
            # arctanh(r) transforms r from [-1, 1] to [-∞, +∞]
            # Multiplying by √(n-3) standardizes it so that under H₀ (r=0),
            # z follows N(0, 1).
            n_timepoints = (
                self._time_series.shape[0] if self._time_series is not None else 100
            )
            z_scores = np.arctanh(matrix) * np.sqrt(n_timepoints - 3)

            # Two-tailed p-value: P(|Z| > |observed z|)
            # We use |z| because we care about both positive AND negative correlations
            p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
            np.fill_diagonal(p_values, 1.0)   # don't test self-correlation

            if method == "fdr":
                # ── Benjamini-Hochberg procedure ──
                # Only look at upper triangle (matrix is symmetric)
                p_flat = p_values[np.triu_indices(n, k=1)]
                n_tests = len(p_flat)

                # Sort p-values ascending
                sorted_idx = np.argsort(p_flat)
                sorted_p = p_flat[sorted_idx]

                # BH thresholds: (rank / total_tests) × α
                thresholds = alpha * np.arange(1, n_tests + 1) / n_tests

                # Find the largest p-value that passes its threshold
                below = np.where(sorted_p <= thresholds)[0]
                p_threshold = sorted_p[below[-1]] if len(below) > 0 else 0

                mask = p_values > p_threshold

            else:   # bonferroni
                # Simple: divide α by number of tests
                mask = p_values > (alpha / (n * (n - 1) / 2))

            thresholded[mask] = 0
            np.fill_diagonal(thresholded, 0)

        elif method == "percentile":
            # Keep only connections in the top (alpha × 100)%
            cutoff = np.percentile(
                np.abs(matrix[np.triu_indices(n, k=1)]),
                (1 - alpha) * 100,
            )
            thresholded[np.abs(matrix) < cutoff] = 0

        n_edges = np.count_nonzero(thresholded[np.triu_indices(n, k=1)])
        total_edges = n * (n - 1) // 2
        print(
            f"[INFO] Thresholding: {n_edges}/{total_edges} edges survived "
            f"({method}, α={alpha})."
        )
        return thresholded

    # ══════════════════════════════════════════════════════════════════
    # STEP 5: DEFAULT MODE NETWORK (DMN) EXTRACTION
    # ══════════════════════════════════════════════════════════════════

    def extract_dmn(self, matrix=None):
        """
        Extract the Default Mode Network submatrix.

        WHAT IS THE DEFAULT MODE NETWORK?
        -----------------------------------
        The DMN is a set of brain regions that are MOST ACTIVE when you're
        NOT doing any specific task — when you're daydreaming, mind-wandering,
        remembering the past, or thinking about yourself.

        It was discovered by accident: researchers noticed that certain
        brain regions consistently DECREASED activity during tasks and
        INCREASED during rest.  This "default" activity pattern is now
        one of the most studied networks in neuroscience.

        DMN regions:
        • Posterior Cingulate Cortex (PCC) / Precuneus
            → The "hub" of the DMN.  Most connected region.
            → Involved in self-awareness and consciousness.
        • Medial Prefrontal Cortex (mPFC)
            → Self-referential thought ("what do I think about this?")
        • Angular Gyrus
            → Semantic memory, language comprehension
        • Temporal Pole
            → Autobiographical memory, social cognition
        • Parahippocampal Gyrus
            → Spatial memory, scene reconstruction
        • Hippocampus
            → Episodic memory (remembering specific events)

        WHY IS THE DMN CLINICALLY IMPORTANT?
        --------------------------------------
        • Alzheimer's disease: DMN connectivity breaks down EARLY — even
          before memory symptoms appear.  → potential early biomarker
        • Depression: DMN is OVERACTIVE → rumination (repetitive negative
          thinking).  The brain can't "turn off" self-focused thought.
        • Schizophrenia: abnormal DMN boundaries → blurred self/other
        • ADHD: DMN doesn't deactivate properly during tasks → distraction

        METHOD USED HERE:
        ------------------
        We identify DMN regions by matching keywords in the atlas labels.
        This is simple but effective for Harvard-Oxford atlas.  More
        sophisticated approaches use ICA (Independent Component Analysis)
        or seed-based analysis with a known DMN coordinate.

        Returns
        -------
        tuple: (dmn_matrix, dmn_labels, dmn_indices)
            dmn_matrix  — N_dmn × N_dmn connectivity submatrix
            dmn_labels  — list of DMN region names
            dmn_indices — their indices in the full matrix
        """
        if matrix is None:
            matrix = self._connectivity_matrix
        if matrix is None:
            raise RuntimeError("No connectivity matrix available.")

        # Keywords that identify DMN regions in Harvard-Oxford labels
        dmn_keywords = [
            "cingulate",          # Posterior Cingulate Cortex
            "precuneous",         # Precuneus (note: HO atlas spells it this way)
            "angular",            # Angular Gyrus
            "frontal medial",     # Medial Prefrontal Cortex
            "temporal pole",      # Temporal Pole
            "parahippocampal",    # Parahippocampal Gyrus
            "hippocampus",        # Hippocampus
        ]

        dmn_indices = []
        dmn_labels = []
        for i, label in enumerate(self._labels):
            if any(kw in label.lower() for kw in dmn_keywords):
                if i < matrix.shape[0]:   # don't exceed matrix bounds
                    dmn_indices.append(i)
                    dmn_labels.append(label)

        if not dmn_indices:
            print("[WARN] No DMN regions found with current atlas.")
            return None, [], []

        # Extract the submatrix — np.ix_ creates an open mesh of indices
        # so we can select both rows AND columns simultaneously.
        # Example: indices = [3, 7, 12] → extracts a 3×3 submatrix
        dmn_matrix = matrix[np.ix_(dmn_indices, dmn_indices)]

        print(f"[INFO] Extracted DMN subnetwork: {len(dmn_indices)} regions.")
        return dmn_matrix, dmn_labels, dmn_indices

    # ══════════════════════════════════════════════════════════════════
    # VISUALIZATION SHORTCUTS
    # ══════════════════════════════════════════════════════════════════

    def plot_matrix(self, matrix=None, threshold=None, labels=None, **kwargs):
        """Plot the connectivity matrix as a heatmap."""
        if matrix is None:
            matrix = self._connectivity_matrix
        if labels is None:
            labels = self._labels
        return plot_connectivity_matrix(
            matrix, labels=labels, threshold=threshold, **kwargs
        )

    def plot_glass_brain(self, matrix=None, threshold=0.3, **kwargs):
        """Plot connections on a transparent glass brain."""
        if matrix is None:
            matrix = self._connectivity_matrix
        coords = self._get_region_coords()
        return plot_glass_brain(
            matrix, coords=coords, threshold=threshold, **kwargs
        )

    def _get_region_coords(self):
        """
        Get MNI coordinates for each atlas region.

        WHAT IS MNI SPACE?
        --------------------
        Montreal Neurological Institute standard coordinate system.
        All brains are warped to match a template so that coordinate
        (x, y, z) refers to the same anatomical location across people.

        Convention:
          x: right (+) / left (-)
          y: anterior (+) / posterior (-)    (front / back)
          z: superior (+) / inferior (-)    (top / bottom)

        Example: PCC is approximately at (0, -53, 26)
        """
        from nilearn.plotting import find_parcellation_cut_coords
        return find_parcellation_cut_coords(self._atlas_data)

    # ══════════════════════════════════════════════════════════════════
    # SUMMARY STATISTICS
    # ══════════════════════════════════════════════════════════════════

    def summary(self, matrix=None):
        """
        Print summary statistics of the connectivity matrix.

        WHAT TO LOOK FOR:
        ------------------
        • Mean correlation: typically 0.1–0.3 for resting-state.
          If it's very high (> 0.5), suspect motion artifacts or
          insufficient cleaning.

        • |r| > 0.3 count: moderate-to-strong connections.
          These are the ones worth investigating.

        • |r| > 0.5 count: strong connections — likely within the
          same functional network (e.g., both in DMN, or both in
          the visual network).
        """
        if matrix is None:
            matrix = self._connectivity_matrix
        if matrix is None:
            print("[WARN] No connectivity matrix computed yet.")
            return

        # Extract upper triangle (matrix is symmetric; lower triangle
        # is a mirror, and diagonal is self-correlation = 1.0)
        # k=1 means exclude the diagonal
        upper = matrix[np.triu_indices_from(matrix, k=1)]

        print("\n" + "=" * 50)
        print("  CONNECTIVITY SUMMARY")
        print("=" * 50)
        print(f"  Regions        : {matrix.shape[0]}")
        print(f"  Total edges    : {len(upper)}")
        print(f"  Mean corr.     : {np.mean(upper):.4f}")
        print(f"  Median corr.   : {np.median(upper):.4f}")
        print(f"  Std dev.       : {np.std(upper):.4f}")
        print(f"  Max corr.      : {np.max(upper):.4f}")
        print(f"  Min corr.      : {np.min(upper):.4f}")
        print(f"  |r| > 0.3      : {np.sum(np.abs(upper) > 0.3)}")
        print(f"  |r| > 0.5      : {np.sum(np.abs(upper) > 0.5)}")
        print("=" * 50 + "\n")
