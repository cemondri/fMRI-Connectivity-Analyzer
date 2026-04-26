"""
fMRI Preprocessing Utilities
=============================
 
WHY DO WE NEED PREPROCESSING?
------------------------------
Raw fMRI signals are noisy.  The scanner records everything happening
inside the head — not just neural activity but also:
 
  * Head motion artifacts    — even 0.5 mm of movement creates fake
                                correlations between distant regions
  * Cardiac pulsation        — heart beating ~1 Hz pushes blood vessels
  * Respiratory fluctuations — breathing ~0.3 Hz shifts the whole brain
  * Scanner drift             — slow signal increase over time (hardware)
  * White matter / CSF signal — non-neural tissue that still has signal
 
This module removes those contaminants so that only the neural signal
of interest (the BOLD response, 0.01–0.1 Hz) survives.
 
KEY CONCEPT: BOLD Signal
-------------------------
fMRI doesn't measure neural activity directly.  It measures Blood-Oxygen-
Level-Dependent (BOLD) contrast: when neurons fire, local blood flow
increases to deliver oxygen.  Oxygenated and deoxygenated hemoglobin
have different magnetic properties → the scanner picks up this difference.
The BOLD signal peaks ~5 seconds AFTER neural activity (hemodynamic delay).
"""
 
import numpy as np
from nilearn import signal
 
 
# ═══════════════════════════════════════════════════════════════════════
# 1. MAIN CLEANING FUNCTION
# ═══════════════════════════════════════════════════════════════════════
 
def preprocess_fmri(
    time_series,           # shape: (n_timepoints, n_regions)
    confounds=None,        # shape: (n_timepoints, n_confounds) or None
    low_pass=0.1,          # Hz — cut everything ABOVE this frequency
    high_pass=0.01,        # Hz — cut everything BELOW this frequency
    t_r=2.0,               # repetition time in seconds (time between scans)
    detrend=True,          # remove slow linear drift?
    standardize=True,      # z-score each region's signal?
):
    """
    Clean raw fMRI time series in four steps.
 
    Processing pipeline (applied in this order internally):
    -------------------------------------------------------
    1. DETREND
       Remove the slow linear trend from the signal.
       Why?  The scanner's magnetic field drifts slightly over time,
       causing a gradual upward or downward slope in the signal.
       This has nothing to do with the brain — remove it.
 
    2. CONFOUND REGRESSION
       Regress out known nuisance variables (head motion parameters,
       white matter signal, CSF signal).  This is a linear regression:
       signal_clean = signal - (confounds × beta_coefficients)
       The 6 standard motion confounds are:
         trans_x, trans_y, trans_z  (translation in mm)
         rot_x, rot_y, rot_z       (rotation in radians)
 
    3. BANDPASS FILTERING
       Keep only frequencies between high_pass and low_pass.
       Why 0.01–0.1 Hz specifically?
         - Below 0.01 Hz: scanner drift, very slow physiology (not neural)
         - 0.01–0.1 Hz:   resting-state brain networks oscillate here ✓
         - Above 0.1 Hz:  cardiac (~1 Hz), respiratory (~0.3 Hz) noise
       Think of it like a radio — we tune in to the "brain network" band.
 
    4. STANDARDIZE (Z-SCORE)
       For each region: subtract the mean, divide by std deviation.
       Result: mean = 0, std = 1 for every region.
       Why?  Different brain regions have different baseline signal levels
       (e.g., occipital cortex naturally has higher BOLD than white matter).
       Z-scoring makes them comparable so that correlations are meaningful.
 
    Parameters
    ----------
    time_series : numpy.ndarray, shape (n_timepoints, n_regions)
        Each column is one brain region's raw BOLD time course.
        Example: 168 time points × 48 regions → shape (168, 48)
 
    confounds : numpy.ndarray or None
        Matrix of nuisance regressors.  Nilearn provides these
        automatically when you download sample datasets.
 
    Returns
    -------
    numpy.ndarray : same shape as input, but cleaned
    """
 
    # Nilearn's signal.clean() does all four steps in the right order.
    # Under the hood it uses scipy.signal for filtering and numpy for
    # the linear algebra of confound regression.
    cleaned = signal.clean(
        time_series,
        confounds=confounds,
        low_pass=low_pass,
        high_pass=high_pass,
        t_r=t_r,
        detrend=detrend,
        standardize=standardize,
    )
    return cleaned
 
 
# ═══════════════════════════════════════════════════════════════════════
# 2. MOTION QUALITY CONTROL
# ═══════════════════════════════════════════════════════════════════════
 
def compute_motion_metrics(confounds_df):
    """
    Compute Framewise Displacement (FD) — a single number that tells you
    how much the head moved between consecutive scans.
 
    WHAT IS FRAMEWISE DISPLACEMENT?
    --------------------------------
    Imagine drawing a dot on the subject's nose.  FD measures how far
    that dot moved (in mm) from one scan to the next.  It combines all
    6 motion parameters (3 translations + 3 rotations) into one number.
 
    WHY DOES IT MATTER?
    --------------------
    Head motion is the #1 enemy of functional connectivity analysis.
    When the head moves, ALL voxels shift position simultaneously.
    This creates spurious correlations between regions that have nothing
    to do with neural synchronization.
 
    Classic example:
      - Subject nods head → all posterior regions shift DOWN together
      - This looks like "connectivity" between those regions, but it's fake
 
    Rule of thumb:
      FD < 0.2 mm  → excellent data quality
      FD < 0.5 mm  → acceptable
      FD > 0.5 mm  → flag this time point for scrubbing (removal)
      Mean FD > 0.5 → consider excluding this subject entirely
 
    WHAT IS SCRUBBING?
    -------------------
    Removing high-motion time points from the analysis entirely.
    If 10% of your data has FD > 0.5 mm, you throw away that 10%.
    Better to lose some data than to have motion artifacts contaminating
    your connectivity estimates.
 
    Parameters
    ----------
    confounds_df : pandas.DataFrame or str (file path)
        Confounds table from Nilearn or fMRIPrep.
        Must contain a column named 'framewise_displacement' (or 'FD').
 
    Returns
    -------
    dict with keys:
        framewise_displacement : array of FD values per time point
        mean_fd                : average FD (lower = better)
        max_fd                 : worst single movement
        high_motion_volumes    : indices of bad time points
        percent_scrubbed       : % of data that would be removed
    """
    import pandas as pd
 
    # Accept either a DataFrame or a path to a TSV file
    if isinstance(confounds_df, str):
        confounds_df = pd.read_csv(confounds_df, sep="\t")
 
    # Different software packages use different column names for FD
    fd_col = None
    for candidate in ["framewise_displacement", "FramewiseDisplacement", "FD"]:
        if candidate in confounds_df.columns:
            fd_col = candidate
            break
 
    if fd_col is None:
        raise ValueError(
            "Could not find framewise displacement column.  "
            "Expected one of: framewise_displacement, FramewiseDisplacement, FD"
        )
 
    fd = confounds_df[fd_col].values
 
    # The first time point has no "previous" scan → FD is undefined (NaN).
    # Replace NaN with 0.0 (no movement assumed for the first volume).
    fd = np.nan_to_num(fd, nan=0.0)
 
    threshold = 0.5  # mm — standard threshold in the literature
    high_motion = np.where(fd > threshold)[0]
 
    return {
        "framewise_displacement": fd,
        "mean_fd": np.mean(fd),
        "max_fd": np.max(fd),
        "high_motion_volumes": high_motion,
        "percent_scrubbed": 100 * len(high_motion) / len(fd),
    }
 
 
# ═══════════════════════════════════════════════════════════════════════
# 3. MANUAL BANDPASS FILTER (for educational purposes)
# ═══════════════════════════════════════════════════════════════════════
 
def bandpass_filter(time_series, low_freq=0.01, high_freq=0.1, t_r=2.0):
    """
    Apply a Butterworth bandpass filter to fMRI time series.
 
    NOTE: In practice, you'd use preprocess_fmri() above, which calls
    Nilearn's built-in filtering.  This function exists to show HOW
    bandpass filtering works under the hood.
 
    WHAT IS A BUTTERWORTH FILTER?
    ------------------------------
    A type of signal filter designed to have a maximally flat frequency
    response in the passband.  "Flat" means it doesn't amplify or
    attenuate frequencies within the desired range — it just lets them
    through unchanged.
 
    WHY 5th ORDER?
    ---------------
    Higher order = sharper cutoff (more aggressive at removing unwanted
    frequencies).  5th order is a common choice that balances sharpness
    with stability (very high orders can cause numerical issues).
 
    WHY filtfilt() INSTEAD OF lfilter()?
    --------------------------------------
    lfilter() applies the filter in one direction → introduces a time
    delay (phase shift).  Your peaks would be shifted in time!
    filtfilt() filters forward AND backward → the phase shifts cancel
    out.  Result: zero phase distortion, no timing artifacts.
 
    NYQUIST THEOREM
    ----------------
    You can only detect frequencies up to half your sampling rate.
    With TR = 2.0s, sampling rate = 0.5 Hz, so Nyquist = 0.25 Hz.
    This means we CANNOT detect anything above 0.25 Hz — which is fine,
    because our upper cutoff (0.1 Hz) is well below Nyquist.
 
    Filter frequencies must be normalized to [0, 1] relative to Nyquist:
      0.01 Hz / 0.25 Hz = 0.04  (normalized low cutoff)
      0.10 Hz / 0.25 Hz = 0.40  (normalized high cutoff)
 
    Parameters
    ----------
    time_series : numpy.ndarray, shape (n_timepoints, n_regions)
    low_freq    : float, lower bound in Hz (default 0.01)
    high_freq   : float, upper bound in Hz (default 0.1)
    t_r         : float, repetition time in seconds
 
    Returns
    -------
    numpy.ndarray : filtered time series, same shape
    """
    from scipy.signal import butter, filtfilt
 
    # Step 1: compute Nyquist frequency
    nyquist = 1 / (2 * t_r)        # 0.25 Hz when TR = 2.0s
 
    # Step 2: normalize cutoff frequencies to [0, 1] range
    low = low_freq / nyquist        # 0.01 / 0.25 = 0.04
    high = high_freq / nyquist      # 0.10 / 0.25 = 0.40
 
    # Safety clamp to valid range
    low = max(low, 0.001)
    high = min(high, 0.999)
 
    # Step 3: design the Butterworth filter
    # b, a = filter coefficients (numerator, denominator of transfer function)
    b, a = butter(N=5, Wn=[low, high], btype="band")
 
    # Step 4: apply zero-phase filtering (forward + backward)
    # axis=0 means filter each column (region) independently
    filtered = filtfilt(b, a, time_series, axis=0)
 
    return filtered
