from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import welch

BANDS: tuple[tuple[str, float, float], ...] = (
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 40.0),
)

FEATURE_NAMES: tuple[str, ...] = (
    "rel_power_delta",
    "rel_power_theta",
    "rel_power_alpha",
    "rel_power_beta",
    "rel_power_gamma",
    "hjorth_activity",
    "hjorth_mobility",
    "hjorth_complexity",
    "spectral_entropy",
)


def _validate_windows(windows: np.ndarray) -> tuple[int, int, int]:
    if not isinstance(windows, np.ndarray):
        raise TypeError("windows must be a numpy array")
    if windows.ndim != 3:
        raise ValueError(
            "Expected windows shape (n_windows, n_channels, n_samples), "
            f"got {windows.shape}"
        )
    n_windows, n_channels, n_samples = windows.shape
    if n_windows <= 0 or n_channels <= 0 or n_samples < 8:
        raise ValueError(
            f"Invalid windows dimensions: n_windows={n_windows}, "
            f"n_channels={n_channels}, n_samples={n_samples}"
        )
    if not np.isfinite(windows).all():
        raise ValueError("windows contains non-finite values")
    return n_windows, n_channels, n_samples


def _validate_sfreqs(sfreqs: float | np.ndarray, n_windows: int) -> np.ndarray:
    if np.isscalar(sfreqs):
        sfreq_arr = np.full(n_windows, float(sfreqs), dtype=float)
    else:
        sfreq_arr = np.asarray(sfreqs, dtype=float)
    if sfreq_arr.shape != (n_windows,):
        raise ValueError(f"sfreqs must be scalar or shape ({n_windows},), got {sfreq_arr.shape}")
    if not np.isfinite(sfreq_arr).all() or np.any(sfreq_arr <= 0):
        raise ValueError("sfreqs must be finite positive values")
    return sfreq_arr


def _channel_psd(x: np.ndarray, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    nperseg = min(256, x.size)
    freqs, psd = welch(x, fs=sfreq, nperseg=nperseg)
    return freqs, np.maximum(psd, 1e-12)


def _relative_bandpower_window(window: np.ndarray, sfreq: float) -> np.ndarray:
    out = np.zeros(len(BANDS), dtype=float)
    for i, (_, f_lo, f_hi) in enumerate(BANDS):
        rel_vals: list[float] = []
        for ch in window:
            freqs, psd = _channel_psd(ch, sfreq)
            total_mask = (freqs >= 0.5) & (freqs <= 40.0)
            band_mask = (freqs >= f_lo) & (freqs < f_hi)
            total = float(np.trapezoid(psd[total_mask], freqs[total_mask]))
            band = float(np.trapezoid(psd[band_mask], freqs[band_mask]))
            rel_vals.append(band / total if total > 0 else 0.0)
        out[i] = float(np.mean(rel_vals))
    return out


def _hjorth_channel(x: np.ndarray) -> tuple[float, float, float]:
    dx = np.diff(x)
    ddx = np.diff(dx)
    var0 = float(np.var(x))
    var1 = float(np.var(dx)) if dx.size else 0.0
    var2 = float(np.var(ddx)) if ddx.size else 0.0

    activity = var0
    mobility = np.sqrt(var1 / var0) if var0 > 1e-12 else 0.0
    complexity = (
        (np.sqrt(var2 / var1) / mobility) if (var1 > 1e-12 and mobility > 1e-12) else 0.0
    )
    return activity, mobility, complexity


def _spectral_entropy_channel(x: np.ndarray, sfreq: float) -> float:
    freqs, psd = _channel_psd(x, sfreq)
    mask = (freqs >= 0.5) & (freqs <= 40.0)
    p = psd[mask]
    p = p / np.sum(p)
    h = -np.sum(p * np.log2(np.maximum(p, 1e-12)))
    h_norm = np.log2(p.size) if p.size > 1 else 1.0
    return float(h / h_norm)


def extract_window_features(windows: np.ndarray, sfreqs: float | np.ndarray) -> np.ndarray:
    """Extract channel-aggregated features per window."""
    n_windows, _, _ = _validate_windows(windows)
    sfreq_arr = _validate_sfreqs(sfreqs, n_windows)
    X = np.zeros((n_windows, len(FEATURE_NAMES)), dtype=float)

    for i in range(n_windows):
        win = windows[i]
        sf = float(sfreq_arr[i])
        rel = _relative_bandpower_window(win, sf)

        hjorth_vals = np.asarray([_hjorth_channel(ch) for ch in win], dtype=float)
        hj = hjorth_vals.mean(axis=0)

        ent = float(np.mean([_spectral_entropy_channel(ch, sf) for ch in win]))
        X[i] = np.concatenate([rel, hj, [ent]])

    if not np.isfinite(X).all():
        raise ValueError("Extracted features contain non-finite values")
    return X


def build_xy_groups(
    windows: np.ndarray,
    window_metadata: pd.DataFrame,
    sfreqs: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build X, y, groups from window tensor and per-window metadata."""
    if not isinstance(window_metadata, pd.DataFrame):
        raise TypeError("window_metadata must be a pandas DataFrame")
    needed = {"patient_id", "label"}
    missing = needed - set(window_metadata.columns)
    if missing:
        raise ValueError(f"window_metadata missing required columns: {sorted(missing)}")

    n_windows, _, _ = _validate_windows(windows)
    if len(window_metadata) != n_windows:
        raise ValueError(
            f"window_metadata rows ({len(window_metadata)}) must match n_windows ({n_windows})"
        )

    y = pd.to_numeric(window_metadata["label"], errors="raise").astype(int).to_numpy()
    bad_labels = set(np.unique(y)) - {0, 1}
    if bad_labels:
        raise ValueError(f"Labels must be binary 0/1, found {sorted(bad_labels)}")

    groups = window_metadata["patient_id"].astype(str).to_numpy()
    X = extract_window_features(windows, sfreqs=sfreqs)
    return X, y, groups
