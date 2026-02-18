import numpy as np
import pandas as pd
import pytest

from eeg_seizure.features import FEATURE_NAMES, build_xy_groups, extract_window_features


def test_feature_extraction_sine_waves_shape_and_finite() -> None:
    sfreq = 128.0
    t = np.arange(0.0, 4.0, 1.0 / sfreq)
    ch1 = np.sin(2.0 * np.pi * 6.0 * t)
    ch2 = 0.5 * np.sin(2.0 * np.pi * 10.0 * t)
    ch3 = np.sin(2.0 * np.pi * 20.0 * t)

    win1 = np.stack([ch1, ch2, ch3], axis=0)
    win2 = np.stack([0.8 * ch1, 1.2 * ch2, 0.7 * ch3], axis=0)
    windows = np.stack([win1, win2], axis=0)

    X = extract_window_features(windows, sfreqs=sfreq)
    assert X.shape == (2, len(FEATURE_NAMES))
    assert np.isfinite(X).all()


def test_build_xy_groups_returns_expected_arrays() -> None:
    sfreq = 100.0
    t = np.arange(0.0, 4.0, 1.0 / sfreq)
    win = np.stack(
        [
            np.sin(2.0 * np.pi * 5.0 * t),
            np.sin(2.0 * np.pi * 12.0 * t),
        ],
        axis=0,
    )
    windows = np.stack([win, win], axis=0)
    meta = pd.DataFrame(
        [
            {"patient_id": "p1", "label": 1},
            {"patient_id": "p1", "label": 0},
        ]
    )

    X, y, groups = build_xy_groups(windows, meta, sfreqs=np.array([sfreq, sfreq]))
    assert X.shape == (2, len(FEATURE_NAMES))
    assert y.tolist() == [1, 0]
    assert groups.tolist() == ["p1", "p1"]
    assert np.isfinite(X).all()


def test_feature_rejects_invalid_shape() -> None:
    bad = np.zeros((2, 100))
    with pytest.raises(ValueError, match="Expected windows shape"):
        extract_window_features(bad, sfreqs=100.0)
