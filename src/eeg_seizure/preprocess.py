from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from eeg_seizure.config import PreprocessConfig
from eeg_seizure.data import validate_layout


@dataclass(slots=True)
class WindowedRecording:
    windows: np.ndarray
    metadata: pd.DataFrame
    sfreq: float


def _validate_cfg(cfg: PreprocessConfig) -> None:
    if cfg.window_sec <= 0:
        raise ValueError("window_sec must be > 0")
    if not 0 <= cfg.overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    if cfg.l_freq <= 0 or cfg.h_freq <= 0:
        raise ValueError("l_freq and h_freq must be > 0")
    if cfg.l_freq >= cfg.h_freq:
        raise ValueError("l_freq must be smaller than h_freq")


def preprocess_recording(
    edf_path: str | Path,
    patient_id: str,
    recording_id: str,
    label: int,
    cfg: PreprocessConfig | None = None,
) -> WindowedRecording:
    """Read EDF, apply filtering/re-referencing, and segment into windows."""
    cfg = cfg or PreprocessConfig()
    _validate_cfg(cfg)

    edf_path = Path(edf_path)
    if not edf_path.exists():
        raise FileNotFoundError(f"EDF file not found: {edf_path}")

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    nyquist = sfreq / 2.0

    if cfg.notch_freq < nyquist:
        raw.notch_filter(freqs=[cfg.notch_freq], verbose="ERROR")

    h_freq = min(cfg.h_freq, nyquist - 1e-3)
    if h_freq <= cfg.l_freq:
        raise ValueError(
            f"Invalid effective bandpass for {edf_path}: l_freq={cfg.l_freq}, h_freq={h_freq}, sfreq={sfreq}"
        )
    raw.filter(l_freq=cfg.l_freq, h_freq=h_freq, verbose="ERROR")

    try:
        raw.set_eeg_reference("average", verbose="ERROR")
    except Exception:
        pass

    data = raw.get_data()
    if data.ndim != 2:
        raise ValueError(f"Expected shape (n_channels, n_samples), got {data.shape}")

    n_channels, n_samples = data.shape
    win = int(round(cfg.window_sec * sfreq))
    step = int(round(win * (1.0 - cfg.overlap)))
    if win <= 0 or step <= 0:
        raise ValueError(f"Invalid window params in samples: win={win}, step={step}")
    if n_samples < win:
        raise ValueError(
            f"Recording too short: {edf_path} has {n_samples} samples, requires at least {win}"
        )

    windows: list[np.ndarray] = []
    rows = []
    idx = 0
    for start in range(0, n_samples - win + 1, step):
        end = start + win
        windows.append(data[:, start:end])
        rows.append(
            {
                "patient_id": patient_id,
                "recording_id": recording_id,
                "window_index": idx,
                "label": int(label),
            }
        )
        idx += 1

    if not windows:
        raise ValueError(f"No windows produced for {edf_path}")

    return WindowedRecording(
        windows=np.asarray(windows, dtype=float),
        metadata=pd.DataFrame(rows),
        sfreq=sfreq,
    )


def preprocess_dataset(
    data_dir: str | Path,
    metadata_df: pd.DataFrame,
    cfg: PreprocessConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Preprocess all recordings and return window-level arrays and metadata."""
    cfg = cfg or PreprocessConfig()
    _validate_cfg(cfg)

    valid_rows = validate_layout(data_dir, metadata_df)
    all_windows: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_groups: list[np.ndarray] = []
    all_sfreqs: list[np.ndarray] = []
    all_meta: list[pd.DataFrame] = []

    for row in valid_rows:
        wr = preprocess_recording(
            edf_path=row.edf_path,
            patient_id=row.patient_id,
            recording_id=row.recording_id,
            label=row.label,
            cfg=cfg,
        )
        n_windows = wr.windows.shape[0]
        if n_windows == 0:
            raise ValueError(f"No windows generated for {row.edf_path}")

        meta = wr.metadata.copy()
        meta["sfreq"] = float(wr.sfreq)
        meta["n_channels"] = int(wr.windows.shape[1])
        meta["window_samples"] = int(wr.windows.shape[2])

        all_windows.append(wr.windows)
        all_labels.append(np.full(n_windows, row.label, dtype=int))
        all_groups.append(np.full(n_windows, row.patient_id, dtype=object))
        all_sfreqs.append(np.full(n_windows, wr.sfreq, dtype=float))
        all_meta.append(meta)

    if not all_windows:
        raise ValueError("No recordings available for preprocessing.")

    windows = np.concatenate(all_windows, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    groups = np.concatenate(all_groups, axis=0)
    sfreqs = np.concatenate(all_sfreqs, axis=0)
    metadata = pd.concat(all_meta, ignore_index=True)

    return windows, labels, groups, sfreqs, metadata
