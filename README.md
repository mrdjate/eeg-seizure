# EEG seizure detection

Python 3.11+ repo for EDF-based seizure detection using MNE and scikit-learn.

This project started in december 2023 when I wanted, for medical and personnal reason, to undertand and build a minimal, reproducible baseline for labeled EEG seizure detection. It stayed on my machine for that long. In 2026, it was updated to a clean foundation for fast experimentation and clinical relevancy threshold tuning.

Expected data layout:

```text
data/
  raw/
    <patient_id>/
      <recording_id>.edf
  metadata.csv
```

`metadata.csv` required columns:
- `patient_id`
- `recording_id`
- `label` (`0` or `1`, weak recording-level seizure label)

Validation behavior:
- raises `FileNotFoundError` if metadata is missing
- raises `ValueError` if required columns are missing or labels are not binary
- raises `FileNotFoundError` with per-file details when metadata references missing EDF files

Preprocessing defaults:
- EDF read with `mne.io.read_raw_edf(preload=True, verbose="ERROR")`
- notch filter at `50 Hz`
- bandpass `0.5-40 Hz`
- average reference when available
- windowing: `4.0s` windows with `0.5` overlap

Window-level output includes:
- `windows`: `(n_windows, n_channels, n_samples_per_window)`
- `labels`: `(n_windows,)`
- `groups`: `(n_windows,)` patient ids
- `sfreqs`: `(n_windows,)`
- metadata columns: `patient_id`, `recording_id`, `window_index`, `label`
