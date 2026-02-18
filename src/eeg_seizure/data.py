from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"patient_id", "recording_id", "label"}


@dataclass(frozen=True, slots=True)
class RecordingRow:
    patient_id: str
    recording_id: str
    label: int
    edf_path: Path


def load_metadata(metadata_path: str | Path) -> pd.DataFrame:
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")

    df = pd.read_csv(metadata_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing metadata columns: {sorted(missing)}. "
            f"Required: {sorted(REQUIRED_COLUMNS)}"
        )
    if df.empty:
        raise ValueError(f"Metadata CSV is empty: {metadata_path}")

    df = df.copy()
    df["patient_id"] = df["patient_id"].astype(str)
    df["recording_id"] = df["recording_id"].astype(str)
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)

    bad = sorted(set(df["label"].tolist()) - {0, 1})
    if bad:
        raise ValueError(f"Invalid label values {bad}. Allowed values are 0/1.")
    return df


def expected_edf_path(data_dir: str | Path, patient_id: str, recording_id: str) -> Path:
    return Path(data_dir) / patient_id / f"{recording_id}.edf"


def validate_layout(data_dir: str | Path, metadata_df: pd.DataFrame) -> list[RecordingRow]:
    rows: list[RecordingRow] = []
    missing: list[str] = []

    for row in metadata_df.itertuples(index=False):
        patient_id = str(row.patient_id)
        recording_id = str(row.recording_id)
        label = int(row.label)
        edf_path = expected_edf_path(data_dir, patient_id, recording_id)
        if not edf_path.exists():
            missing.append(f"{patient_id}/{recording_id}.edf -> {edf_path}")
            continue
        rows.append(
            RecordingRow(
                patient_id=patient_id,
                recording_id=recording_id,
                label=label,
                edf_path=edf_path,
            )
        )

    if missing:
        head = "\n".join(missing[:10])
        tail = "" if len(missing) <= 10 else f"\n... and {len(missing) - 10} more"
        raise FileNotFoundError(f"Missing EDF files:\n{head}{tail}")

    return rows
