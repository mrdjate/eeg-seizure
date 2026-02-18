from pathlib import Path

import pandas as pd
import pytest

from eeg_seizure.data import load_metadata, validate_layout


def test_metadata_schema_validation(tmp_path: Path) -> None:
    bad_csv = tmp_path / "metadata.csv"
    pd.DataFrame({"patient_id": ["p1"], "recording_id": ["r1"]}).to_csv(
        bad_csv, index=False
    )
    with pytest.raises(ValueError, match="Missing metadata columns"):
        load_metadata(bad_csv)


def test_missing_edf_validation(tmp_path: Path) -> None:
    metadata_csv = tmp_path / "metadata.csv"
    pd.DataFrame(
        [{"patient_id": "p1", "recording_id": "rec1", "label": 1}]
    ).to_csv(metadata_csv, index=False)
    df = load_metadata(metadata_csv)
    with pytest.raises(FileNotFoundError, match="Missing EDF files"):
        validate_layout(tmp_path / "raw", df)
