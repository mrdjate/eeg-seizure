from dataclasses import dataclass


@dataclass(slots=True)
class PreprocessConfig:
    notch_freq: float = 50.0
    l_freq: float = 0.5
    h_freq: float = 40.0
    window_sec: float = 4.0
    overlap: float = 0.5
