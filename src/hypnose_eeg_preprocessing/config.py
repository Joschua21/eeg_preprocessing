"""Configuration defaults for the EEG preprocessing pipeline."""

from pathlib import Path

DEFAULT_SAMPLING_RATE_HZ = 512
DEFAULT_SLEEP_STAGE_RESOLUTION_S = 10

DEFAULT_CHANNEL_LABELS = [
    "EEG EEG1A-B",
    "EEG EEG2A-B",
    "EMG EMG",
]

# Name of the EEG dataset directory under the shared server root
# (…/hypnose/hypnose_eeg), used by the data-location resolver in paths.py.
EEG_SUBDIR = "hypnose_eeg"

# Legacy fallback: repo-local symlink to the EEG dataset, used only when the shared
# hypnose-analysis data-location system is unavailable (see paths.py).
DATA_SYMLINK = Path("data/hypnose_eeg")

# Output labels: 0=Wake, 1=NREM, 2=REM, 3=Undefined
MODEL_TO_OUTPUT_LABEL = {
    0: 3,
    1: 0,
    2: 1,
    3: 2,
}

# Manual CSV codes: 0=Wake, 1=NREM, 2=REM, 4=Undefined
MANUAL_TO_OUTPUT_LABEL = {
    0: 0,
    1: 1,
    2: 2,
    4: 3,
}

PROBABILITY_JSON_KEYS = {
    "W": 1,
    "N": 2,
    "R": 3,
    "U": 0,
}
