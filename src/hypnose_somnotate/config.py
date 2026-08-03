"""Configuration defaults for the EEG preprocessing pipeline."""

from pathlib import Path

DEFAULT_SAMPLING_RATE_HZ = 512

# Somnotate's scoring epoch: the resolution at which sleep stages are scored and
# manual annotations are collapsed. NOT the same thing as the analysis epoch used
# by the state-distribution plots — see DEFAULT_DISTRIBUTION_EPOCH_LENGTH_S below.
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
# hypnose-behavior-analysis data-location system is unavailable (see paths.py).
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

# --------------------------------------------------------------------------------
# Viewer / figure defaults
# --------------------------------------------------------------------------------
# Used when a caller passes nothing. Every one of these is overridable per call
# (in a notebook) or per flag (from the CLI) — they are defaults, not policy.
#
# Note: somnotate's own vendored configuration.py sets default_view_length = 60.,
# but the detailed viewer has always overridden it with 120.0; DEFAULT_VIEW_LENGTH_S
# keeps that behaviour and makes it visible in one place.

# Which EDF to use when a subject/date has several recordings.
DEFAULT_RECORDING_INDEX = 0

# Which EEG channel the detailed viewer and the distributions use (0 = EEG1, 1 = EEG2).
DEFAULT_EEG_CHANNEL = 0

# Width of the interactive viewer window, in seconds.
DEFAULT_VIEW_LENGTH_S = 120.0

# Analysis epoch for the state-distribution plots, in seconds: the window each
# metric (delta power, theta power, T:D ratio, EMG RMS) is reduced over.
#
# Deliberately independent of DEFAULT_SLEEP_STAGE_RESOLUTION_S (somnotate's 10 s
# scoring epoch). These two are different quantities and must not be unified:
# scoring resolution is fixed by the model, whereas the analysis window is a free
# parameter of the distribution plot.
DEFAULT_DISTRIBUTION_EPOCH_LENGTH_S = 5.0

# Histogram bin width for the normalised state-distribution overlay
# (fraction of the 0–1 normalised range).
DEFAULT_DISTRIBUTION_BIN_WIDTH = 0.05

# --------------------------------------------------------------------------------
# Figure style
# --------------------------------------------------------------------------------
# Which hypnose-behavior-analysis figure style every plot in this package is drawn with.
# One of "nature", "poster", "presentation", or None to leave matplotlib alone.
#
# This is only a declaration — nothing is applied at import time. The style is
# applied by io.style.ensure_style(), which each plotting entry point calls before
# creating a figure. That ordering matters: somnotate's vendored configuration.py
# writes figure.figsize / {axes,xtick,ytick}.labelsize when it is imported, so a
# style applied at import time would win or lose depending on import order.
# Applying at figure-creation time is deterministic.
#
# Edit this and restart the kernel to change the style everywhere, or switch it
# live for one session with ensure_style("poster", force=True).
DEFAULT_FIGURE_STYLE = "presentation"  # "nature", "poster", "presentation"
