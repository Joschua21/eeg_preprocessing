"""Plotting of scored recordings and manual/automated comparisons.

`load_somnotate_vector` now lives in `hypnose_eeg_preprocessing.io.loading`; it is
re-exported here so existing callers keep working.
"""

from ..io.loading import load_somnotate_vector
from .visualization import (
    compare_vectors,
    plot_detailed_comparison,
    plot_recording_comparison,
    plot_scoring_detailed,
    plot_state_distributions,
)

__all__ = [
    "compare_vectors",
    "load_somnotate_vector",
    "plot_detailed_comparison",
    "plot_recording_comparison",
    "plot_scoring_detailed",
    "plot_state_distributions",
]
