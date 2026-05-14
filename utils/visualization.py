"""Visualization utilities for Somnotate outputs."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from utils.somnotate._manual_state_annotation import TimeSeriesStateViewer
from utils.somnotate._utils import convert_state_intervals_to_state_vector, _get_intervals
from utils.somnotate_pipeline.configuration import (
    plot_raw_signals,
    plot_states,
    state_to_color,
    state_display_order,
    state_to_int,
    default_selection_length,
    default_view_length,
)


def plot_recording_comparison(
    raw_signals: np.ndarray,
    sampling_rate_hz: float,
    somnotate_states: list[str],
    somnotate_intervals: list[tuple[float, float]],
    manual_states: list[str] | None = None,
    manual_intervals: list[tuple[float, float]] | None = None,
    epoch_seconds: int = 10,
):
    fig = plt.figure(figsize=(12, 7))
    from matplotlib.gridspec import GridSpec

    gs = GridSpec(5 if manual_states else 4, 1)
    data_axis = fig.add_subplot(gs[:3, 0])
    auto_axis = fig.add_subplot(gs[3, 0], sharex=data_axis)
    manual_axis = fig.add_subplot(gs[4, 0], sharex=data_axis) if manual_states else None
    fig.tight_layout(**{"rect": [0.05, 0, 1, 1], "pad": 2.0, "h_pad": 0.5})

    plot_raw_signals(
        raw_signals,
        sampling_frequency=sampling_rate_hz,
        ax=data_axis,
        linewidth=1.0,
    )

    plot_states(somnotate_states, somnotate_intervals, ax=auto_axis, linewidth=5.0)
    auto_axis.set_ylabel("Somnotate")

    if manual_states and manual_axis is not None:
        plot_states(manual_states, manual_intervals, ax=manual_axis, linewidth=5.0)
        manual_axis.set_ylabel("Manual")

    auto_axis.set_xlabel("Time [s]")
    return fig


def compare_vectors(auto_vec: np.ndarray, manual_vec: np.ndarray) -> list[tuple[int, int]]:
    auto_vec = np.abs(auto_vec)
    manual_vec = np.abs(manual_vec)
    min_len = min(len(auto_vec), len(manual_vec))
    auto_vec = auto_vec[:min_len]
    manual_vec = manual_vec[:min_len]
    is_discrepancy = auto_vec != manual_vec
    return _get_intervals(is_discrepancy)
