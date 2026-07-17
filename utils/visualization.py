"""Visualization utilities for Somnotate outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utils.somnotate._manual_state_annotation import TimeSeriesStateViewer
from utils.somnotate._utils import (
    convert_state_intervals_to_state_vector,
    convert_state_vector_to_state_intervals,
    _get_intervals,
)
from utils.somnotate_pipeline import configuration
from utils.somnotate_pipeline.configuration import (
    plot_raw_signals,
    plot_states,
    state_to_color,
    state_display_order,
    state_to_int,
    default_selection_length,
    default_view_length,
)

from .config import DEFAULT_CHANNEL_LABELS, DEFAULT_SAMPLING_RATE_HZ, MODEL_TO_OUTPUT_LABEL
from .paths import find_recordings


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


def _states_present(state_list) -> list[str]:
    """Return configuration.state_display_order filtered to states that actually appear."""
    present = set(state_list)
    return [s for s in configuration.state_display_order if s in present]


def _shade_mismatch_intervals(
    axes: list,
    epoch_intervals: np.ndarray,
    time_resolution_s: float,
    color: str = "lightblue",
    alpha: float = 0.4,
) -> None:
    """Draw a translucent vertical band on each axis for every mismatch epoch range.

    `epoch_intervals` is an (n, 2) array of [start, stop] epoch indices (the format
    `_get_intervals` returns); converted to seconds via `time_resolution_s`.
    """
    if epoch_intervals is None or len(epoch_intervals) == 0:
        return
    intervals_s = np.asarray(epoch_intervals, dtype=float) * float(time_resolution_s)
    for ax in axes:
        for start, stop in intervals_s:
            ax.axvspan(start, stop, color=color, alpha=alpha, zorder=0, linewidth=0)


def _rolling_mean(signal: np.ndarray, sampling_rate_hz: float, window_s: float) -> np.ndarray:
    win = max(int(round(sampling_rate_hz * window_s)), 1)
    kernel = np.ones(win, dtype=float) / win
    return np.convolve(signal.astype(float), kernel, mode="same")


def _rolling_rms(signal: np.ndarray, sampling_rate_hz: float, window_s: float) -> np.ndarray:
    return np.sqrt(_rolling_mean(signal.astype(float) ** 2, sampling_rate_hz, window_s))


def _bandpass_rms(
    signal: np.ndarray,
    sampling_rate_hz: float,
    lowcut: float,
    highcut: float,
    window_s: float,
) -> np.ndarray:
    """Banded power: bandpass, then windowed RMS — matches Spike2's `banded power`
    and is the closest time-domain analogue to somnotate's windowed STFT power.
    """
    filtered = configuration.chebychev_bandpass_filter(
        signal, lowcut=lowcut, highcut=highcut, fs=sampling_rate_hz
    )
    return _rolling_rms(filtered, sampling_rate_hz, window_s)


def _stacked_signal_plot(
    ax,
    signals: list[np.ndarray],
    sampling_rate_hz: float,
    labels: list[str],
    colors: list[str],
    linewidth: float = 0.8,
    show_range: bool = True,
) -> None:
    """Plot multiple signals on one axis, stacked vertically with per-signal colors.

    First signal in the list ends up at the top of the plot (matches plot_signals convention).
    Signal names are drawn as vertical (top-to-bottom) tick labels to save width; when
    `show_range` is set, each band is annotated with the value at its top and bottom edge
    (the 1st/99th-percentile scaling bounds) so absolute scale stays readable.
    """
    from matplotlib.transforms import blended_transform_factory

    arr = np.column_stack([s.astype(float) for s in signals])

    # robust per-signal [0,1] rescale using 1st/99th percentiles to suppress outliers
    lo = np.nanpercentile(arr, 1.0, axis=0)
    hi = np.nanpercentile(arr, 99.0, axis=0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    arr = np.clip((arr - lo) / span, 0.0, 1.0)

    # reverse order so the first input signal is at the top
    arr = arr[:, ::-1]
    labels_rev = labels[::-1]
    colors_rev = colors[::-1]
    lo_rev = lo[::-1]
    hi_rev = hi[::-1]

    # vertical offsets so signals don't overlap
    offsets = np.arange(arr.shape[1], dtype=float)
    arr = arr + offsets[None, :]

    n_samples = arr.shape[0]
    time = np.arange(n_samples, dtype=float) / sampling_rate_hz

    for i in range(arr.shape[1]):
        ax.plot(time, arr[:, i], color=colors_rev[i], linewidth=linewidth)
        ax.axhline(offsets[i] + 0.5, color="0.85", linewidth=0.5, linestyle=":")

    if show_range:
        # numeric scale bounds pinned to the right edge, one pair per band; the value
        # at the top of the band (hi) and at the bottom (lo). Uses the full-recording
        # percentiles the bands are normalized against.
        trans = blended_transform_factory(ax.transAxes, ax.transData)
        for i in range(arr.shape[1]):
            ax.text(0.998, offsets[i] + 0.96, f"{hi_rev[i]:.3g}", transform=trans,
                    ha="right", va="top", fontsize=6, color="0.45")
            ax.text(0.998, offsets[i] + 0.04, f"{lo_rev[i]:.3g}", transform=trans,
                    ha="right", va="bottom", fontsize=6, color="0.45")

    ax.set_yticks(offsets + 0.5)
    ax.set_yticklabels(labels_rev, rotation=-90, va="center")
    ax.set_xlim(time[0], time[-1])


def _align_to_shortest(
    somnotate_vec: np.ndarray,
    manual_vectors: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    lengths = [len(somnotate_vec)] + [len(v) for v in manual_vectors.values()]
    min_len = min(lengths)
    return somnotate_vec[:min_len], {k: v[:min_len] for k, v in manual_vectors.items()}


def plot_detailed_comparison(
    raw_signals: np.ndarray,
    sampling_rate_hz: int,
    somnotate_vec: np.ndarray,
    manual_vectors: dict[str, np.ndarray] | None = None,
    scorers: list[str] | None = None,
    eeg_channel: int = 0,
    view_length_s: float = 120.0,
    emg_rms_window_s: float = 5.0,
    band_smoothing_window_s: float = 5.0,
    delta_band: tuple[float, float] = (0.5, 4.0),
    theta_band: tuple[float, float] = (6.0, 10.0),
    time_resolution_s: float | None = None,
):
    """Interactive state view with derived signal channels; human scorers optional.

    Signals plotted (top to bottom): EEG raw, EMG raw, EMG RMS, Delta envelope,
    Theta envelope, Theta:Delta ratio. State rows: somnotate plus whichever scorers
    are selected via `scorers` (default: all of `manual_vectors`).

    With no `manual_vectors` (the scoring case) only the somnotate row is drawn and
    no mismatch regions are shaded. When scorers are present, mismatch ROIs are
    computed against the selected ones only.

    Keyboard navigation (left/right etc.) is inherited from TimeSeriesStateViewer.
    """
    time_resolution_s = time_resolution_s or configuration.time_resolution
    manual_vectors = dict(manual_vectors or {})

    if scorers is None:
        selected = manual_vectors
    else:
        missing = [s for s in scorers if s not in manual_vectors]
        if missing:
            raise KeyError(
                f"Unknown scorer(s): {missing}. Available: {list(manual_vectors.keys())}"
            )
        selected = {s: manual_vectors[s] for s in scorers}

    if eeg_channel not in (0, 1):
        raise ValueError("eeg_channel must be 0 (EEG1) or 1 (EEG2)")

    somnotate_vec, selected = _align_to_shortest(somnotate_vec, selected)

    eeg = raw_signals[:, eeg_channel]
    emg = raw_signals[:, 2]

    print("Computing derived signal channels…")
    eeg_disp = configuration.chebychev_bandpass_filter(eeg, lowcut=0.5, highcut=30.0, fs=sampling_rate_hz)
    emg_disp = configuration.chebychev_bandpass_filter(emg, lowcut=10.0, highcut=45.0, fs=sampling_rate_hz)
    emg_rms = _rolling_rms(emg_disp, sampling_rate_hz, emg_rms_window_s)
    delta_env = _bandpass_rms(
        eeg, sampling_rate_hz, *delta_band, window_s=band_smoothing_window_s
    )
    theta_env = _bandpass_rms(
        eeg, sampling_rate_hz, *theta_band, window_s=band_smoothing_window_s
    )
    td_ratio = theta_env / (delta_env + 1e-6)

    signals = [eeg_disp, emg_disp, emg_rms, delta_env, theta_env, td_ratio]
    smoothing_suffix = f", {band_smoothing_window_s:g}s" if band_smoothing_window_s else ""
    labels = [
        f"EEG{eeg_channel + 1} (raw)",
        "EMG (raw)",
        f"EMG RMS ({emg_rms_window_s:g}s)",
        f"Delta ({delta_band[0]:g}-{delta_band[1]:g} Hz{smoothing_suffix})",
        f"Theta ({theta_band[0]:g}-{theta_band[1]:g} Hz{smoothing_suffix})",
        f"T:D ratio{smoothing_suffix}" if smoothing_suffix else "T:D ratio",
    ]
    colors = ["green", "green", "orange", "pink", "red", "blue"]

    state_rows = 1 + len(selected)
    fig = plt.figure(figsize=(13, 8 + 1.0 * state_rows))
    from matplotlib.gridspec import GridSpec

    gs = GridSpec(1 + state_rows, 1, height_ratios=[4] + [1] * state_rows)
    data_axis = fig.add_subplot(gs[0, 0])
    auto_axis = fig.add_subplot(gs[1, 0], sharex=data_axis)
    manual_axes = [fig.add_subplot(gs[i + 2, 0], sharex=data_axis) for i in range(len(selected))]
    fig.tight_layout(**{"rect": [0.05, 0, 1, 1], "pad": 2.0, "h_pad": 0.6})

    _stacked_signal_plot(
        data_axis,
        signals=signals,
        sampling_rate_hz=sampling_rate_hz,
        labels=labels,
        colors=colors,
        linewidth=0.8,
    )

    auto_states, auto_intervals = convert_state_vector_to_state_intervals(
        somnotate_vec,
        time_resolution=time_resolution_s,
        mapping=configuration.int_to_state,
    )
    # The viewer plots somnotate state intervals on auto_axis itself — don't double-plot.
    auto_axis.set_ylabel("Somnotate")

    for axis, (name, vec) in zip(manual_axes, selected.items()):
        manual_states, manual_intervals = convert_state_vector_to_state_intervals(
            vec, time_resolution=time_resolution_s, mapping=configuration.int_to_state
        )
        configuration.plot_states(manual_states, manual_intervals, ax=axis, linewidth=5.0)
        axis.set_ylabel(name)

    auto_axis.set_xlabel("Time [s]")

    regions_of_interest = None
    if selected:
        mismatch = np.zeros_like(somnotate_vec, dtype=bool)
        for vec in selected.values():
            mismatch |= (somnotate_vec != vec)
        regions_of_interest = _get_intervals(mismatch)
        _shade_mismatch_intervals(
            [auto_axis, *manual_axes], regions_of_interest, time_resolution_s
        )

    viewer = TimeSeriesStateViewer(
        data_axis,
        auto_axis,
        interval_to_state=zip(auto_intervals, auto_states),
        regions_of_interest=regions_of_interest,
        state_to_color=configuration.state_to_color,
        state_display_order=_states_present(auto_states),
        default_selection_length=configuration.default_selection_length,
        default_view_length=view_length_s,
    )
    return fig, viewer


def load_somnotate_vector(pred_path: Path) -> np.ndarray:
    """Read a scoring predictions parquet into a model-label vector.

    Prefers the `label_model` column; falls back to inverting MODEL_TO_OUTPUT_LABEL
    on `label` for older prediction files that only stored output labels.
    """
    pred_df = pd.read_parquet(pred_path)
    if "label_model" in pred_df.columns:
        return pred_df["label_model"].to_numpy(dtype=int)
    if "label" not in pred_df.columns:
        raise ValueError(f"No 'label_model' or 'label' column in {pred_path}")
    inverse_map = {value: key for key, value in MODEL_TO_OUTPUT_LABEL.items()}
    return pred_df["label"].map(lambda v: inverse_map.get(int(v), 0)).to_numpy(dtype=int)


def plot_scoring_detailed(
    subjid: int | str,
    date: int | str,
    repo_root: Path,
    manual_vectors: dict[str, np.ndarray] | None = None,
    recording_index: int = 0,
    channel_labels: list[str] | None = None,
    sampling_rate_hz: int = DEFAULT_SAMPLING_RATE_HZ,
    **plot_kwargs,
):
    """Load one scored recording by subject/date and show the detailed state view.

    Finds the recording, reads its EDF and the somnotate predictions written by
    `score_recordings`, and hands both to `plot_detailed_comparison`. Human scorer
    vectors are optional — pass `manual_vectors` to add scorer rows and mismatch
    shading, or omit it to view the somnotate scoring on its own.

    `recording_index` selects among multiple EDFs for the same subject/date.
    Extra keyword arguments (eeg_channel, view_length_s, delta_band, …) pass
    through to `plot_detailed_comparison`.
    """
    repo_root = Path(repo_root)
    channel_labels = channel_labels or DEFAULT_CHANNEL_LABELS

    recordings = find_recordings(repo_root, [subjid], dates=[date])
    if not recordings:
        raise ValueError(f"No recordings found for subject {subjid} on date {date}")

    # Clamp rather than crash: an out-of-range index falls back to the nearest valid
    # recording (so recording_index=3 on a subject/date with one recording still shows it).
    clamped_index = min(max(recording_index, 0), len(recordings) - 1)
    if clamped_index != recording_index:
        print(
            f"recording_index {recording_index} out of range "
            f"({len(recordings)} recording(s) for subject {subjid} on date {date}); "
            f"showing [{clamped_index}] instead."
        )
    recording = recordings[clamped_index]
    if len(recordings) > 1:
        print(
            f"{len(recordings)} recordings for subject {subjid} on date {date}; "
            f"showing [{clamped_index}] {recording.edf_path.name}"
        )

    pred_path = recording.output_dir / f"{recording.edf_path.stem}_somnotate_predictions.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(
            f"No somnotate predictions at {pred_path}. Run score_recordings for this recording first."
        )

    # imported lazily: data_io pulls in EDF readers that are slow to import
    from utils.somnotate_pipeline.data_io import load_raw_signals

    print(f"Loading {recording.edf_path.name}…")
    raw_signals = load_raw_signals(str(recording.edf_path), channel_labels)
    somnotate_vec = load_somnotate_vector(pred_path)

    return plot_detailed_comparison(
        raw_signals,
        sampling_rate_hz=sampling_rate_hz,
        somnotate_vec=somnotate_vec,
        manual_vectors=manual_vectors,
        **plot_kwargs,
    )
