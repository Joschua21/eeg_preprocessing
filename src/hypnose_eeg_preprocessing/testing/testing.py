"""Testing workflow for Somnotate models against annotated MAT files."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..somnotate._automated_state_annotation import StateAnnotator
from ..somnotate._manual_state_annotation import TimeSeriesStateViewer
from ..somnotate._utils import convert_state_vector_to_state_intervals, _get_intervals
from ..somnotate_pipeline.utils import configuration
from ..somnotate_pipeline.preprocessing.mat_to_csv import mat_to_signal_tables

from ..config import DEFAULT_SAMPLING_RATE_HZ, DEFAULT_SLEEP_STAGE_RESOLUTION_S
from ..io.paths import get_derivatives_root
from ..preprocessing.preprocessing import preprocess_multichannel


def get_test_root(repo_root: Path, model_name: str, test_name: str | None) -> Path:
    root = get_derivatives_root(repo_root) / "somnotate_testing" / model_name
    return root / test_name if test_name else root


def get_csv_dir(repo_root: Path, model_name: str, test_name: str | None) -> Path:
    return get_test_root(repo_root, model_name, test_name) / "intermediate" / "csv"


def get_predictions_dir(repo_root: Path, model_name: str, test_name: str | None) -> Path:
    return get_test_root(repo_root, model_name, test_name) / "somnotate_predictions"


SIGNAL_TABLE_SUFFIXES: tuple[str, ...] = (".parquet", ".csv")


def load_signal_table(path: Path) -> pd.DataFrame:
    """Read a per-recording signal table (parquet or CSV) into a DataFrame."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path, encoding="latin1")
    raise ValueError(f"Unsupported signal-table format: {path}")


def list_csv_files(csv_dir: Path) -> list[Path]:
    """Return signal-table files in csv_dir (parquet + csv).

    If both a .parquet and .csv exist for the same stem, parquet wins.
    Kept under the legacy name `list_csv_files` for API stability.
    """
    if not csv_dir.exists():
        return []
    by_stem: dict[str, Path] = {}
    # Iterate in order of suffix preference: parquet first, csv as fallback.
    for suffix in SIGNAL_TABLE_SUFFIXES:
        for path in csv_dir.glob(f"*{suffix}"):
            if path.name.startswith("._"):
                continue
            by_stem.setdefault(path.stem, path)
    return sorted(by_stem.values())


def find_somnotate_predictions(predictions_dir: Path) -> Path | None:
    if not predictions_dir.exists():
        return None
    preds = sorted(p for p in predictions_dir.glob("*_somnotate.parquet") if not p.name.startswith("._"))
    return preds[0] if preds else None


def prepare_testing_csvs(
    testing_mat_dir: Path,
    repo_root: Path,
    model_name: str,
    test_name: str | None = None,
    sampling_rate_hz: int = DEFAULT_SAMPLING_RATE_HZ,
    sleep_stage_resolution_s: int = DEFAULT_SLEEP_STAGE_RESOLUTION_S,
) -> Path:
    if not model_name:
        raise ValueError("model_name is required")

    derivatives_root = get_derivatives_root(repo_root)
    if not derivatives_root.exists():
        raise FileNotFoundError(
            f"Derivatives root not found at {derivatives_root}. Create data/hypnose_eeg/derivatives."
        )

    csv_dir = get_csv_dir(repo_root, model_name, test_name)
    csv_dir.mkdir(parents=True, exist_ok=True)

    mat_to_signal_tables(str(testing_mat_dir), str(csv_dir), sampling_rate_hz, sleep_stage_resolution_s)
    return csv_dir


def ensure_csvs(
    testing_mat_dir: Path,
    repo_root: Path,
    model_name: str,
    test_name: str | None = None,
    force: bool = False,
    sampling_rate_hz: int = DEFAULT_SAMPLING_RATE_HZ,
    sleep_stage_resolution_s: int = DEFAULT_SLEEP_STAGE_RESOLUTION_S,
) -> tuple[Path, list[Path]]:
    """Return (csv_dir, signal_files). Skip mat_to_signal_tables if all expected files already exist.

    A recording counts as "already present" if either `<stem>.parquet` or `<stem>.csv` exists.
    """
    csv_dir = get_csv_dir(repo_root, model_name, test_name)
    csv_files = list_csv_files(csv_dir)

    mat_files = sorted(testing_mat_dir.glob("*.mat")) if testing_mat_dir.exists() else []
    expected_stems = {p.stem for p in mat_files if not p.name.startswith("._")}
    present_stems = {p.stem for p in csv_files}
    missing = expected_stems - present_stems

    if not force and csv_files and not missing:
        print(f"Using {len(csv_files)} existing signal table(s) in {csv_dir}")
        return csv_dir, csv_files

    if not mat_files:
        raise FileNotFoundError(
            f"No signal tables in {csv_dir} and no .mat files in {testing_mat_dir} to convert."
        )

    print(f"Converting {len(mat_files)} .mat file(s) -> {csv_dir} (parquet)")
    prepare_testing_csvs(
        testing_mat_dir=testing_mat_dir,
        repo_root=repo_root,
        model_name=model_name,
        test_name=test_name,
        sampling_rate_hz=sampling_rate_hz,
        sleep_stage_resolution_s=sleep_stage_resolution_s,
    )
    return csv_dir, list_csv_files(csv_dir)


def ensure_somnotate_predictions(
    csv_files: list[Path],
    predictions_dir: Path,
    model_path: Path,
    force: bool = False,
    sampling_rate_hz: int = DEFAULT_SAMPLING_RATE_HZ,
) -> Path:
    """Return path to a somnotate prediction parquet, computing it from csv_files[0] if missing."""
    if not force:
        existing = find_somnotate_predictions(predictions_dir)
        if existing is not None:
            print(f"Using existing predictions: {existing}")
            return existing

    if not csv_files:
        raise FileNotFoundError(
            f"No somnotate predictions in {predictions_dir} and no CSV files to compute from."
        )
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Computing somnotate predictions from {csv_files[0].name}")
    output_path, _ = save_somnotate_predictions(
        csv_files[0],
        model_path,
        predictions_dir,
        sampling_rate_hz=sampling_rate_hz,
    )
    return output_path


def load_aligned_vectors(
    csv_dir: Path,
    predictions_dir: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Load raw_signals, somnotate_vec, manual_vectors from disk and align lengths.

    Raises FileNotFoundError with a clear hint if either input is missing.
    """
    csv_files = list_csv_files(csv_dir)
    if not csv_files:
        raise FileNotFoundError(
            f"No CSVs in {csv_dir}. Run the data-preparation cell (ensure_csvs / ensure_somnotate_predictions)."
        )
    pred_path = find_somnotate_predictions(predictions_dir)
    if pred_path is None:
        raise FileNotFoundError(
            f"No somnotate predictions in {predictions_dir}. Run the data-preparation cell."
        )

    raw_signals, manual_vectors = load_manual_vectors_from_csvs(csv_files)
    somnotate_vec = load_somnotate_predictions(pred_path)
    somnotate_vec, manual_vectors = align_vectors(somnotate_vec, manual_vectors)
    return raw_signals, somnotate_vec, manual_vectors


def _normalize_stage_values(stage_values: Iterable) -> np.ndarray:
    series = pd.to_numeric(pd.Series(stage_values), errors="coerce").fillna(0).astype(int)
    stage_map = {1: 1, 2: 2, 3: 3, 4: 0, 0: 0}
    mapped = series.map(lambda v: stage_map.get(int(v), 0)).to_numpy(dtype=int)
    return mapped


def _collapse_stage_vector(
    stage_values: np.ndarray,
    sampling_rate_hz: int,
    time_resolution_s: float,
) -> np.ndarray:
    epoch_samples = int(round(sampling_rate_hz * time_resolution_s))
    if epoch_samples <= 0:
        raise ValueError("epoch_samples must be >= 1")

    total_epochs = len(stage_values) // epoch_samples
    if total_epochs == 0:
        return np.array([], dtype=int)

    trimmed = stage_values[: total_epochs * epoch_samples]
    reshaped = trimmed.reshape(total_epochs, epoch_samples)
    epoch_labels = reshaped[:, 0]

    non_uniform = np.sum(~np.all(reshaped == reshaped[:, [0]], axis=1))
    if non_uniform:
        print(
            "Warning: sleepStage values vary within some epochs; "
            f"using the first sample for {non_uniform} epochs."
        )

    return epoch_labels.astype(int)


def load_testing_csv(
    csv_path: Path,
    sampling_rate_hz: int = DEFAULT_SAMPLING_RATE_HZ,
    time_resolution_s: float | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if csv_path.name.startswith("._"):
        raise ValueError(f"Metadata file not supported: {csv_path}")

    time_resolution_s = time_resolution_s or configuration.time_resolution
    df = load_signal_table(csv_path)

    required = ["EEG1", "EEG2", "EMG"]
    if not set(required).issubset(df.columns):
        raise ValueError(f"Missing required columns in {csv_path}")

    raw_signals = df[required].to_numpy(dtype=float)

    stage_columns = [col for col in df.columns if "sleepstage" in col.lower()]
    if not stage_columns:
        raise ValueError(f"No sleepStage columns found in {csv_path}")

    manual_vectors: dict[str, np.ndarray] = {}
    for col in stage_columns:
        normalized = _normalize_stage_values(df[col].to_numpy())
        manual_vectors[col] = _collapse_stage_vector(
            normalized,
            sampling_rate_hz=sampling_rate_hz,
            time_resolution_s=time_resolution_s,
        )

    return raw_signals, manual_vectors


def load_raw_signals_from_csv(
    csv_path: Path,
) -> np.ndarray:
    if csv_path.name.startswith("._"):
        raise ValueError(f"Metadata file not supported: {csv_path}")

    df = load_signal_table(csv_path)
    required = ["EEG1", "EEG2", "EMG"]
    if not set(required).issubset(df.columns):
        raise ValueError(f"Missing required columns in {csv_path}")
    return df[required].to_numpy(dtype=float)


def _short_label_for_stem(stem: str) -> str:
    suffix_match = re.search(r"recording-\d+[_-](.+)$", stem)
    if suffix_match:
        return suffix_match.group(1)
    subses_match = re.search(r"(sub-[^_-]+[_-]ses-[^_-]+)", stem)
    if subses_match:
        return subses_match.group(1)
    return stem


def _resolve_file_labels(csv_paths: list[Path]) -> dict[Path, str]:
    proposed = {p: _short_label_for_stem(p.stem) for p in csv_paths}
    counts = Counter(proposed.values())
    return {p: (lbl if counts[lbl] == 1 else p.stem) for p, lbl in proposed.items()}


def load_manual_vectors_from_csvs(
    csv_paths: Iterable[Path],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    csv_paths = list(csv_paths)
    if not csv_paths:
        raise ValueError("No CSV paths provided")

    file_labels = _resolve_file_labels(csv_paths)
    raw_signals: np.ndarray | None = None
    manual_vectors: dict[str, np.ndarray] = {}

    for csv_path in csv_paths:
        signals, vectors = load_testing_csv(csv_path)
        if raw_signals is None:
            raw_signals = signals
        elif signals.shape != raw_signals.shape:
            raise ValueError(
                "Raw signal shapes differ across CSVs. "
                f"{csv_path.name} has {signals.shape}, expected {raw_signals.shape}."
            )

        file_label = file_labels[csv_path]
        for name, vec in vectors.items():
            label = file_label if len(vectors) == 1 else f"{file_label}:{name}"
            if label in manual_vectors:
                label = f"{csv_path.stem}:{name}"
            manual_vectors[label] = vec

    return raw_signals, manual_vectors


def score_somnotate(
    raw_signals: np.ndarray,
    model_path: Path,
    sampling_rate_hz: int = DEFAULT_SAMPLING_RATE_HZ,
) -> np.ndarray:
    annotator = StateAnnotator()
    annotator.load(str(model_path))

    preprocessed = preprocess_multichannel(raw_signals, sampling_rate_hz)
    predicted = np.array(annotator.predict(preprocessed), dtype=int)
    return np.abs(predicted)


def save_somnotate_predictions(
    csv_path: Path,
    model_path: Path,
    output_dir: Path,
    sampling_rate_hz: int = DEFAULT_SAMPLING_RATE_HZ,
    time_resolution_s: float | None = None,
) -> tuple[Path, np.ndarray]:
    time_resolution_s = time_resolution_s or configuration.time_resolution
    raw_signals = load_raw_signals_from_csv(csv_path)
    somnotate_vec = score_somnotate(raw_signals, model_path, sampling_rate_hz)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{csv_path.stem}_somnotate.parquet"
    timepoints = np.arange(0, len(somnotate_vec) * time_resolution_s, time_resolution_s, dtype=float)
    pd.DataFrame({"time_s": timepoints, "label": somnotate_vec}).to_parquet(output_path, index=False)
    return output_path, somnotate_vec


def load_somnotate_predictions(pred_path: Path) -> np.ndarray:
    if pred_path.suffix == ".parquet":
        df = pd.read_parquet(pred_path)
    else:
        df = pd.read_csv(pred_path)
    if "label" not in df.columns:
        raise ValueError(f"Missing 'label' column in {pred_path}")
    return df["label"].to_numpy(dtype=int)


def align_vectors(
    somnotate_vec: np.ndarray,
    manual_vectors: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    lengths = [len(somnotate_vec)] + [len(vec) for vec in manual_vectors.values()]
    min_len = min(lengths) if lengths else 0
    if min_len == 0:
        return somnotate_vec[:0], {name: vec[:0] for name, vec in manual_vectors.items()}

    somnotate_vec = somnotate_vec[:min_len]
    manual_vectors = {name: vec[:min_len] for name, vec in manual_vectors.items()}
    return somnotate_vec, manual_vectors


def plot_testing_comparison(
    raw_signals: np.ndarray,
    sampling_rate_hz: int,
    somnotate_vec: np.ndarray,
    manual_vectors: dict[str, np.ndarray],
    time_resolution_s: float | None = None,
):
    time_resolution_s = time_resolution_s or configuration.time_resolution

    state_rows = 1 + len(manual_vectors)
    fig = plt.figure(figsize=(12, 6 + 1.2 * state_rows))
    from matplotlib.gridspec import GridSpec

    gs = GridSpec(1 + state_rows, 1)
    data_axis = fig.add_subplot(gs[0, 0])
    auto_axis = fig.add_subplot(gs[1, 0], sharex=data_axis)

    manual_axes = []
    for idx in range(len(manual_vectors)):
        manual_axes.append(fig.add_subplot(gs[idx + 2, 0], sharex=data_axis))

    fig.tight_layout(**{"rect": [0.05, 0, 1, 1], "pad": 2.0, "h_pad": 0.6})

    configuration.plot_raw_signals(
        raw_signals,
        sampling_frequency=sampling_rate_hz,
        ax=data_axis,
        linewidth=1.0,
    )

    auto_states, auto_intervals = convert_state_vector_to_state_intervals(
        somnotate_vec,
        time_resolution=time_resolution_s,
        mapping=configuration.int_to_state,
    )
    # NOTE: do NOT call configuration.plot_states for somnotate here — the viewer below
    # plots the somnotate state intervals on auto_axis itself, so calling plot_states
    # would draw them twice.
    auto_axis.set_ylabel("Somnotate")

    manual_names = list(manual_vectors.keys())
    for axis, name in zip(manual_axes, manual_names):
        manual_states, manual_intervals = convert_state_vector_to_state_intervals(
            manual_vectors[name],
            time_resolution=time_resolution_s,
            mapping=configuration.int_to_state,
        )
        configuration.plot_states(manual_states, manual_intervals, ax=axis, linewidth=5.0)
        axis.set_ylabel(name)

    auto_axis.set_xlabel("Time [s]")

    regions_of_interest = None
    if manual_vectors:
        mismatch = np.zeros_like(somnotate_vec, dtype=bool)
        for vec in manual_vectors.values():
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
        default_view_length=configuration.default_view_length,
    )

    return fig, viewer


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
) -> None:
    """Plot multiple signals on one axis, stacked vertically with per-signal colors.

    First signal in the list ends up at the top of the plot (matches plot_signals convention).
    """
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

    # vertical offsets so signals don't overlap
    offsets = np.arange(arr.shape[1], dtype=float)
    arr = arr + offsets[None, :]

    n_samples = arr.shape[0]
    time = np.arange(n_samples, dtype=float) / sampling_rate_hz

    for i in range(arr.shape[1]):
        ax.plot(time, arr[:, i], color=colors_rev[i], linewidth=linewidth)
        ax.axhline(offsets[i] + 0.5, color="0.85", linewidth=0.5, linestyle=":")

    ax.set_yticks(offsets + 0.5)
    ax.set_yticklabels(labels_rev)
    ax.set_xlim(time[0], time[-1])


def plot_testing_comparison_detailed(
    raw_signals: np.ndarray,
    sampling_rate_hz: int,
    somnotate_vec: np.ndarray,
    manual_vectors: dict[str, np.ndarray],
    scorers: list[str] | None = None,
    eeg_channel: int = 0,
    view_length_s: float = 120.0,
    emg_rms_window_s: float = 5.0,
    band_smoothing_window_s: float = 5.0,
    delta_band: tuple[float, float] = (0.5, 4.0),
    theta_band: tuple[float, float] = (6.0, 10.0),
    time_resolution_s: float | None = None,
):
    """Like plot_testing_comparison, but with derived signal channels and a scorer filter.

    Signals plotted (top to bottom): EEG raw, EMG raw, EMG RMS, Delta envelope,
    Theta envelope, Theta:Delta ratio. State rows: somnotate plus whichever scorers
    are selected via `scorers` (default: all). Mismatch ROIs use only the selected scorers.

    Keyboard navigation (left/right etc.) is inherited from TimeSeriesStateViewer.
    """
    time_resolution_s = time_resolution_s or configuration.time_resolution

    if scorers is None:
        selected = dict(manual_vectors)
    else:
        missing = [s for s in scorers if s not in manual_vectors]
        if missing:
            raise KeyError(
                f"Unknown scorer(s): {missing}. Available: {list(manual_vectors.keys())}"
            )
        selected = {s: manual_vectors[s] for s in scorers}

    if eeg_channel not in (0, 1):
        raise ValueError("eeg_channel must be 0 (EEG1) or 1 (EEG2)")

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


def agreement_matrix(
    somnotate_vec: np.ndarray,
    manual_vectors: dict[str, np.ndarray],
) -> pd.DataFrame:
    vectors = {"somnotate": somnotate_vec}
    vectors.update(manual_vectors)

    names = list(vectors.keys())
    matrix = np.zeros((len(names), len(names)), dtype=float)
    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            vec_i = vectors[name_i]
            vec_j = vectors[name_j]
            if len(vec_i) == 0 or len(vec_j) == 0:
                matrix[i, j] = np.nan
            else:
                min_len = min(len(vec_i), len(vec_j))
                matrix[i, j] = float(np.mean(vec_i[:min_len] == vec_j[:min_len]) * 100.0)

    return pd.DataFrame(matrix, index=names, columns=names)


def plot_agreement_matrix(
    agreement_df: pd.DataFrame,
    output_path: Path | None = None,
    title: str = "Agreement Matrix (%)",
):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(agreement_df.to_numpy(), cmap="Blues", vmin=0, vmax=100)

    ax.set_xticks(range(len(agreement_df.columns)))
    ax.set_yticks(range(len(agreement_df.index)))
    ax.set_xticklabels(agreement_df.columns, rotation=45, ha="right")
    ax.set_yticklabels(agreement_df.index)
    ax.set_title(title)

    for i in range(agreement_df.shape[0]):
        for j in range(agreement_df.shape[1]):
            value = agreement_df.iat[i, j]
            if np.isnan(value):
                label = "nan"
            else:
                label = f"{value:.2f}"
            ax.text(j, i, label, ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)

    return fig


def _state_label(code: int) -> str:
    return {0: "0/undef", 1: "1/wake", 2: "2/NREM", 3: "3/REM"}.get(int(code), f"{code}/?")


def print_agreement_diagnostic(
    somnotate_vec: np.ndarray,
    manual_vectors: dict[str, np.ndarray],
) -> None:
    """Print per-vector stats and somnotate-vs-manual confusion matrices.

    Run after align_vectors so all inputs share length.
    """
    all_vectors: dict[str, np.ndarray] = {"somnotate": somnotate_vec, **manual_vectors}

    print("=== Vector lengths ===")
    for name, vec in all_vectors.items():
        print(f"  {name:40s} {len(vec)} epochs")

    print("\n=== Unique codes per vector ===")
    for name, vec in all_vectors.items():
        uniq = sorted({int(v) for v in np.unique(vec)})
        print(f"  {name:40s} {uniq}")

    all_codes = sorted({int(v) for vec in all_vectors.values() for v in np.unique(vec)})
    print("\n=== State distribution (epoch counts) ===")
    header = "  " + " " * 40 + "".join(f"{_state_label(c):>10s}" for c in all_codes) + f"{'total':>10s}"
    print(header)
    for name, vec in all_vectors.items():
        counts = [int(np.sum(vec == c)) for c in all_codes]
        row = "  " + f"{name:40s}" + "".join(f"{n:>10d}" for n in counts) + f"{len(vec):>10d}"
        print(row)

    for scorer, manual_vec in manual_vectors.items():
        n = min(len(somnotate_vec), len(manual_vec))
        s = somnotate_vec[:n]
        m = manual_vec[:n]
        print(f"\n=== Confusion: somnotate (rows) vs {scorer} (cols) ===")
        codes = sorted({int(v) for v in np.unique(np.concatenate([s, m]))})
        col_header = "  " + " " * 14 + "".join(f"{_state_label(c):>10s}" for c in codes) + f"{'recall':>10s}"
        print(col_header)
        col_totals = np.zeros(len(codes), dtype=int)
        for i, ci in enumerate(codes):
            row_mask = s == ci
            row_total = int(row_mask.sum())
            cells = []
            for j, cj in enumerate(codes):
                cnt = int(np.sum(row_mask & (m == cj)))
                cells.append(cnt)
                col_totals[j] += cnt
            recall = (cells[i] / row_total * 100.0) if row_total else float("nan")
            row = "  " + f"{_state_label(ci):<14s}" + "".join(f"{c:>10d}" for c in cells) + f"{recall:>9.2f}%"
            print(row)
        precisions = []
        for j, cj in enumerate(codes):
            row_mask = s == cj
            tp = int(np.sum(row_mask & (m == cj)))
            denom = int(col_totals[j])
            precisions.append((tp / denom * 100.0) if denom else float("nan"))
        prec_row = "  " + f"{'precision':<14s}" + "".join(f"{p:>9.2f}%" for p in precisions) + " " * 10
        print(prec_row)
        overall = float(np.mean(s == m) * 100.0)
        print(f"  overall agreement: {overall:.2f}%   (recall = % of somnotate's class confirmed by scorer; precision = % of scorer's class confirmed by somnotate)")
