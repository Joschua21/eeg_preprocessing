"""Testing workflow for Somnotate models against annotated MAT files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.somnotate._automated_state_annotation import StateAnnotator
from utils.somnotate._manual_state_annotation import TimeSeriesStateViewer
from utils.somnotate._utils import convert_state_vector_to_state_intervals, _get_intervals
from utils.somnotate_pipeline import configuration
from utils.somnotate_pipeline.mat_to_csv import mat_to_csv

from .config import DEFAULT_SAMPLING_RATE_HZ, DEFAULT_SLEEP_STAGE_RESOLUTION_S
from .paths import get_derivatives_root
from .preprocessing import preprocess_multichannel


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

    output_root = derivatives_root / "somnotate_testing" / model_name
    if test_name:
        output_root = output_root / test_name
    csv_dir = output_root / "intermediate" / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    mat_to_csv(str(testing_mat_dir), str(csv_dir), sampling_rate_hz, sleep_stage_resolution_s)
    return csv_dir


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
    df = pd.read_csv(csv_path, encoding="latin1")

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

    df = pd.read_csv(csv_path, encoding="latin1")
    required = ["EEG1", "EEG2", "EMG"]
    if not set(required).issubset(df.columns):
        raise ValueError(f"Missing required columns in {csv_path}")
    return df[required].to_numpy(dtype=float)


def load_manual_vectors_from_csvs(
    csv_paths: Iterable[Path],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    raw_signals: np.ndarray | None = None
    manual_vectors: dict[str, np.ndarray] = {}

    for csv_path in csv_paths:
        signals, vectors = load_testing_csv(csv_path)
        if raw_signals is None:
            raw_signals = signals
        else:
            if signals.shape != raw_signals.shape:
                raise ValueError(
                    "Raw signal shapes differ across CSVs. "
                    f"{csv_path.name} has {signals.shape}, expected {raw_signals.shape}."
                )

        for name, vec in vectors.items():
            label = name
            if label in manual_vectors:
                label = f"{csv_path.stem}:{name}"
            manual_vectors[label] = vec

    if raw_signals is None:
        raise ValueError("No CSV paths provided")

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
    configuration.plot_states(auto_states, auto_intervals, ax=auto_axis, linewidth=5.0)
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

    viewer = TimeSeriesStateViewer(
        data_axis,
        auto_axis,
        interval_to_state=zip(auto_intervals, auto_states),
        regions_of_interest=regions_of_interest,
        state_to_color=configuration.state_to_color,
        state_display_order=configuration.state_display_order,
        default_selection_length=configuration.default_selection_length,
        default_view_length=configuration.default_view_length,
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
