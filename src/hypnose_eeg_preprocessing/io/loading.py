"""Readers for pipeline artifacts on disk.

Everything here is I/O: resolving derivative paths, iterating directories, and
reading/parsing the files the pipeline writes. Nothing in this module runs a
model or produces a figure, and it imports only from `.paths` and `..config`,
so it stays cheap and free of cycles with the analysis modules that use it.

`load_scored_recording` is the selector-driven entry point: subject + date in,
loaded arrays out.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ..config import (
    DEFAULT_CHANNEL_LABELS,
    DEFAULT_RECORDING_INDEX,
    DEFAULT_SAMPLING_RATE_HZ,
    MODEL_TO_OUTPUT_LABEL,
)
from .paths import find_recordings, get_derivatives_root


# --------------------------------------------------------------------------------
# Derivative path resolution
# --------------------------------------------------------------------------------

def get_test_root(repo_root: Path, model_name: str, test_name: str | None) -> Path:
    root = get_derivatives_root(repo_root) / "somnotate_testing" / model_name
    return root / test_name if test_name else root


def get_csv_dir(repo_root: Path, model_name: str, test_name: str | None) -> Path:
    return get_test_root(repo_root, model_name, test_name) / "intermediate" / "csv"


def get_predictions_dir(repo_root: Path, model_name: str, test_name: str | None) -> Path:
    return get_test_root(repo_root, model_name, test_name) / "somnotate_predictions"


# --------------------------------------------------------------------------------
# Signal tables (parquet / CSV)
# --------------------------------------------------------------------------------

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


# --------------------------------------------------------------------------------
# Prediction readers
# --------------------------------------------------------------------------------

def load_somnotate_predictions(pred_path: Path) -> np.ndarray:
    if pred_path.suffix == ".parquet":
        df = pd.read_parquet(pred_path)
    else:
        df = pd.read_csv(pred_path)
    if "label" not in df.columns:
        raise ValueError(f"Missing 'label' column in {pred_path}")
    return df["label"].to_numpy(dtype=int)


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


# --------------------------------------------------------------------------------
# Manually annotated testing tables
# --------------------------------------------------------------------------------

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

    # `if not …` (not `is None`) to preserve the original `x = x or default` semantics
    if not time_resolution_s:
        # imported lazily: configuration pulls in matplotlib and mutates rcParams
        from ..somnotate_pipeline.utils import configuration

        time_resolution_s = configuration.time_resolution
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


# --------------------------------------------------------------------------------
# Alignment
# --------------------------------------------------------------------------------

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


# --------------------------------------------------------------------------------
# Scored recordings (selector-driven)
# --------------------------------------------------------------------------------

def load_recording_arrays(recording, channel_labels: list[str] | None = None):
    """Load raw signals + somnotate label vector for one resolved recording.

    Returns (raw_signals, somnotate_vec). Raises FileNotFoundError if the recording
    has no somnotate predictions parquet yet.
    """
    channel_labels = channel_labels or DEFAULT_CHANNEL_LABELS
    pred_path = recording.output_dir / f"{recording.edf_path.stem}_somnotate_predictions.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(
            f"No somnotate predictions at {pred_path}. Run score_recordings for this recording first."
        )

    # imported lazily: data_io pulls in EDF readers that are slow to import
    from ..somnotate_pipeline.io.data_io import load_raw_signals

    print(f"Loading {recording.edf_path.name}…")
    raw_signals = load_raw_signals(str(recording.edf_path), channel_labels)
    somnotate_vec = load_somnotate_vector(pred_path)
    return raw_signals, somnotate_vec


def load_scored_recording(
    subjid: int | str,
    date: int | str,
    repo_root: Path,
    recording_index: int = DEFAULT_RECORDING_INDEX,
    channel_labels: list[str] | None = None,
):
    """Resolve a scored recording by subject/date and load its raw signals + labels.

    Returns (recording, raw_signals, somnotate_vec). `recording_index` is clamped to
    the valid range rather than raising, so an out-of-range index still shows a
    recording. Raises if no recording or no predictions file exists.
    """
    repo_root = Path(repo_root)

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

    raw_signals, somnotate_vec = load_recording_arrays(recording, channel_labels)
    return recording, raw_signals, somnotate_vec
