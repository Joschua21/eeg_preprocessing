"""Scoring workflow for Somnotate models."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from utils.somnotate._automated_state_annotation import StateAnnotator
from utils.somnotate._utils import convert_state_vector_to_state_intervals
from utils.somnotate_pipeline.configuration import time_resolution
from utils.somnotate_pipeline.data_io import load_raw_signals, export_hypnogram

from .config import (
    DEFAULT_CHANNEL_LABELS,
    DEFAULT_SAMPLING_RATE_HZ,
    MODEL_TO_OUTPUT_LABEL,
    PROBABILITY_JSON_KEYS,
)
from .gap_correction import PreparedRecording, prepare_recording
from .paths import find_recordings, get_derivatives_root
from .preprocessing import preprocess_multichannel
from .somnotate_pipeline import configuration


UNDEFINED_MODEL_LABEL = 0
UNDEFINED_OUTPUT_LABEL = MODEL_TO_OUTPUT_LABEL.get(UNDEFINED_MODEL_LABEL, 3)


def score_recordings(
    subjids: list[int | str],
    model_path: Path,
    repo_root: Path,
    dates: list[int | str] | None = None,
    date_range: tuple[int | str, int | str] | None = None,
    channel_labels: list[str] | None = None,
    export_visbrain: bool = True,
    sampling_rate_hz: int = DEFAULT_SAMPLING_RATE_HZ,
) -> list[Path]:
    derivatives_root = get_derivatives_root(repo_root)
    if not derivatives_root.exists():
        raise FileNotFoundError(
            f"Derivatives root not found at {derivatives_root}. Create a symlink named 'derivatives' in the repo."
        )

    channel_labels = channel_labels or DEFAULT_CHANNEL_LABELS
    recordings = find_recordings(repo_root, subjids, dates=dates, date_range=date_range)

    annotator = StateAnnotator()
    annotator.load(str(model_path))

    output_paths: list[Path] = []
    for recording in recordings:
        raw_signals = load_raw_signals(str(recording.edf_path), channel_labels)
        prepared = prepare_recording(
            raw_signals,
            sampling_rate_hz=sampling_rate_hz,
            time_resolution_s=time_resolution,
        )

        output_dir = recording.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = recording.edf_path.stem
        output_path = output_dir / f"{stem}_somnotate_predictions.parquet"
        sidecar_path = output_dir / f"{stem}_somnotate_segments.json"

        df = _score_prepared_recording(
            prepared,
            annotator,
            sampling_rate_hz=sampling_rate_hz,
        )
        df.to_parquet(output_path, index=False)
        with open(sidecar_path, "w") as f:
            json.dump(prepared.to_dict(), f, indent=2)

        if export_visbrain:
            hyp_path = output_dir / f"{stem}_somnotate_predictions.txt"
            states, intervals = convert_state_vector_to_state_intervals(
                df["label_model"].to_numpy(dtype=int),
                mapping=configuration.int_to_state,
                time_resolution=time_resolution,
            )
            export_hypnogram(str(hyp_path), states, intervals)

        output_paths.append(output_path)

    return output_paths


def _score_prepared_recording(
    prepared: PreparedRecording,
    annotator: StateAnnotator,
    sampling_rate_hz: float,
) -> pd.DataFrame:
    """Run the model on each scoring chunk and assemble a per-epoch DataFrame.

    Output covers the entire original recording in epoch time. Epochs that fall
    in ``gap`` or ``too_short`` segments are filled with the undefined label and
    zero probabilities.
    """
    time_res = prepared.time_resolution_s
    total_seconds = sum(s.duration_s for s in prepared.segments)
    n_epochs = int(round(total_seconds / time_res))

    label_model = np.full(n_epochs, UNDEFINED_MODEL_LABEL, dtype=int)
    label_output = np.full(n_epochs, UNDEFINED_OUTPUT_LABEL, dtype=int)
    segment_ids = np.full(n_epochs, -1, dtype=int)
    kinds = np.empty(n_epochs, dtype=object)
    kinds[:] = "gap"

    prob_columns: dict[str, np.ndarray] = {
        "prob_wake": np.zeros(n_epochs, dtype=float),
        "prob_nrem": np.zeros(n_epochs, dtype=float),
        "prob_rem": np.zeros(n_epochs, dtype=float),
        "prob_undef": np.zeros(n_epochs, dtype=float),
    }
    prob_key_to_column = {
        "W": "prob_wake",
        "N": "prob_nrem",
        "R": "prob_rem",
        "U": "prob_undef",
    }

    for seg in prepared.segments:
        start_ep = int(round(seg.original_start_s / time_res))
        stop_ep = int(round(seg.original_end_s / time_res))
        segment_ids[start_ep:stop_ep] = seg.segment_id
        kinds[start_ep:stop_ep] = seg.kind

    for chunk in prepared.scoring_chunks:
        preprocessed = preprocess_multichannel(chunk.raw_signal, sampling_rate_hz)
        chunk_predicted = np.abs(np.asarray(annotator.predict(preprocessed), dtype=int))
        chunk_probs = _predict_state_probabilities(annotator, preprocessed)

        chunk_start_ep = int(round(chunk.original_start_s / time_res))
        chunk_n = len(chunk_predicted)
        chunk_stop_ep = chunk_start_ep + chunk_n

        chunk_kinds = kinds[chunk_start_ep:chunk_stop_ep]
        signal_mask = chunk_kinds == "signal"
        if not np.any(signal_mask):
            continue

        chunk_output = np.fromiter(
            (MODEL_TO_OUTPUT_LABEL.get(int(v), UNDEFINED_OUTPUT_LABEL) for v in chunk_predicted),
            dtype=int,
            count=chunk_n,
        )

        label_model_view = label_model[chunk_start_ep:chunk_stop_ep]
        label_output_view = label_output[chunk_start_ep:chunk_stop_ep]
        label_model_view[signal_mask] = chunk_predicted[signal_mask]
        label_output_view[signal_mask] = chunk_output[signal_mask]
        label_model[chunk_start_ep:chunk_stop_ep] = label_model_view
        label_output[chunk_start_ep:chunk_stop_ep] = label_output_view

        for prob_key, state_int in PROBABILITY_JSON_KEYS.items():
            column = prob_key_to_column.get(prob_key)
            if column is None:
                continue
            chunk_vec = chunk_probs.get(state_int, np.zeros(chunk_n))
            chunk_vec = np.clip(chunk_vec.astype(float), 0.0, 1.0)
            prob_view = prob_columns[column][chunk_start_ep:chunk_stop_ep]
            prob_view[signal_mask] = chunk_vec[signal_mask]
            prob_columns[column][chunk_start_ep:chunk_stop_ep] = prob_view

    timepoints = np.arange(n_epochs, dtype=float) * time_res

    df = pd.DataFrame(
        {
            "time_s": timepoints,
            "label": label_output,
            "label_model": label_model,
            "label_output": label_output,
            "segment_id": segment_ids,
            "kind": kinds.astype(str),
            **prob_columns,
        }
    )
    return df


def _predict_state_probabilities(annotator: StateAnnotator, signal_array: np.ndarray) -> dict[int, np.ndarray]:
    transformed = annotator.transform(signal_array)
    probability_array = annotator.hmm.predict_proba([sample for sample in transformed])

    probability_dict: dict[int, np.ndarray] = {}
    for ii, state in enumerate(annotator.hmm.states):
        if state.distribution is None:
            continue
        probability_dict[int(state.name)] = probability_array[:, ii]

    return probability_dict
