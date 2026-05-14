"""Scoring workflow for Somnotate models."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from utils.somnotate._automated_state_annotation import StateAnnotator
from utils.somnotate._utils import convert_state_vector_to_state_intervals
from utils.somnotate_pipeline.configuration import time_resolution
from utils.somnotate_pipeline.data_io import load_raw_signals, export_hypnogram

from .config import DEFAULT_CHANNEL_LABELS, MODEL_TO_OUTPUT_LABEL, PROBABILITY_JSON_KEYS
from .paths import find_recordings, get_derivatives_root
from .preprocessing import preprocess_multichannel
from .somnotate_pipeline import configuration


def score_recordings(
    subjids: list[int | str],
    model_path: Path,
    repo_root: Path,
    dates: list[int | str] | None = None,
    date_range: tuple[int | str, int | str] | None = None,
    channel_labels: list[str] | None = None,
    export_visbrain: bool = True,
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
        preprocessed = preprocess_multichannel(raw_signals, sampling_rate_hz=512)

        predicted = np.array(annotator.predict(preprocessed), dtype=int)
        predicted = np.abs(predicted)
        output_labels = np.array([MODEL_TO_OUTPUT_LABEL.get(int(v), 3) for v in predicted], dtype=int)

        probabilities = _predict_state_probabilities(annotator, preprocessed)
        timepoints = np.arange(0, len(output_labels) * time_resolution, time_resolution, dtype=float)

        output_dir = recording.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{recording.edf_path.stem}_somnotate_predictions.parquet"

        prob_df = _format_probability_frame(probabilities)
        df = pd.DataFrame({
            "time_s": timepoints,
            "label": output_labels,
            "label_model": predicted,
            "label_output": output_labels,
        })
        df = pd.concat([df, prob_df], axis=1)
        df.to_parquet(output_path, index=False)
        if export_visbrain:
            states, intervals = convert_state_vector_to_state_intervals(
                predicted,
                mapping=configuration.int_to_state,
                time_resolution=time_resolution,
            )
            hyp_path = output_dir / f"{recording.edf_path.stem}_somnotate_predictions.txt"
            export_hypnogram(str(hyp_path), states, intervals)
        output_paths.append(output_path)

    return output_paths


def _predict_state_probabilities(annotator: StateAnnotator, signal_array: np.ndarray) -> dict[int, np.ndarray]:
    transformed = annotator.transform(signal_array)
    probability_array = annotator.hmm.predict_proba([sample for sample in transformed])

    probability_dict: dict[int, np.ndarray] = {}
    for ii, state in enumerate(annotator.hmm.states):
        if state.distribution is None:
            continue
        probability_dict[int(state.name)] = probability_array[:, ii]

    return probability_dict


def _format_probability_frame(probability_dict: dict[int, np.ndarray]) -> pd.DataFrame:
    length = next(iter(probability_dict.values())).shape[0] if probability_dict else 0
    prob_matrix = {}
    for key, state in PROBABILITY_JSON_KEYS.items():
        values = probability_dict.get(state, np.zeros(length))
        prob_matrix[key] = np.clip(values.astype(float), 0.0, 1.0)

    return pd.DataFrame({
        "prob_wake": prob_matrix.get("W", np.zeros(length)),
        "prob_nrem": prob_matrix.get("N", np.zeros(length)),
        "prob_rem": prob_matrix.get("R", np.zeros(length)),
        "prob_undef": prob_matrix.get("U", np.zeros(length)),
    })
