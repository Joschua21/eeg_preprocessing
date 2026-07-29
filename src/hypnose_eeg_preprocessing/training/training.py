"""Training workflow wrappers for Somnotate."""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from ..somnotate._automated_state_annotation import StateAnnotator
from ..somnotate_pipeline.utils import configuration
from ..somnotate_pipeline.io.data_io import load_state_vector, load_raw_signals
from ..somnotate_pipeline.preprocessing.mat_to_csv import mat_to_signal_tables
from ..somnotate_pipeline.processing.adjust_delimiters import adjust_delimiters_in_txt_files

from ..config import DEFAULT_SAMPLING_RATE_HZ, DEFAULT_SLEEP_STAGE_RESOLUTION_S
from ..io.paths import get_derivatives_root
from ..preprocessing.preprocessing import preprocess_multichannel


def _generate_edf_and_visbrain(csv_dir: Path, edf_dir: Path, ann_dir: Path, sampling_rate_hz: float) -> list[Path]:
    import pyedflib

    edf_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    from ..testing.testing import list_csv_files, load_signal_table

    edf_paths: list[Path] = []
    for csv_path in list_csv_files(csv_dir):
        df = load_signal_table(csv_path)
        if not {"EEG1", "EEG2", "EMG", "sleepStage"}.issubset(df.columns):
            raise ValueError(f"Missing required columns in {csv_path}")

        eeg1_data = df["EEG1"].to_numpy()
        eeg2_data = df["EEG2"].to_numpy()
        emg_data = df["EMG"].to_numpy()
        all_data = np.array([eeg1_data, eeg2_data, emg_data])

        edf_path = edf_dir / f"output_{csv_path.stem}.edf"
        with pyedflib.EdfWriter(str(edf_path), len(all_data), file_type=pyedflib.FILETYPE_EDFPLUS) as writer:
            labels = ["EEG1", "EEG2", "EMG"]
            for i, label in enumerate(labels):
                signal_info = {
                    "label": label,
                    "dimension": "uV",
                    "sample_frequency": sampling_rate_hz,
                    "physical_min": float(np.min(all_data[i])),
                    "physical_max": float(np.max(all_data[i])),
                    "digital_min": -32768,
                    "digital_max": 32767,
                    "transducer": "",
                    "prefilter": "",
                }
                writer.setSignalHeader(i, signal_info)
            writer.writeSamples(all_data)

        ann_path = ann_dir / f"annotations_visbrain_{csv_path.stem}.txt"
        _export_visbrain_from_sleep_stage(df["sleepStage"].to_numpy(), ann_path, sampling_rate_hz)
        edf_paths.append(edf_path)

    return edf_paths


def _export_visbrain_from_sleep_stage(stage_values: np.ndarray, output_path: Path, sampling_rate_hz: float) -> None:
    annotations = []
    current_stage = None
    start_time = 0.0

    for i, label in enumerate(stage_values):
        current_time = i / sampling_rate_hz
        if label != current_stage:
            if current_stage is not None:
                annotations.append((start_time, current_time, current_stage))
            current_stage = label
            start_time = current_time
    annotations.append((start_time, len(stage_values) / sampling_rate_hz, current_stage))

    last_time_value = annotations[-1][1] if annotations else 0.0
    stage_label_map = {
        1: "awake",
        2: "non-REM",
        3: "REM",
        4: "undefined",
    }

    with open(output_path, "w") as f:
        f.write(f"*Duration_sec    {last_time_value}\n")
        f.write("*Datafile\tUnspecified\n")
        for _, end, stage in annotations:
            stage_label = stage_label_map.get(int(stage), "undefined")
            f.write(f"{stage_label}    {end}\n")


def run_training(
    model_name: str,
    training_mat_dir: Path,
    repo_root: Path,
    sampling_rate_hz: int = DEFAULT_SAMPLING_RATE_HZ,
    sleep_stage_resolution_s: int = DEFAULT_SLEEP_STAGE_RESOLUTION_S,
    show_plots: bool = False,
) -> Path:
    if not model_name:
        raise ValueError("model_name is required")

    derivatives_root = get_derivatives_root(repo_root)
    if not derivatives_root.exists():
        raise FileNotFoundError(
            f"Derivatives root not found at {derivatives_root}. Create a symlink named 'derivatives' in the repo."
        )

    output_root = derivatives_root / "somnotate_training" / model_name
    intermediate_dir = output_root / "intermediate"
    csv_dir = intermediate_dir / "csv"
    edf_dir = intermediate_dir / "edfs"
    ann_dir = intermediate_dir / "annotations"
    preprocessed_dir = intermediate_dir / "preprocessed"
    figures_dir = output_root / "figures"
    results_dir = output_root / "results"

    for path in [csv_dir, edf_dir, ann_dir, preprocessed_dir, figures_dir, results_dir]:
        path.mkdir(parents=True, exist_ok=True)

    mat_to_signal_tables(str(training_mat_dir), str(csv_dir), sampling_rate_hz, sleep_stage_resolution_s)

    edf_paths = _generate_edf_and_visbrain(csv_dir, edf_dir, ann_dir, sampling_rate_hz)
    adjust_delimiters_in_txt_files(str(ann_dir))

    signal_arrays = []
    state_vectors = []
    for edf_path in edf_paths:
        raw_signals = load_raw_signals(str(edf_path), ["EEG1", "EEG2", "EMG"])
        preprocessed = preprocess_multichannel(raw_signals, sampling_rate_hz)
        np.save(preprocessed_dir / f"{edf_path.stem}.npy", preprocessed)

        hyp_path = ann_dir / f"annotations_visbrain_{edf_path.stem.replace('output_', '')}.txt"
        state_vector = load_state_vector(
            str(hyp_path),
            mapping=configuration.state_to_int,
            time_resolution=configuration.time_resolution,
        )

        min_len = min(len(preprocessed), len(state_vector))
        signal_arrays.append(preprocessed[:min_len])
        state_vectors.append(state_vector[:min_len])

    recording_names = [edf_path.stem for edf_path in edf_paths]
    _evaluate_holdout(
        signal_arrays,
        state_vectors,
        recording_names,
        results_dir,
        figures_dir,
        show_plots,
    )

    annotator = StateAnnotator()
    annotator.fit(signal_arrays, state_vectors)
    model_path = output_root / "model.pickle"
    annotator.save(str(model_path))

    metadata = {
        "model_name": model_name,
        "sampling_rate_hz": sampling_rate_hz,
        "sleep_stage_resolution_s": sleep_stage_resolution_s,
        "time_resolution_s": configuration.time_resolution,
    }
    with open(output_root / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return model_path


def _evaluate_holdout(
    signal_arrays: list[np.ndarray],
    state_vectors: list[np.ndarray],
    recording_names: list[str],
    results_dir: Path,
    figures_dir: Path,
    show_plots: bool,
) -> None:
    total_datasets = len(signal_arrays)
    if total_datasets < 2:
        return

    accuracies = []
    confusion_all = None

    for ii in range(total_datasets):
        training_signals = [arr for jj, arr in enumerate(signal_arrays) if jj != ii]
        training_states = [vec for jj, vec in enumerate(state_vectors) if jj != ii]

        annotator = StateAnnotator()
        annotator.fit(training_signals, training_states)

        predicted = annotator.predict(signal_arrays[ii])
        truth = np.abs(state_vectors[ii])
        accuracies.append(float(np.mean(predicted == truth)))

        cm = confusion_matrix(truth, predicted, labels=[0, 1, 2, 3])
        confusion_all = cm if confusion_all is None else confusion_all + cm

    np.savez(results_dir / "evaluation.npz", accuracy=np.array(accuracies), confusion=confusion_all)
    if len(recording_names) == len(accuracies):
        pd.DataFrame(
            {
                "recording": recording_names,
                "accuracy": accuracies,
            }
        ).to_csv(results_dir / "evaluation_by_recording.csv", index=False)

    if confusion_all is not None:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(confusion_all, cmap="Blues")
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(["U", "W", "N", "R"])
        ax.set_yticklabels(["U", "W", "N", "R"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Manual")
        totals = confusion_all.sum(axis=1, keepdims=True)
        confusion_pct = np.divide(
            confusion_all,
            totals,
            out=np.zeros_like(confusion_all, dtype=float),
            where=totals != 0,
        )
        for i in range(confusion_pct.shape[0]):
            for j in range(confusion_pct.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{confusion_pct[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                )

        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(figures_dir / "confusion_matrix.png")
        if show_plots:
            plt.show()
        plt.close(fig)
