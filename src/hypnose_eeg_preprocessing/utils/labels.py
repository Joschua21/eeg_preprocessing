"""Manual label handling and alignment utilities."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from ..config import MANUAL_TO_OUTPUT_LABEL


def load_manual_labels_csv(
    csv_path: Path,
    sampling_rate_hz: int,
) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    if "Timestamp" not in df.columns or "sleepStage" not in df.columns:
        raise ValueError("Manual CSV must include 'Timestamp' and 'sleepStage' columns.")

    timestamps = df["Timestamp"].to_numpy()
    labels = df["sleepStage"].to_numpy()

    if timestamps.size == 0:
        return np.array([], dtype=float), np.array([], dtype=int)

    seconds = (timestamps / float(sampling_rate_hz)).astype(int)
    max_second = int(seconds.max())

    output_labels = np.full(max_second + 1, fill_value=3, dtype=int)
    for second in range(max_second + 1):
        mask = seconds == second
        if not np.any(mask):
            continue
        stage_values = labels[mask]
        stage_values = np.array([MANUAL_TO_OUTPUT_LABEL.get(int(v), 3) for v in stage_values])
        counts = np.bincount(stage_values, minlength=4)
        output_labels[second] = int(np.argmax(counts))

    time_seconds = np.arange(0, len(output_labels), dtype=float)
    return time_seconds, output_labels
