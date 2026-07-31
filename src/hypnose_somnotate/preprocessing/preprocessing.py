"""Preprocessing helpers built on the Somnotate pipeline."""

from __future__ import annotations

import numpy as np

from ..somnotate_pipeline.preprocessing.preprocess_signals import preprocess
from ..somnotate_pipeline.utils.configuration import time_resolution


def preprocess_multichannel(raw_signals: np.ndarray, sampling_rate_hz: float) -> np.ndarray:
    preprocessed_signals = []
    for signal in raw_signals.T:
        _, _, preprocessed_signal = preprocess(
            signal,
            sampling_rate_hz,
            time_resolution_in_sec=time_resolution,
            low_cut=1.0,
            high_cut=90.0,
            notch_low_cut=45.0,
            notch_high_cut=55.0,
        )
        preprocessed_signals.append(preprocessed_signal)

    return np.concatenate([signal.T for signal in preprocessed_signals], axis=1)
