"""Signal preprocessing and recording gap correction."""

from .gap_correction import (
    PreparedRecording,
    RecordingSegment,
    ScoringChunk,
    prepare_recording,
)
from .preprocessing import preprocess_multichannel

__all__ = [
    "PreparedRecording",
    "RecordingSegment",
    "ScoringChunk",
    "prepare_recording",
    "preprocess_multichannel",
]
