"""Dataset discovery and data-location resolution."""

from .paths import (
    RecordingRef,
    find_recordings,
    get_derivatives_root,
    get_raw_root,
    normalize_subjid,
)

__all__ = [
    "RecordingRef",
    "find_recordings",
    "get_derivatives_root",
    "get_raw_root",
    "normalize_subjid",
]
