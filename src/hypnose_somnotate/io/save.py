"""Figure saving for the EEG derivatives tree.

Saving and styling both live in hypnose-helpers (`hypnose_helpers.viz`) so that every
Hypnose repo emits figures that look the same. Only the *destination* differs: the
EEG derivatives tree is laid out as

    derivatives/sub-XXX/ses-YY_date-YYYYMMDD/figures/

whereas the behavioural repo resolves against its own tree. hypnose-helpers'
`save_figure` takes `fig_dir` as a plain argument, so this module simply resolves the EEG
destination and passes it in -- no resolver registration, no import-time global state.
"""

from __future__ import annotations

from pathlib import Path

from hypnose_helpers.viz.save import save_figure as _save_figure
from hypnose_helpers.io.layout import filter_sessions, list_sessions

from .paths import _find_subject_dir, get_derivatives_root, normalize_subjid


def _coerce_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _session_dirs(subject_dir: Path, date: str | None) -> list[Path]:
    """Session directories under `subject_dir`, optionally filtered to one date.

    Shares the family's layout walker (restructure_2 Phase 2b) but keeps this module's
    never-raise contract: `resolve_eeg_figure_dir` degrades to a coarser directory
    rather than failing a save, so an ambiguous or malformed tree must read as "no
    match" here rather than propagating.
    """
    if not subject_dir.exists():
        return []
    try:
        sessions = list_sessions(subject_dir)
        return [s.path for s in filter_sessions(sessions, date=date)]
    except (ValueError, OSError):
        return []


def resolve_eeg_figure_dir(subjids, dates=None) -> Path:
    """Where figures for this subject/date scope belong in the EEG derivatives tree.

    Mirrors hypnose-behavior-analysis's scoping rules so figures land in a predictable place
    regardless of which repo produced them:

    - several subjects           -> derivatives/figures
    - one subject, several dates -> derivatives/sub-XXX/figures
    - one subject, one date      -> derivatives/sub-XXX/ses-YY_date-…/figures

    Falls back to the next level up whenever a directory cannot be resolved (an
    unknown subject, or a date with no matching session), so saving a figure never
    fails just because the tree is not laid out as expected.
    """
    derivatives_root = Path(get_derivatives_root())
    subj_list = _coerce_list(subjids)
    date_list = _coerce_list(dates)

    if len(subj_list) != 1:
        # zero or many subjects -> dataset-level figures directory
        return derivatives_root / "figures"

    sub_label = normalize_subjid(subj_list[0])
    subject_dir = _find_subject_dir(derivatives_root, sub_label)
    if subject_dir is None:
        return derivatives_root / "figures"

    if len(date_list) != 1:
        return subject_dir / "figures"

    session_dirs = _session_dirs(subject_dir, str(date_list[0]))
    if not session_dirs:
        return subject_dir / "figures"
    return session_dirs[0] / "figures"


def save_figure(fig, save_name: str, *, subjids, dates=None, **kwargs):
    """Save `fig` as a styled PDF into the EEG derivatives tree.

    Thin pass-through to hypnose-helpers' `save_figure` -- every keyword it accepts
    (`subdir`, `dpi`, `bbox_inches`, `clear_legends`, `boxplot`) works here unchanged.
    Destination comes from `resolve_eeg_figure_dir` unless an explicit `fig_dir=` is given.
    """
    kwargs.setdefault("fig_dir", resolve_eeg_figure_dir(subjids, dates))
    return _save_figure(fig, save_name, subjids=subjids, dates=dates, **kwargs)
