"""Figure saving for the EEG derivatives tree.

Saving and styling both live in hypnose-behavior-analysis (`hypnose_behavior.io.save`) so that every
Hypnose repo emits figures that look the same. Only the *destination* differs: the
EEG derivatives tree is laid out as

    derivatives/sub-XXX/ses-YY_date-YYYYMMDD/figures/

whereas hypnose-behavior-analysis resolves against the behavioural tree. Rather than wrap
`save_figure`, this module registers `resolve_eeg_figure_dir` with hypnose-behavior-analysis
once at import; `save_figure` then routes here automatically and every other
argument behaves exactly as upstream.

hypnose-behavior-analysis is an optional dependency. Without it `save_figure` raises a clear
message pointing at the install step, and nothing else in the package is affected.
"""

from __future__ import annotations

from pathlib import Path

from .paths import _find_subject_dir, get_derivatives_root, normalize_subjid


def _coerce_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _session_dirs(subject_dir: Path, date: str | None) -> list[Path]:
    """Session directories under `subject_dir`, optionally filtered to one date."""
    if not subject_dir.exists():
        return []
    pattern = f"ses-*_date-{date}" if date else "ses-*_date-*"
    return sorted(p for p in subject_dir.glob(pattern) if p.is_dir())


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


_HYPNOSE_ANALYSIS_HINT = (
    "Saving figures requires hypnose-behavior-analysis (it owns the shared figure styles).\n"
    "Install it into this environment:  pip install -e /path/to/hypnose-behavior-analysis"
)


def _register_resolver() -> bool:
    """Point hypnose-behavior-analysis's save_figure at the EEG tree. True if registered."""
    try:
        from hypnose_behavior.io.save import set_figure_dir_resolver
    except ImportError:
        return False
    set_figure_dir_resolver(resolve_eeg_figure_dir)
    return True


_REGISTERED = _register_resolver()


def save_figure(fig, save_name: str, *, subjids, dates=None, **kwargs):
    """Save `fig` as a styled PDF into the EEG derivatives tree.

    Thin pass-through to hypnose-behavior-analysis's `save_figure` — every keyword it accepts
    (`subdir`, `fig_dir`, `dpi`, `bbox_inches`, `clear_legends`, `boxplot`) works
    here unchanged. Destination comes from `resolve_eeg_figure_dir` unless an
    explicit `fig_dir=` is passed.
    """
    try:
        from hypnose_behavior.io.save import save_figure as _save_figure
    except ImportError as exc:
        raise ImportError(_HYPNOSE_ANALYSIS_HINT) from exc

    if not _REGISTERED:
        # hypnose-behavior-analysis appeared after import time; register now.
        _register_resolver()

    return _save_figure(fig, save_name, subjids=subjids, dates=dates, **kwargs)
