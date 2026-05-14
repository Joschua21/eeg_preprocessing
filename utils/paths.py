"""Path resolution and dataset discovery utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

from .config import DATA_SYMLINK

SUBJECT_RE = re.compile(r"^sub-(\d{3})")
SESSION_DATE_RE = re.compile(r"^ses-([^-_]+)_date-(\d{8})$")


@dataclass(frozen=True)
class RecordingRef:
    subject: str
    session: str
    date: str
    edf_path: Path
    output_dir: Path


def normalize_subjid(subjid: int | str) -> str:
    if isinstance(subjid, int):
        return f"sub-{subjid:03d}"
    subjid = str(subjid)
    if subjid.startswith("sub-"):
        return subjid
    if subjid.isdigit():
        return f"sub-{int(subjid):03d}"
    return f"sub-{subjid}"


def _iter_subject_dirs(raw_root: Path) -> Iterable[Path]:
    if not raw_root.exists():
        return []
    return [p for p in raw_root.iterdir() if p.is_dir() and SUBJECT_RE.match(p.name)]


def _find_subject_dir(raw_root: Path, subjid: str) -> Path | None:
    for subject_dir in _iter_subject_dirs(raw_root):
        if subject_dir.name.startswith(subjid):
            return subject_dir
    return None


def _parse_session_dir(session_dir: Path) -> tuple[str | None, str | None]:
    match = SESSION_DATE_RE.match(session_dir.name)
    if not match:
        return None, None
    session, date = match.groups()
    return f"ses-{session}", date


def _date_in_filter(date: str, date_list: Iterable[str] | None, date_range: tuple[str, str] | None) -> bool:
    if date_list:
        return date in date_list
    if date_range:
        start, end = date_range
        return start <= date <= end
    return True


def get_raw_root(repo_root: Path) -> Path:
    return (repo_root / DATA_SYMLINK / "rawdata").resolve()


def get_derivatives_root(repo_root: Path) -> Path:
    return (repo_root / DATA_SYMLINK / "derivatives").resolve()


def find_recordings(
    repo_root: Path,
    subjids: Iterable[int | str],
    dates: Iterable[int | str] | None = None,
    date_range: tuple[int | str, int | str] | None = None,
) -> list[RecordingRef]:
    raw_root = get_raw_root(repo_root)
    derivatives_root = get_derivatives_root(repo_root)

    normalized_dates = [str(d) for d in dates] if dates else None
    normalized_range = None
    if date_range:
        normalized_range = (str(date_range[0]), str(date_range[1]))

    results: list[RecordingRef] = []
    for subjid in subjids:
        sub_label = normalize_subjid(subjid)
        subject_dir = _find_subject_dir(raw_root, sub_label)
        if subject_dir is None:
            continue
        for session_dir in subject_dir.iterdir():
            if not session_dir.is_dir():
                continue
            session, date = _parse_session_dir(session_dir)
            if not session or not date:
                continue
            if not _date_in_filter(date, normalized_dates, normalized_range):
                continue
            ephys_dir = session_dir / "ephys"
            if not ephys_dir.exists():
                continue
            for edf_path in sorted(ephys_dir.glob(f"{sub_label}_ses-*recording-*.edf")):
                output_dir = (
                    derivatives_root
                    / subject_dir.name
                    / session_dir.name
                    / "saved_results"
                )
                results.append(
                    RecordingRef(
                        subject=sub_label,
                        session=session,
                        date=date,
                        edf_path=edf_path,
                        output_dir=output_dir,
                    )
                )

    return results
