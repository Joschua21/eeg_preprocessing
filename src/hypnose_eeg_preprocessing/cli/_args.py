"""Shared argument parsing for the CLI.

Selector arguments (subjects, dates, date ranges) are deliberately forgiving: the
same value can be written the way it appears in a directory name, the way it is
typed in a notebook, or the way it comes out of a shell loop. All of these mean
the same thing:

    --sub 66            --sub 066           --sub sub-066
    --sub 66,67,68      --sub 66 67 68      --subjids 66,67 --sub 68
    --date 20260707     --dates 20260707,20260708           --date 20260707 20260708
    --date-range 20260707,20260718          --date-range 20260707-20260718

Parsing lives here rather than in argparse callbacks so the same normalisation is
reachable from Python (and testable) without constructing a parser.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

# A date as it appears in session directory names.
DATE_RE = re.compile(r"^\d{8}$")

# Accepted separators inside a single token: "66,67" and "66;67". Whitespace
# separation is handled by argparse's nargs="+", shell word splitting, or here.
_SPLIT_RE = re.compile(r"[,;\s]+")

# "20260707-20260718" — only valid for date ranges, since a bare date never
# contains a hyphen. Kept separate from _SPLIT_RE so "sub-066" is not split.
_RANGE_SPLIT_RE = re.compile(r"[,;\s]+|(?<=\d)-(?=\d)")


def _flatten(values) -> list[str]:
    """Split every token on commas/semicolons/whitespace and drop empties."""
    if values is None:
        return []
    if isinstance(values, (str, int)):
        values = [values]
    out: list[str] = []
    for value in values:
        for part in _SPLIT_RE.split(str(value).strip()):
            if part:
                out.append(part)
    return out


def parse_subjects(values) -> list[int]:
    """Normalise subject arguments to plain integers.

    Accepts "66", "066", "sub-066", and any comma/space separated combination.
    Integers are returned (not "sub-066" labels) because that is what
    `find_recordings` and `save_figure` both want; `normalize_subjid` turns them
    back into labels where needed.

    Duplicates are removed, order of first appearance is preserved.
    """
    subjects: list[int] = []
    for token in _flatten(values):
        cleaned = token.lower()
        if cleaned.startswith("sub-"):
            cleaned = cleaned[4:]
        if not cleaned.isdigit():
            raise argparse.ArgumentTypeError(
                f"Invalid subject {token!r}; expected a number like 66, 066 or sub-066."
            )
        subject = int(cleaned)
        if subject not in subjects:
            subjects.append(subject)
    return subjects


def parse_dates(values) -> list[str]:
    """Normalise date arguments to a list of YYYYMMDD strings.

    Strings rather than ints, because that is the form session directories and
    `find_recordings` use. Duplicates removed, first-appearance order preserved.
    """
    dates: list[str] = []
    for token in _flatten(values):
        if not DATE_RE.match(token):
            raise argparse.ArgumentTypeError(
                f"Invalid date {token!r}; expected YYYYMMDD (e.g. 20260707)."
            )
        if token not in dates:
            dates.append(token)
    return dates


def parse_date_range(value) -> tuple[str, str] | None:
    """Parse an inclusive date range into (start, end).

    Accepts "20260707,20260718" and "20260707-20260718". The bounds are sorted, so
    an inverted range still selects the intended span rather than nothing.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        parts = [str(v).strip() for v in value]
    else:
        parts = [p for p in _RANGE_SPLIT_RE.split(str(value).strip()) if p]

    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Invalid date range {value!r}; expected START,END or START-END "
            "(e.g. 20260707,20260718 or 20260707-20260718)."
        )
    for part in parts:
        if not DATE_RE.match(part):
            raise argparse.ArgumentTypeError(
                f"Invalid date {part!r} in range {value!r}; expected YYYYMMDD."
            )
    start, end = sorted(parts)
    return start, end


def add_selector_arguments(parser: argparse.ArgumentParser, *, subject_required: bool = True) -> None:
    """Add the --sub / --date / --date-range family to a parser.

    `--subjid` and `--subjids` are accepted as aliases of `--sub`, and `--dates` of
    `--date`, so whichever name you reach for works.
    """
    parser.add_argument(
        "--sub", "--subjid", "--subjids",
        dest="sub", nargs="+", required=subject_required, metavar="ID",
        help="Subject id(s): 66, 066 or sub-066; comma- or space-separated.",
    )
    parser.add_argument(
        "--date", "--dates",
        dest="date", nargs="+", metavar="YYYYMMDD",
        help="Session date(s), comma- or space-separated. Default: all dates.",
    )
    parser.add_argument(
        "--date-range",
        dest="date_range", metavar="START,END",
        help="Inclusive date range, START,END or START-END.",
    )


def resolve_selector(args) -> tuple[list[int], list[str] | None, tuple[str, str] | None]:
    """Turn parsed selector arguments into (subjects, dates, date_range).

    `dates` is None (not []) when unset, since `find_recordings` treats None as
    "all dates" and an empty list as a filter matching nothing.
    """
    subjects = parse_subjects(getattr(args, "sub", None))
    dates = parse_dates(getattr(args, "date", None)) or None
    date_range = parse_date_range(getattr(args, "date_range", None))

    if dates and date_range:
        raise argparse.ArgumentTypeError(
            "Use either --date or --date-range, not both."
        )
    return subjects, dates, date_range


def add_repo_root_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--repo-root", type=Path, default=Path.cwd(), metavar="PATH",
        help="Repo root used to resolve the data location (default: cwd).",
    )
