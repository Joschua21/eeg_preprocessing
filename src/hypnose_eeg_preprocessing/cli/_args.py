"""Shared argument parsing for the CLI.

Selector arguments (subjects, dates, date ranges) are deliberately forgiving: the
same value can be written the way it appears in a directory name, the way it is
typed in a notebook, or the way it comes out of a shell loop. All of these mean
the same thing:

    --sub 66            --sub 066           --sub sub-066
    --sub 66,67,68      --sub 66 67 68      --subjids 66,67 --sub 68
    --date 20260707     --dates 20260707,20260708           --date 20260707 20260708
    --date-range 20260707,20260718          --date-range 20260707-20260718

The normalisation itself lives in `io.selectors` so the Python API and the CLI agree
on what a selector means; this module only wires it into argparse.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..io.selectors import (  # re-exported for callers that already import them here
    DATE_RE,
    parse_date_range,
    parse_dates,
    parse_subjects,
)

__all__ = [
    "DATE_RE",
    "add_repo_root_argument",
    "add_selector_arguments",
    "parse_date_range",
    "parse_dates",
    "parse_subjects",
    "resolve_selector",
]


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

    Selector problems surface as `argparse.ArgumentTypeError` so the calling command
    can hand them straight to `parser.error`.
    """
    try:
        subjects = parse_subjects(getattr(args, "sub", None))
        dates = parse_dates(getattr(args, "date", None)) or None
        date_range = parse_date_range(getattr(args, "date_range", None))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc

    if dates and date_range:
        raise argparse.ArgumentTypeError("Use either --date or --date-range, not both.")
    return subjects, dates, date_range


def add_repo_root_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--repo-root", type=Path, default=Path.cwd(), metavar="PATH",
        help="Repo root used to resolve the data location (default: cwd).",
    )
