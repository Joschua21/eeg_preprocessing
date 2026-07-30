"""Normalisation of subject/date selectors.

One definition of what `66`, `"066"` and `"sub-066"` mean, shared by the Python API
and the CLI, so a selector behaves identically wherever it is written. Lives in `io`
rather than `cli` because the loaders need it and the dependency runs one way:
`cli` imports from `io`, never the reverse.

Raises `ValueError` on bad input; the CLI converts that into an argparse error.
"""

from __future__ import annotations

import re

# A date as it appears in session directory names.
DATE_RE = re.compile(r"^\d{8}$")

# Separators inside a single token: "66,67" and "66;67". Whitespace-separated values
# arrive already split (by the shell, or by argparse nargs="+").
_SPLIT_RE = re.compile(r"[,;\s]+")

# "20260707-20260718" — only valid for ranges, since a bare date never contains a
# hyphen. Kept separate from _SPLIT_RE so "sub-066" is not split on its hyphen.
_RANGE_SPLIT_RE = re.compile(r"[,;\s]+|(?<=\d)-(?=\d)")


def flatten(values) -> list[str]:
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

    Accepts 66, "66", "066", "sub-066", and any comma/space separated combination of
    those. Integers are returned because that is what `find_recordings` and
    `save_figure` both want. Duplicates removed, first-appearance order preserved.
    """
    subjects: list[int] = []
    for token in flatten(values):
        cleaned = token.lower()
        if cleaned.startswith("sub-"):
            cleaned = cleaned[4:]
        if not cleaned.isdigit():
            raise ValueError(
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
    for token in flatten(values):
        if not DATE_RE.match(token):
            raise ValueError(f"Invalid date {token!r}; expected YYYYMMDD (e.g. 20260707).")
        if token not in dates:
            dates.append(token)
    return dates


def parse_date_range(value) -> tuple[str, str] | None:
    """Parse an inclusive date range into (start, end).

    Accepts "20260707,20260718", "20260707-20260718", and a 2-element sequence. The
    bounds are sorted, so an inverted range still selects the intended span.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        parts = [str(v).strip() for v in value]
    else:
        parts = [p for p in _RANGE_SPLIT_RE.split(str(value).strip()) if p]

    if len(parts) != 2:
        raise ValueError(
            f"Invalid date range {value!r}; expected START,END or START-END "
            "(e.g. 20260707,20260718 or 20260707-20260718)."
        )
    for part in parts:
        if not DATE_RE.match(part):
            raise ValueError(f"Invalid date {part!r} in range {value!r}; expected YYYYMMDD.")
    start, end = sorted(parts)
    return start, end
